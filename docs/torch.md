# PyTorch checkpoints in the ledger

`vestigo.torch` makes every checkpoint a PyTorch run saves an accepted release of the
ledger. You keep training with
[torch-checkpointing](https://pypi.org/project/torch-checkpointing/)'s
`CheckpointManager`; one call points it at the ledger. In return you get:

- runs found again by what they compute, resumed after a crash in a new attempt that
  locks out the old process;
- branches that start a new run from any accepted checkpoint;
- named heads, such as the best epoch so far;
- reads that serve only accepted checkpoints and check every byte against its digest;
- metadata on SQLite or Postgres, bytes on local disk, S3, GCS or Azure;
- one checkpoint across every rank of a distributed job.

```bash
uv add "vestigo[torch]"   # torch and the pinned torch-checkpointing
```

The scripts in [`examples/torch/`](../examples/torch/) run everything below, and the
suite runs them too (`tests/test_torch.py`, "The examples").

## The pieces

| Term | What it is | How you get it |
| --- | --- | --- |
| model | a node: the module's class and `state_dict` schema (every key's shape and dtype) under a version you name | `ledger.define_model(model, version=...)` |
| program | a node over the model: its loss, optimizer class and defaults, and any config you declare | `ledger.define_program(model_def, loss=..., optimizer=..., config=...)` |
| execution | one run of a program on concrete inputs (a build), found again by its computation key | `ledger.start(program, inputs)` |
| attempt | one process's hold on an execution, with a lease a thread renews | each `start`, `resume`, `branch` |
| checkpoint | one step's files from every rank, accepted together as one release | `manager.save(run.checkpoint_id(step), ...)` |

Weights are not part of a model's identity: a retrained model keeps its node. Source is
not hashed either; bump `version` when behaviour changes. Two programs that differ in
loss, optimizer or config are two programs over one model.

`inputs` maps a role to a concrete version, such as `{"train": "vestigo://artifact/ds@v=3"}`
or a table at a snapshot. The program, inputs and starting release make the
computation key, and that key is how a run is found again.

## Train and checkpoint

```python
from torch_checkpointing.checkpoint_manager import CheckpointManager
from vestigo.torch import Ledger

with Ledger("ledger") as ledger:
    program = ledger.define_program(
        ledger.define_model(model, version="regression-v1"),
        loss=nn.MSELoss(),
        optimizer=optimizer,
        config={"batch": 32},
    )
    with ledger.start(program, {"train": "synthetic-regression@seed=1"}) as run:
        manager = run.configure(CheckpointManager.Config.with_sync_save()).build()
        try:
            for epoch in range(1, epochs + 1):
                train_one_epoch()
                manager.save(
                    run.checkpoint_id(epoch),
                    {"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch},
                )
        finally:
            manager.close()
        run.finish()
```

`run.configure` sets four fields on the config and returns it:

| Field | Set to | Why |
| --- | --- | --- |
| `storage_config` | `LedgerStorageConfig` | every file goes into the ledger's blob store as a `building` version, created before its bytes |
| `pre_finalize_callback` | `vestigo.torch.report` | each rank records what it wrote, before the barrier |
| `finalize_callback` | the run's `Checkpoints` | after the barrier, rank 0 accepts the step as one release, only while the run holds its lease |
| `save.writer_config.write_to_temp_dir` | `False` | the ledger is the commit point, so nothing is renamed |

`run.checkpoint_id(step)` is a name under the attempt, such as
`vestigo-torch/<execution>/<attempt>/step-5`; it names no file on any disk. Once a
synchronous save returns, `run.release_of(step)` is the accepted release.

## Resume after a crash

Find the unfinished execution of the same computation and resume it:

```python
execution = ledger.find(program, inputs)
run = ledger.resume(execution) if execution else ledger.start(program, inputs)
with run:
    manager = run.configure(CheckpointManager.Config.with_sync_save()).build()
    checkpoint = run.restore()  # None for a fresh start
    if checkpoint is not None:
        state = manager.load(
            checkpoint,
            into={"model": model.state_dict(), "optim": None, "epoch": None},
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        start = state["epoch"]
```

`resume` opens a new attempt from the execution's latest accepted checkpoint, and
`restore()` names it. Pass `from_release=` to start from an older one. `into` must
name every item you want back: a live `state_dict` is loaded into in place, and
`None` reads the item as it was saved.

How the old process ended decides when you can resume:

- **It left the `with` block,** by an exception or `SystemExit`, with at least one
  checkpoint. It dropped its lease, so a resume can start at once.
- **It was killed.** Its lease holds until `lease_seconds` (60 by default) after its
  last renewal. Until then `resume` refuses with "another attempt's lease on it is
  live". Pass `Ledger(..., lease_seconds=...)` to shorten the wait.
- **It ended before any checkpoint.** The execution failed, and `find` no longer
  returns it, so the next run starts fresh.

Once a new attempt holds the execution, the old process is fenced. Its saves are
refused, and so are its `set_head`, `observe` and `finish` calls.

`finish()` completes the execution, and nothing resumes it after that. `find` returns
only unfinished executions, so the same computation started again after a finish is a
new execution.

## Branch

Start a new execution from any accepted release, of this run or another:

```python
with ledger.branch(finetune_program, release_id, inputs) as run:
    manager = run.configure(CheckpointManager.Config.with_sync_save()).build()
    state = manager.load(run.restore(), into={"model": model.state_dict()})
```

The release is part of the branch's computation. A branch never dedups against the run
it came from, or against another branch of the same release.

## Heads

`run.set_head(name, release_id)` points a reference at an accepted release, and
`ledger.releases.reference(name)` returns `(release_id, revision)`. Names are global to
the ledger, so give each one a prefix (`regression/best`). A head keeps its release from
garbage collection.

## Asynchronous saves

`CheckpointManager.Config.with_async_save()` works unchanged. The writer runs in a
subprocess, so the storage and callbacks are picklable descriptions of the ledger, and
each one opens its own connection. `save` returns a future, and the step's release exists
once that future completes. Call `.result()` on it before `release_of`;
`manager.close()` waits for the last write.

On macOS, torch-checkpointing 0.1.1's async stager fails whenever MPS is available,
because it calls CUDA device functions. That is an upstream bug. Synchronous saves are
unaffected.

## Several ranks

Rank 0 owns the run. The other ranks get its `Checkpoints` (and the checkpoint to
restore) by broadcast, then configure their managers the same way:

```python
shared = [None, None]
if rank == 0:
    run = ledger.start(program, inputs)  # or resume
    shared = [run.checkpoints, run.restore()]
dist.broadcast_object_list(shared, src=0)
checkpoints, checkpoint = shared
manager = checkpoints.configure(CheckpointManager.Config.with_sync_save()).build()
```

Each rank writes its own files (`model_0.pt`, `model_1.pt`, ...) and loads its own back.
Rank 0 accepts a step only after the barrier has seen every rank write. If a rank dies
before writing, the barrier times out, and that step never becomes a release.

The ranks find their rank and world size from the default process group. In the async
writer subprocess they read it from `RANK` and `WORLD_SIZE`.

## Where the ledger lives

`Ledger("ledger")` keeps SQLite metadata and local bytes under one directory. That is
enough for one machine, any number of ranks. Ranks on several machines need a ledger
they can all reach:

```python
from vestigo.blob_store import BlobSettings
from vestigo.torch import Ledger, RemoteBlobs

ledger = Ledger(
    "ledger",
    dsn="postgresql://user@db/ledger",
    blobs=RemoteBlobs("s3", "my-bucket", settings=BlobSettings(s3_region="us-east-1")),
)
```

`RemoteBlobs` takes `"s3"`, `"gcs"` or `"azure"` (on Azure, `bucket` is the container), and
`BlobSettings` fields left unset fall back to the provider's ambient credentials.
Each backend needs its extra: `vestigo[torch,postgres,s3]`, or `gcs` or `azure` in
place of `s3`.

## Record an exported program

```python
exported = torch.export.export(model, (example,), dynamic_shapes=({0: torch.export.Dim("batch")},))
run.observe(exported, scope="model_forward", coverage="complete_under_declared_constraints")
ledger.observations(run.execution_id)
```

`observe` stores the PT2 archive as a version of the execution. Its metadata records
the archive, schema and opset versions, the torch version and the range constraints.
`scope` (what the graph is of) and `coverage` (how much of it) are recorded but never
interpreted.

An observation never changes a node, so the model's and program's ids stay the same.
The same program exported again under the same scope and coverage is one version: the
comparison covers its graph, weights and constants, not its example inputs or the zip's
timestamps. Like a save, `observe` needs the lease.

## What the storage refuses

`LedgerStorage` is a ledger, not a filesystem:

- It serves only accepted checkpoints. A file that was written but never accepted
  (its step crashed, or its release was rejected) reads as missing.
- A file whose bytes no longer match the digest the ledger accepted fails the load
  with `OSError`.
- An attempt cannot write under another attempt's checkpoint id.
- `delete`, `rmdir` and `rename` raise. Old checkpoints go through retention and
  garbage collection.

## Limits

- torch-checkpointing is pinned at 0.1.1. It is pre-1.0, and its `Storage` interface
  and callback shapes have changed before.
- A load holds each file in memory once while it checks the file's digest.
- Sharded state (DTensor, FSDP) goes through torch-checkpointing's own resharding.
  vestigo stores whatever files it writes.
