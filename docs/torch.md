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
- signed checkpoints that a separate job verifies before it loads or ships them;
- metadata on SQLite or Postgres, bytes on local disk, S3, GCS or Azure;
- one checkpoint across every rank of a distributed job.

```bash
uv add "vestigo[torch]"   # torch and the pinned torch-checkpointing
uv add "vestigo[torch,sign]"  # and Ed25519 signatures on checkpoints
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
new execution. The checkpoint the run ended on (its attempt's last, or the one it
restored if it saved none) stays under the reference `vestigo-torch/<execution>`, and
`ledger.final_checkpoint(execution)` returns that release.

## Branch

Start a new execution from any accepted checkpoint, of this run or another:

```python
with ledger.branch(finetune_program, release_id, inputs) as run:
    manager = run.configure(CheckpointManager.Config.with_sync_save()).build()
    state = manager.load(run.restore(), into={"model": model.state_dict()})
```

The release is part of the branch's computation, so a branch never dedups against the
run it came from. Two branches of one release with the same program and inputs do share
a computation key: each `branch` call opens an execution of its own, and
`ledger.find(program, inputs, init=release_id)` finds the newest unfinished one.

`branch` and `resume(from_release=)` refuse a release that is not a checkpoint (a final
or derived release) and a checkpoint `collect` has taken, before opening anything.

## Heads

`run.set_head(name, release_id)` points a reference at an accepted release, and
`ledger.releases.reference(name)` returns `(release_id, revision)`. Names are global to
the ledger, so give each one a prefix (`regression/best`). A head keeps its release
through every collection.

## Derived releases

An export, evaluation metrics or an inference package is computed once a checkpoint
has been accepted, so it cannot join that checkpoint's release. It becomes a release
of its own, derived from it:

```python
evaluation = ledger.derive(
    release_id,
    "eval",
    {"metrics": json.dumps(metrics).encode(), "weights": Path("export.safetensors")},
)
ledger.derived(release_id, "eval")  # the same ReleaseBundle, found again
```

`outputs` maps a label to bytes or a file. The outputs become current together, as
the final release of a small build of their own. Each output's spec names the
checkpoint it came from (`params["from"]`), and that is the only edge. It is not an
input, so keeping a derived release (a head, a pin) keeps only its own outputs, never
the optimizer state of the checkpoint behind it. `collect` leaves derived releases
alone, and `derived` still finds one after its checkpoint is gone.

Deriving the same name from the same checkpoint again returns the release already
there and stores nothing. The outputs' ids are per execution and name
(`torch_<execution>@derived=eval@output=metrics`), so a derivation from a later
checkpoint becomes their current value, and the earlier versions stay readable until
`garbage_collect(keep_superseded=)` trims them. `train.py` derives an evaluation of
the checkpoint it finished on.

## Keep the latest checkpoints

Every save stores a full copy of what you pass it, and nothing is deleted until you
collect:

```python
ledger.collect(keep=2)
```

`collect` runs over the whole ledger. It keeps:

- of each unfinished execution, its newest `keep` checkpoints and the one its attempt
  resumed from;
- of each finished execution, the checkpoint it ended on;
- the release each unfinished branch starts from;
- every release a head points at.

Every other checkpoint's files, rank reports and manifest are deleted. So are the files
a fenced or ended attempt staged but never released (a step that crashed mid-save). An
attempt that still holds its execution keeps everything it is writing, so you can
collect while runs train. `examples/torch/train.py --keep 1` collects after every epoch.

`collect` returns the store's counts (`deleted_count`, `deleted_bytes`) plus `orphaned`,
the staged files of ended attempts it failed. `dry_run=True` deletes nothing and lists
what it would take under `collected`. Loading a checkpoint, or branching from it,
counts as a use, and `min_idle_seconds=` spares whatever was used that recently. Pass
it when branches may open while `collect` runs, since a branch opened after `collect`
has read the executions is not yet among the ones it keeps.

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

Each rank reads its rank and the world size when it calls `configure`, from the default
process group, else from `RANK` and `WORLD_SIZE`, so configure after
`init_process_group`. The async writer subprocess uses what `configure` read: it has no
process group, and when the group was given its rank directly (as with `mp.spawn`), no
such variables either.

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

## Signed checkpoints

torch-checkpointing deserializes every file with `torch.load(..., weights_only=False)`
and its metadata with `pickle`. Loading a tampered checkpoint therefore runs whatever
the tamperer put in it, and a subtler change (a shifted weight) loads and serves
without anyone noticing. Signing lets a job that loads checkpoints prove that the
training job wrote them, whoever else can write the stores.

### What each check stops

Every load checks each file's sha256 before any of it is deserialized. Without a key,
the digest it checks against is the one the database recorded when the checkpoint was
accepted. With a verify key, the load takes the checkpoint's files, their sizes and
their digests from the manifest the training job signed, and the database only finds
that manifest.

| Who can write | Without signing | With signing |
| --- | --- | --- |
| the bucket (an object PUT) | refused: the bytes differ from the recorded digest | refused |
| the bucket, while a save is in flight (a file and the rank's report of it) | **loaded**: rank 0 records the digest the forged report gives | refused: each rank signs its report, so the save fails, or the manifest names the digest the rank computed |
| the bucket and the database (a PUT and an UPDATE) | **loaded**: the forger rewrites the recorded digest too | refused: the signed manifest names the old digest, and rewriting it breaks the signature |
| the database only, pointing a head or a resume at an older checkpoint | loaded | **loaded**: the older checkpoint is genuinely signed |
| whoever holds the signing key | loaded | **loaded** |

So signing protects you against a compromised or careless store. It does not protect
you against the training job itself or a stolen key, and it says nothing about which
signed checkpoint is the newest.

### A promotion gate, end to end

The usual setup is two jobs. A training job holds the signing key. A gate (an
evaluation, an export, a deploy step) holds only the public key and loads the
checkpoint before anything ships. The scripts in [`examples/torch/`](../examples/torch/)
run it on one machine:

```bash
uv run python examples/torch/keys.py keys            # keys/signing.key (0600), keys/verify.key
uv run python examples/torch/train.py ledger --signing-key keys/signing.key
uv run python examples/torch/evaluate.py ledger --verify-key keys/verify.key \
    --out model.safetensors                          # verified, scored, shipped
```

To see what the signature catches, run the drill. `tamper.py` does what an insider
with bucket and database access could do. It shifts one bias of the best checkpoint's
model file, by little enough that the held-out score barely moves, and rewrites the
digest the database records for it:

```bash
uv run python examples/torch/tamper.py ledger
uv run python examples/torch/evaluate.py ledger --out forged.safetensors     # ships the forgery
uv run python examples/torch/evaluate.py ledger --verify-key keys/verify.key \
    --out model.safetensors                          # refused, nothing written
```

The gate writes safetensors rather than a pickle, so the server that loads the
weights unpickles nothing.

### Keys

A key is 32 raw bytes. `keys.py` writes each one as hex:

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

key = Ed25519PrivateKey.generate()
signing_key = key.private_bytes_raw()  # secret: the training job only
verify_key = key.public_key().public_bytes_raw()  # public: every job that loads
```

- **The signing key** belongs in a secret manager, and reaches the training job as a
  mounted file or an environment variable. Read it with `bytes.fromhex(...)`, as
  `train.py --signing-key` does. It never appears in a `LedgerConfig`'s repr.
- **The verify key** is public. Commit it beside the gate's configuration, or put it
  in the job's environment.
- **One key per trust boundary**, such as one per team or per project. Every job that
  holds the signing key can produce checkpoints the gate accepts.

### The training job

```python
ledger = Ledger("ledger", signing_key=bytes.fromhex(Path("/secrets/signing.key").read_text()))
```

Nothing else changes. Each rank signs its report of what it wrote, with the digests it
computed as the bytes went out, bound to the execution, attempt, step and rank. Rank 0
verifies every report before it signs the checkpoint's manifest, and a report that does
not verify fails the save with an `OSError`. The manifest's signature covers the
execution, the attempt, the step, the release id, the computation key, and every
file's id, version, size and digest. A signing ledger also verifies its own loads with
its own public key, so a resume or branch refuses a checkpoint that was changed after
it was signed.

The key goes wherever the ledger's description goes:

- to every rank, inside the `Checkpoints` that rank 0 broadcasts over the process
  group;
- to the async writer subprocess, pickled.

Treat every machine in the training job as holding the key.

### Jobs that only load

A gate needs no run. `ledger.configure` points a `CheckpointManager.Config` at a
storage that loads and refuses to save. It holds no lease, and opens no execution:

```python
reader = Ledger("ledger", verify_key=bytes.fromhex(Path("verify.key").read_text()))
release_id, _ = reader.releases.reference("regression/best")
manager = reader.configure(CheckpointManager.Config.with_sync_save()).build()
try:
    state = manager.load(reader.checkpoint_id(release_id), into={"model": model.state_dict()})
except OSError as exc:
    raise SystemExit(f"refused {release_id}: {exc}")
finally:
    manager.close()
```

A ledger with only the verify key cannot sign. If such a ledger opens a run, that
run's saves are unsigned, and its own loads refuse them. Give a job that trains the
signing key.

### When a load is refused

Each refusal is an `OSError`, raised before any of the file is deserialized:

| Message | Means |
| --- | --- |
| `... is not signed` | the training job ran without a signing key, the checkpoint predates signing, or someone stripped the signature from its manifest |
| `the signature on ... does not verify` | it was signed with another key (a rotation, another team), or its manifest was rewritten |
| `the manifest of ... is signed for another checkpoint` | another checkpoint's genuine manifest was copied into this one |
| `... is missing`, `... is malformed`, `... has 2 manifests` | the manifest's bytes or rows were deleted, garbled or duplicated |
| `... differs from the checkpoint the ledger accepted` | the bytes are not the ones that were signed |

Since an unsigned manifest cannot be told from a stripped one, a gate that sees
`is not signed` for a checkpoint it expected to be signed should treat it as tampering.

### Rotating a key

A ledger verifies with one key. After a rotation, train new runs with the new signing
key and verify them with the new public key. Keep the old public key for as long as
anything must still load checkpoints signed with it: a gate that reads an old head
opens a `Ledger` with the old key. Collect or re-run what you no longer trust. A
checkpoint is never re-signed, because versions are immutable.

### What is not signed

- **Which checkpoint is current.** Heads, `final_checkpoint` and resume targets are
  database rows, so a database writer can point them at an older, genuinely signed
  checkpoint. When it matters which one ships, have the training job report the
  release id it finished on through a channel the database cannot rewrite (its job
  log, a CI output), and have the gate load that id rather than a head.
- **Derived releases and observations.** These go through `Ledger.derive` and
  `Run.observe`. The ledger records their digests, but nothing in vestigo reads them
  back and checks them. Whoever reads one checks it.

## What the storage refuses

`LedgerStorage` is a ledger, not a filesystem:

- It serves only accepted checkpoints. A file that was written but never accepted
  (its step crashed, or its release was rejected) reads as missing.
- A file whose bytes no longer match the digest the ledger accepted fails the load
  with `OSError`.
- An attempt cannot write under another attempt's checkpoint id.
- `delete`, `rmdir` and `rename` raise. Old checkpoints go through `collect`.

## Limits

- torch-checkpointing is pinned at 0.1.1. It is pre-1.0, and its `Storage` interface
  and callback shapes have changed before.
- A load holds each file in memory once while it checks the file's digest.
- Sharded state (DTensor, FSDP) goes through torch-checkpointing's own resharding.
  vestigo stores whatever files it writes.
