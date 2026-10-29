# PyTorch checkpoints in the ledger

`vestigo.torch` makes every checkpoint a PyTorch run saves an accepted release of the
ledger. Your model and optimizer stay plain PyTorch objects; the run saves them through
[torch-checkpointing](https://pypi.org/project/torch-checkpointing/), whose
`CheckpointManager` it builds and closes for you. In return you get:

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
| model | a node: the module's class and `state_dict` schema (every key's shape and dtype) under a version you name | `ledger.run(model, ..., version=...)` |
| program | a node over the model: its loss, optimizer class and defaults, and any config you declare | `ledger.run(..., optimizer, loss=..., config=...)` |
| execution | one run of a program on concrete inputs (a build), found again by its computation key | `ledger.run(...)` starts one, or resumes the unfinished one |
| attempt | one process's hold on an execution, with a lease a thread renews | each `ledger.run` |
| checkpoint | one step's files from every rank, accepted together as one release | `run.save(step, state)` |

Weights are not part of a model's identity: a retrained model keeps its node. Source is
not hashed either; bump `version` when behaviour changes. Two programs that differ in
loss, optimizer or config are two programs over one model.

`inputs` maps a role you choose to the data the run trains on. The program, inputs and
starting release make the computation key, and that key is how a run is found again.

## Inputs

An input is a `Path`, a `Remote` or a `str`:

```python
from vestigo.torch import Remote

inputs = {
    "train": Path("data/train.parquet"),  # a file: pinned to its bytes
    "shards": Path("data/shards/"),  # a directory: pinned to every file in it
    "eval": Remote("s3://bucket/eval/"),  # objects: pinned to their store's names
    "seed": "seed=1",  # a label, taken as given
}
```

**A `Path` is verified.** `ledger.run` hashes it (sha256 of a file; for a directory,
over every file's name, size and digest), and the digest goes into the key, not the
path. So:

- the same bytes in another checkout or on another machine are the same computation,
  and a resume there resumes;
- data changed between a crash and the rerun is another computation, so the run
  starts fresh instead of resuming a checkpoint on different data;
- before each checkpoint is accepted, and before `finish`, the files are checked
  against those bytes. A changed, added or removed file raises `InputChanged`
  (from `run.save`, or from its future for an asynchronous save), releases nothing,
  and gives up the lease. Put the bytes back and the same script resumes from the
  last checkpoint taken before the change.

The check is a `stat` per file. A file whose stat changed with its bytes the same (a
`chmod`, a `touch`, a hardlink, an attribute a file browser adds) is hashed again and
passes. So the record says every checkpoint was accepted while exactly those bytes
sat at those paths; an edit undone between two checkpoints goes unseen. Nor can it
prove the training code read them: read the paths you passed.

A directory takes every file in it, hidden ones too, except what an operating system
writes into a folder it shows: `.DS_Store`, `Thumbs.db`, `desktop.ini` and `._`
AppleDouble files. It refuses a link out of the tree or to a directory, a link loop,
a special file such as a pipe, and a directory with no files; a link to a file inside
the tree counts as that file. A relaunch costs a `stat` per file, not a read: digests
are cached per machine in `$VESTIGO_CACHE_DIR`, else `$XDG_CACHE_HOME/vestigo`, else
`~/.cache/vestigo` (with none usable, every file is hashed). A file whose mtime or
ctime is within the last two seconds is hashed again each time, since a write in the
same timestamp tick could leave its stat unchanged.

**A `Remote` is verified through its object store.** `Remote("s3://bucket/key")` names
one object and `Remote("gs://bucket/prefix/")` (ending in `/`) every object under it;
`az://container/...` is Azure. Nothing is downloaded: `ledger.run` lists the objects
and pins each by what its store says names its bytes, and every check lists them again.

| Store | Names the bytes | Same bytes uploaded again |
| --- | --- | --- |
| S3 | ETag | same name |
| GCS | MD5; a composite object's generation | same name; a new generation |
| Azure | Content-MD5 when the upload set one, else ETag | same name; a new ETag |

The URI is part of the pin, unlike a local path: it names the same objects on every
machine, and the same objects in another bucket are another input. Other names, or an
object added or removed under a prefix, fail the next save as for a `Path`; a store
that does not answer fails only that step. Credentials and endpoint come from
`Remote(uri, settings=BlobSettings(...))`, as for the ledger's own `RemoteBlobs`, else
the provider's defaults. Every rank lists the objects with them, and they travel from
rank 0 to every rank. The store's names say the objects were not replaced;
like a `Path`, they cannot say what the code read, so read the URI you passed.

**A `str` is a label.** Nothing reads it. It must change whenever the data does, so
pin a version: a table snapshot, a dataset revision, a vestigo artifact URI with its
`@v=`. A bare `s3://bucket/train.parquet` as a `str` resumes a checkpoint on
overwritten data without warning; pass it as a `Remote`.

The role names are yours. Renaming one changes the key, and the next run starts
fresh.

## Train and checkpoint

```python
from vestigo.torch import Ledger

with Ledger("ledger") as ledger:
    with ledger.run(
        model,
        optimizer,
        loss=nn.MSELoss(),
        version="regression-v1",
        inputs={"train": Path("data/regression.pt")},
        config={"batch": 32},
    ) as run:
        state = run.restore({"model": model, "optim": optimizer, "epoch": 0})
        for epoch in range(state["epoch"] + 1, epochs + 1):
            train_one_epoch()
            run.save(epoch, {"model": model, "optim": optimizer, "epoch": epoch})
        run.finish()
```

`ledger.run` defines the model and program nodes, then resumes this computation's
unfinished execution if there is one, and starts a new one if not. The same script
therefore trains from scratch the first time and carries on after a crash.

`run.save(step, state)` saves each module, optimizer or scheduler in `state` by its
`state_dict`, and anything else as it is. A synchronous save returns the accepted
release. `run.restore(state)` is its mirror. Each object is loaded in place and
returned as itself, and each other value is a default that the saved one replaces. On
a fresh start it returns `state` as given. Only the keys you pass are read, so a
fine-tune can restore `{"model": model}` alone; a key the checkpoint never saved
raises `FileNotFoundError`.

Use the run as a context manager. Leaving the block closes it: it waits for the save
in flight, then drops its lease. `finish()` also waits for that save, then ends the
execution. A block left without `finish()` (an exception, `SystemExit`, a `break` on
preemption) keeps the execution resumable.

For explicit control, `ledger.define_model` and `ledger.define_program` return the
nodes. `ledger.find(program, inputs)`, `ledger.start(program, inputs)` and
`ledger.resume(execution, inputs=inputs)` are the three steps `run` takes. An execution
with a `Path` input resumes only with its inputs, which must be its own.

## Resume after a crash

Run the same script again. `ledger.run` finds the unfinished execution and resumes it
in a new attempt, from its latest accepted checkpoint. `run.restores_from` is that
release, or None for a fresh start. `ledger.resume(execution, from_release=...,
inputs=...)` resumes from an older checkpoint (`inputs` as the run had them, needed
when one is a `Path`).

How the old process ended decides when you can resume:

- **It left the `with` block,** by an exception or `SystemExit`, with at least one
  checkpoint. It dropped its lease, so a resume can start at once.
- **It was killed.** Its lease holds until `lease_seconds` (60 by default) after its
  last renewal. Until then `ledger.run` refuses with "another attempt's lease on it is
  live". Pass `Ledger(..., lease_seconds=...)` to shorten the wait.
- **It left the `with` block before any checkpoint.** The execution failed, and the
  next run starts fresh. A process killed before any checkpoint fails nothing: once its
  lease lapses, the next run starts it over, in the same execution.

Once a new attempt holds the execution, the old process is fenced. Its saves are
refused, and so are its `set_head`, `observe` and `finish` calls. A process that stalls
past `lease_seconds` without renewing (a long pause, a database locked throughout) is
fenced the same way, even if nothing took over: a lapsed lease is never renewed, so
resume the execution in a new process.

`finish()` completes the execution, and nothing resumes it after that. The same
computation run again after a finish is a new execution. The checkpoint the run ended
on (its attempt's last, or the one it restored if it saved none) stays under the
reference `vestigo-torch/<execution>`, and `ledger.final_checkpoint(execution)` returns
that release.

## Branch

Start a new execution from any accepted checkpoint, of this run or another:

```python
with ledger.run(
    model, optimizer, loss=..., version=..., inputs=..., from_release=release_id
) as run:
    run.restore({"model": model})  # the optimizer starts fresh
```

The release is part of the branch's computation, so a branch never dedups against the
run it came from. Run the same branch again after a crash and it resumes. Once it has
finished, it starts another. `ledger.branch(program, release_id, inputs)` always opens
a new execution.

`ledger.run(from_release=)`, `branch` and `resume(from_release=)` refuse a release that
is not a checkpoint (a final or derived release) and a checkpoint `collect` has taken,
before opening anything.

## Lineage

Every execution keeps the record it was started from, so the ledger can say what ran
on what without any logging call in the training script:

```python
execution = ledger.execution(run.execution_id)
execution.program, execution.inputs, execution.init  # the bound computation
execution.attempts  # each Attempt, and the checkpoint it resumed from
execution.checkpoints  # (step, release_id) of each accepted checkpoint still stored

ledger.executions(program)  # every execution of a program
ledger.executions(model)  # of every program over a model
ledger.executions(reading=Path("data"))  # every execution that read these bytes
ledger.resumed_from(release_id)  # the attempts that continued from it
ledger.descendants(run.execution_id)  # its branches, and theirs, at any depth
```

`inputs` hold each role's pin as the computation key hashes it: a `str` as given, a
`Path` or `Remote` as the digests or object names it was pinned to. `reading` is pinned
the same way, so a copy of the data on another machine finds the same executions.
Attempts come in the order they opened; one with no `resumed_from` started from the
execution's `init`. Checkpoints come in the order they were accepted, so a step saved
again after a resume from an earlier checkpoint is listed twice. Each
branch names the checkpoint it started from in `init`, and still descends from its run
after `collect` has taken that checkpoint. Lists are oldest first. A derived release is
not an execution; `ledger.derived(release_id, name)` finds it.

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
`garbage_collect(keep_superseded=)` trims them. Once a name is derived from a
checkpoint, deriving it from an earlier one of the execution raises `ValueError`, since
that would make the earlier checkpoint's outputs current again; this also holds when
the earlier derivation was trimmed. Give such a derivation its own name. `train.py`
derives an evaluation of the checkpoint it finished on.

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
a fenced or ended attempt staged but never released (a step that crashed mid-save), and
those of a save that can no longer be released in an attempt still running: one whose
acceptance failed, or one of a step at or before the newest the attempt accepted. The
save the attempt is writing now stays, so you can collect while runs train. Failed
rows go too (an observation that failed to store, a derivation that failed), except
the placeholder a finished execution's build row still names, which stays with it.
`examples/torch/train.py --keep 1` collects after every epoch.

`collect` returns the store's counts (`deleted_count`, `deleted_bytes`) plus `orphaned`,
the staged files it failed. `dry_run=True` deletes nothing and lists
what it would take under `collected`. Loading a checkpoint, or branching from it,
counts as a use, and `min_idle_seconds=` spares whatever was used that recently. Pass
it when branches may open while `collect` runs, since a branch opened after `collect`
has read the executions is not yet among the ones it keeps.

## Asynchronous saves

```python
from torch_checkpointing.checkpoint_manager import CheckpointManager

run = ledger.run(..., checkpointing=CheckpointManager.Config.with_async_save())
pending = run.save(step, state)  # a future
pending.result()
run.release_of(step)
```

`checkpointing` is the `CheckpointManager.Config` the run builds its manager from. The
run copies it and sets the ledger's storage and callbacks on the copy; per-item
settings (`items`, `default`) carry over. With async saves the writer runs in a
subprocess, so the storage and callbacks are picklable descriptions of the ledger, and
each one opens its own connection. `save` returns the write's future, and the step's
release exists once it completes. The manager has one save in flight. `finish()` and
leaving the block wait for it. `run.manager` is the run's manager, for anything the run
does not wrap.

On macOS, torch-checkpointing 0.1.1's async stager fails whenever MPS is available,
because it calls CUDA device functions. That is an upstream bug. Synchronous saves are
unaffected.

## Several ranks

Every rank runs the same code. `ledger.run` is a collective call, like
`DistributedDataParallel(model)`: call it on every rank, after `init_process_group`,
with the same arguments.

```python
dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # as DistributedDataParallel needs
ledger = Ledger("ledger")  # the same ledger on every rank
with ledger.run(
    model,
    optimizer,
    loss=nn.MSELoss(),
    version="v1",
    inputs={"train": Path("/data/shards")},
) as run:
    epoch = run.restore({"model": model, "optim": optimizer, "epoch": 0})["epoch"]
    while epoch < epochs:
        epoch += 1
        ...
        release = run.save(epoch, {"model": model, "optim": optimizer, "epoch": epoch})
    run.finish()
```

Rank 0 starts or resumes the run and holds its lease; it sends the attempt to the
others, and each rank gets its own `Run`. Each rank writes its own files
(`model_0.pt`, `model_1.pt`, ...) and loads its own back. Rank 0 accepts a step only
after the barrier has seen every rank write. If a rank dies before writing, the barrier
times out, and that step never becomes a release.

Every rank first checks that it runs what rank 0 runs: the same model (class,
`state_dict` shapes and `version`), loss, optimizer, `config`, `from_release`, input
names, save mode (synchronous or not) and barrier timeout, and verify key. A rank with
another config would train another computation under rank 0's key, so a difference
raises `ValueError` on every rank, naming the rank and what differs, before anything is
written; a term that is not JSON on one rank raises `TypeError` on all. Then rank 0
pins the inputs, every other rank checks its copies against the pins, and only then
does rank 0 open the run. Each rank writes through its own `Ledger`, which must be
rank 0's: a rank whose ledger does not hold the run raises on every rank, and rank 0
closes it.

These exchanges go over a gloo group of their own, so they need no GPU whatever the
training backend.

A synchronous `save` and `finish` are collective too. Every rank gets rank 0's outcome:
`save` returns the step's release on every rank, and an error rank 0 hit accepting the
step (another rank's copy changed, the lease was lost, the barrier timed out) is raised
on every rank. Rank 0 posts a save's outcome to the process group's store, waiting for
no one; every other rank waits for it at most the barrier's timeout. So a rank that
comes to the barrier after rank 0 gave up raises rank 0's error, and one that hears
nothing raises rather than wait. If the other ranks gave up at the barrier and rank 0
reached it late, the step is accepted (every rank's files are whole) though their saves
raised. An asynchronous `save` returns each rank's own write future. `close` is each
rank's own; only rank 0's touches the lease.

A rank that raises outside a save leaves the others waiting at the next `ledger.run` or
`finish` until the process group's timeout. A launcher such as `torchrun` ends every
rank when one fails.

### Inputs on several ranks

Each rank passes its own copy of each `Path` input, so a rank on another machine passes
its local path. The copy must hold the bytes rank 0 pinned: a rank whose copy differs
raises `InputChanged` on every rank, at `ledger.run`. A rank on rank 0's machine with
rank 0's path is not read again. A `Remote` names the same objects on every rank.

At each save every rank checks its own copy. Each other rank's report (signed, with a
signing key) records the path and bytes it found after its write; rank 0 checks its own
after the barrier and records that in the manifest. The step is accepted only when
every copy has the bytes the run pinned. Otherwise every rank's save raises
`InputChanged` naming the rank and path: other bytes lapse the lease, and a copy that
cannot be read (say, a transient error on a shared mount) fails only that step. No rank
raises before the barrier, so a bad copy never holds the others up. `finish` checks
every rank's copy once more, and rank 0 lists the `Remote` inputs once for all.

Every rank walks its copy at every save, a `stat` per file, before the barrier: on a
large tree, save less often rather than check less.

`ledger.start`, `ledger.branch` and `ledger.resume` are collective in the same way.
Under a launcher that sets `WORLD_SIZE` above 1, a run without a process group raises
`RuntimeError` rather than have each rank start its own.

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
The same program exported again by the same attempt, under the same scope and
coverage, from example inputs of the same shapes, is one version: the example values
and the zip's timestamps are not compared. Example inputs of another shape change the
graph the archive records, so they make a new version, and so does an export after a
resume, since each attempt records its own. Like a save, `observe` needs the lease.

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

- to every rank, in the attempt rank 0 sends over the process group at `ledger.run`;
- to the async writer subprocess, pickled.

Treat every machine in the training job as holding the key.

### Jobs that only load

A gate needs no run. `ledger.load(release_id, state)` loads an accepted checkpoint
into `state` as `run.restore` does. It holds no lease, opens no execution, and closes
its manager before it returns:

```python
reader = Ledger("ledger", verify_key=bytes.fromhex(Path("verify.key").read_text()))
release_id, _ = reader.releases.reference("regression/best")
try:
    reader.load(release_id, {"model": model})
except OSError as exc:
    raise SystemExit(f"refused {release_id}: {exc}")
```

`ledger.configure` points a `CheckpointManager.Config` of your own at the same
load-only storage, for loads the run does not wrap; load by `ledger.checkpoint_id`.

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
opens a `Ledger` with the old key. A checkpoint is never re-signed, because versions
are immutable. To retire checkpoints signed with a key you no longer trust, move every
head off them and finish or re-run their executions; `collect` then takes them, as it
takes any checkpoint nothing keeps. It cannot pick checkpoints out by key.

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
