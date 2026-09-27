# Local input data

- `archives/`: downloaded SMPL, SMPL-X, and optional FLAME archives. The body converter reads ZIP files directly. FLAME is retained but currently unused.
- `datasets/`: evaluation datasets, such as APPA-REAL and LAGENDA.

These directories are ignored by Git. Converted models remain in `Models/` at the
repository root, where the CLI, app, and packaging script already look for them.

The cleanup moved root-level ZIP files here and `Datasets/` to `data/datasets/`.
Sample photos moved from `Test Photos/` to `examples/photos/`. Existing build and
render outputs remain in `build/` and `out/`.

The body converter prefers archives here, with a fallback to root-level archives
for older checkouts. Explicit `--smpl-zip` and `--smplx-zip` paths still work.
