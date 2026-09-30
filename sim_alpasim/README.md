# sim-alpasim

The platform's sim MCP server over AlpaSim's traffic service. It exists for
one verb — `sim_rollout` (sim protocol 1.1): *given a recorded scene and a
replacement trajectory for one agent, what would everyone else have done?*
AlpaSim's CAT-K world model answers that; this package adapts the platform's
episode schema to AlpaSim's scene contract and drives the model in-process.

| module | what it does |
|---|---|
| `episode_scene.py` | data-contracts episode -> AlpaSim `SceneDataSource` (`Rig`, `TrafficObjects`, trajdata `VectorMap` from lane centrelines); the focal agent becomes `"EGO"` |
| `rollout.py` | builds AlpaSim's own `TrafficServiceServicer` on the LFS-tracked CAT-K weights, swaps its USDZ loader for an episode provider, and runs `start_session` / `simulate` in-process; a `FAILED_PRECONDITION` is raised, never replayed |
| `server.py` | FastMCP server: the shared protocol verbs (traffic-only, so most return `ok:false` with a reason) plus `sim_rollout` |

## Environment

The fork has no git history in common with upstream (it was re-initialised),
so the traffic service comes from an upstream checkout:

```bash
git worktree add ../alpasim-upstream upstream/main          # NVlabs/alpasim
cd ../alpasim-upstream && git lfs checkout data/trafficsim-models   # 70 MB CAT-K checkpoint
UV_PYTHON=3.12 uv sync --package alpasim-trafficsim        # needs a Rust toolchain for utils_rs
uv pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv pip install "mcp>=1.2,<2"
```

Run the server with that interpreter and this repo on the path:

```bash
PYTHONPATH=/path/to/alpasim SIM_ALPASIM_DEVICE=cuda \
  ../alpasim-upstream/.venv/bin/python -m sim_alpasim.server
```

`SIM_ALPASIM_MODELS` overrides the weights directory; `SIM_ALPASIM_DEVICE=cpu`
works (a 110-step, 34-agent AV2 episode re-simulates in ~3.5 s on CPU).

## Conventions that bit

* `utils_rs.Pose` stores quaternions **xyzw**; the proto is **wxyz** and
  `Pose.from_proto` reorders, `Pose.from_denormalized_quat` does not. Getting
  this wrong yields yaw 0 for every heading and the model still "reacts".
* Timestamps are microseconds and must be positive: episode time is offset
  by `TIME_ORIGIN_US`.
* `VectorMap` derives its `lanes` list in `__post_init__`, so elements are
  passed to the constructor, not added afterwards.

## Tests

CPU only, no model: `PYTHONPATH=. ../alpasim-upstream/.venv/bin/python -m pytest tests -q`.
