# sim-alpasim: deviations from the shared sim MCP protocol

Protocol version implemented: **1.1** (adds the optional `sim_rollout`).

This backend is **traffic-only**: it re-simulates every agent in a recorded
scene around a client-supplied trajectory for one of them. It has no world
to step, no renderer, and no runtime config, so:

| verb | behaviour |
|---|---|
| `sim_start` / `sim_stop` | supported; `mode` is `alpasim-traffic` |
| `sim_load` | loads a data-contracts episode JSON as the current scene |
| `sim_rollout` | **the reason this server exists** — see the protocol doc |
| `sim_step`, `sim_get_obs`, `sim_set_actions`, `sim_add_asset`, `sim_set_config`, `sim_add_logical_actor` | return `ok: false` with a reason; the shared tools remain present per the protocol |

Determinism: identical `(scene, focal_trajectory, seed)` gives an identical
episode on the same device and weights. The manifest records
`backend: alpasim-catk`.
