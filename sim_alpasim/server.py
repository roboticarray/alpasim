"""sim-alpasim MCP server: the shared sim protocol over AlpaSim's traffic service.

The shared verbs are present because the protocol says shared tools must
remain; this backend is traffic-only, so the ones that need a full runtime
(assets, config, stepping a world) return ``ok: false`` with a reason rather
than pretending. ``sim_rollout`` (protocol 1.1) is the verb this server
exists for.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from sim_alpasim import __version__
from sim_alpasim.rollout import DEFAULT_MODELS, RolloutError, build_servicer, rollout

mcp = FastMCP("sim-alpasim")

_STATE: dict[str, Any] = {"started": False, "servicer": None, "provider": None, "scene": None}
_LOCK = threading.Lock()
_NOT_SUPPORTED = "traffic-only backend: use sim-isaac / sim-unreal for a full runtime"


def _ok(**kw: Any) -> str:
    return json.dumps({"ok": True, **kw})


def _err(message: str) -> str:
    return json.dumps({"ok": False, "error": message})


def _servicer() -> tuple[Any, Any]:
    """The CAT-K servicer, built on first use (it loads the model onto the device)."""
    with _LOCK:
        if _STATE["servicer"] is None:
            models = Path(os.environ.get("SIM_ALPASIM_MODELS", str(DEFAULT_MODELS)))
            device = os.environ.get("SIM_ALPASIM_DEVICE", "cuda")
            _STATE["servicer"], _STATE["provider"] = build_servicer(models=models, device=device)
        return _STATE["servicer"], _STATE["provider"]


@mcp.tool()
def sim_start(headless: bool = True) -> str:
    """Start the simulation (traffic-only; nothing to render)."""
    _STATE["started"] = True
    return _ok(headless=headless, mode="alpasim-traffic", version=__version__)


@mcp.tool()
def sim_stop() -> str:
    """Stop the simulation and release the model."""
    with _LOCK:
        _STATE.update({"started": False, "servicer": None, "provider": None, "scene": None})
    return _ok()


@mcp.tool()
def sim_load(scenario_path: str, map_path: str | None = None) -> str:
    """Load a data-contracts episode JSON as the current scene."""
    try:
        episode = json.loads(Path(scenario_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return _err(f"could not load {scenario_path}: {exc}")
    _STATE["scene"] = episode
    return _ok(
        scenario=str(scenario_path),
        description=str(episode.get("episode_id", "")),
        ego=str(episode["agents"][0]["agent_id"]) if episode.get("agents") else "",
        actors=len(episode.get("agents", [])),
        map=map_path,
    )


@mcp.tool()
def sim_step(n: int = 1) -> str:
    """Not supported: this backend re-simulates whole episodes via sim_rollout."""
    return _err(_NOT_SUPPORTED)


@mcp.tool()
def sim_get_obs() -> str:
    """Not supported here; use sim_rollout."""
    return _err(_NOT_SUPPORTED)


@mcp.tool()
def sim_set_actions(actions: str) -> str:
    """Not supported here; the focal trajectory is given to sim_rollout whole."""
    return _err(_NOT_SUPPORTED)


@mcp.tool()
def sim_add_asset(asset_path: str, position: str | None = None) -> str:
    """Not supported: no renderer in the traffic-only backend."""
    return _err(_NOT_SUPPORTED)


@mcp.tool()
def sim_set_config(key: str, value: str) -> str:
    """Not supported: the traffic model has no runtime config to change."""
    return _err(_NOT_SUPPORTED)


@mcp.tool()
def sim_add_logical_actor(actor_type: str, position: str, behavior: str | None = None) -> str:
    """Not supported: actors come from the loaded episode."""
    return _err(_NOT_SUPPORTED)


@mcp.tool()
def sim_rollout(
    scene: str, focal_agent_id: str, focal_trajectory: str, seed: int = 0, history_steps: int = 50
) -> str:
    """Closed-loop re-simulation (protocol 1.1).

    Args:
        scene: Data-contracts episode JSON, or "" to use the scene from sim_load.
        focal_agent_id: The agent whose trajectory is replaced.
        focal_trajectory: JSON list of states ({time_s, position, heading_rad}) on
            the episode's time grid, history and future.
        seed: Forwarded to the traffic service.
        history_steps: How many leading states are history; the handover is the last.
    """
    try:
        episode = json.loads(scene) if scene else _STATE["scene"]
        if not episode:
            return _err("no scene: pass one or call sim_load first")
        trajectory = json.loads(focal_trajectory)
        servicer, provider = _servicer()
        result = rollout(
            servicer,
            provider,
            episode,
            focal_agent_id=focal_agent_id,
            focal_trajectory=trajectory,
            history_steps=history_steps,
            seed=seed,
        )
    except RolloutError as exc:
        return _err(str(exc))
    except (KeyError, ValueError, TypeError, IndexError, json.JSONDecodeError) as exc:
        return _err(f"bad request: {exc}")
    except Exception as exc:  # noqa: BLE001 -- the protocol requires ok:false, never a crash
        return _err(f"rollout failed: {type(exc).__name__}: {exc}")
    return _ok(
        episode=result.episode,
        backend="alpasim-catk",
        reacted=list(result.reacted),
        replayed=list(result.replayed),
        queries=result.queries,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
