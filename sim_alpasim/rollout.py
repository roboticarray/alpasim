"""Closed-loop re-simulation of one episode through AlpaSim's traffic service.

Drives ``TrafficServiceServicer`` in-process — the exact code path the gRPC
server runs, minus the socket — so a rollout is the CAT-K world model
reacting to the client's replacement trajectory for the focal agent.

Session shape, in the service's terms:

* ``logged_object_trajectories``: the focal agent's *recorded history* as
  ``"EGO"``, and every other agent's full recording. Before ``handover`` the
  service replays; after it, it predicts.
* ``handover_time_us``: the last history timestamp. From there the focal's
  replacement trajectory is fed as the ``EGO`` update on every ``simulate``,
  and the service conditions its world model on it.
* ``simulate`` is queried every ``prediction_steps * dt`` until the episode's
  last timestamp; each reply carries the other agents' poses for the steps it
  forecast. A ``FAILED_PRECONDITION`` (the model could not act) is raised,
  never papered over with a replay — the protocol requires ``ok: false``.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import numpy as np
from alpasim_grpc.v0 import common_pb2, traffic_pb2
from alpasim_utils.geometry import trajectory_from_grpc, trajectory_to_grpc
from utils_rs import Trajectory

from sim_alpasim.episode_scene import (
    TIME_ORIGIN_US,
    EpisodeScene,
    EpisodeSceneProvider,
    trajectory_from_states,
)

#: Where the LFS-tracked CAT-K weights live in the upstream checkout.
DEFAULT_MODELS = Path("/home/ubuntu/datasets/jobs/alpasim-upstream/data/trafficsim-models")


class RolloutError(RuntimeError):
    """The traffic service refused or failed the session."""


class _Context:
    """The subset of ``grpc.ServicerContext`` the servicer uses, in-process."""

    def __init__(self) -> None:
        self.code: grpc.StatusCode = grpc.StatusCode.OK
        self.details = ""

    def set_code(self, code: grpc.StatusCode) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details

    def raise_for_status(self, what: str) -> None:
        if self.code != grpc.StatusCode.OK:
            raise RolloutError(f"{what}: {self.code.name}: {self.details}")


def _us(state: Mapping[str, Any]) -> int:
    """A state's timestamp on the service's microsecond clock."""
    return TIME_ORIGIN_US + int(round(float(state["time_s"]) * 1_000_000))


#: Half-width, in samples, of the velocity smoothing applied to simulated
#: states. AV2 tracker velocity jitters ~0.10 m/s step to step; a raw central
#: difference of 10 Hz positions jitters ~0.26. The predictor reads velocity,
#: so raw derivatives make a re-simulated agent look synthetic before it has
#: done anything -- the same artefact motion-curation's generator had.
VELOCITY_SMOOTH_HALF = 2


def _patched_states(
    states: Sequence[Mapping[str, Any]],
    poses: Mapping[int, tuple[float, float, float]],
    handover_us: int,
) -> list[dict[str, Any]]:
    """One agent's states with the simulated poses written in after handover.

    States at or before handover -- and any the service did not return -- are
    kept verbatim, velocity included. Simulated states get position and heading
    from the service and a smoothed central-difference velocity over the patched
    positions.
    """
    out = [dict(s) for s in states]
    simulated = [i for i, s in enumerate(states) if _us(s) > handover_us and _us(s) in poses]
    for i in simulated:
        x, y, yaw = poses[_us(states[i])]
        out[i]["position"] = [x, y]
        out[i]["heading_rad"] = yaw
    raw: list[tuple[float, float]] = []
    n = len(out)
    for i in range(n):
        j, k = min(i + 1, n - 1), max(i - 1, 0)
        dt = float(out[j]["time_s"]) - float(out[k]["time_s"])
        if dt <= 0:
            raw.append((0.0, 0.0))
            continue
        raw.append(
            (
                (out[j]["position"][0] - out[k]["position"][0]) / dt,
                (out[j]["position"][1] - out[k]["position"][1]) / dt,
            )
        )
    for i in simulated:
        lo, hi = max(i - VELOCITY_SMOOTH_HALF, 0), min(i + VELOCITY_SMOOTH_HALF, n - 1)
        window = raw[lo : hi + 1]
        out[i]["velocity"] = [
            sum(v[0] for v in window) / len(window),
            sum(v[1] for v in window) / len(window),
        ]
    return out

def build_servicer(
    *,
    models: Path = DEFAULT_MODELS,
    device: str = "cuda",
    prediction_steps: int = 5,
    num_history_steps: int = 16,
) -> tuple[Any, EpisodeSceneProvider]:
    """A ``TrafficServiceServicer`` reading scenes from an episode provider.

    The upstream servicer hard-wires a USDZ loader in its constructor; it is
    constructed with a throwaway folder and the loader is swapped for one whose
    scenes are registered at runtime. Everything else — adapter, predictor,
    session factory — is upstream's own.
    """
    from alpasim_trafficsim.grpc.config import CatkConfig, CatkLoaderConfig, CatkModelConfig
    from alpasim_trafficsim.grpc.servicer import TrafficServiceServicer

    cfg = CatkConfig(
        device=device,
        loader=CatkLoaderConfig(
            usdz_folder=str(models),  # unused once the loader is replaced
            num_history_steps=num_history_steps,
            prediction_steps=prediction_steps,
        ),
        model=CatkModelConfig(
            config_path=str(models / "catk_v120" / "config.yaml"),
            ckpt_path=str(models / "catk_v120" / "latest.ckpt"),
            token_pkl_dir=str(models / "tokens"),
        ),
    )
    servicer = TrafficServiceServicer(None, catk_config=cfg, usdz_folder=models)
    provider = EpisodeSceneProvider()
    servicer._scene_loader = provider  # noqa: SLF001 -- the documented swap above
    return servicer, provider


def _logged(
    scene: EpisodeScene, focal_trajectory: Sequence[Mapping[str, Any]], handover_us: int
) -> list[traffic_pb2.ObjectTrajectory]:
    """Logged trajectories: EGO history up to handover, others in full.

    The EGO history is taken from the *replacement* trajectory, not the
    recording: the protocol says the client's trajectory replaces the focal
    agent's history as well as its future, and a perturbation changes both.
    """
    out: list[traffic_pb2.ObjectTrajectory] = []
    history = [s for s in focal_trajectory if _us(s) <= handover_us]
    rig = scene.rig
    vc = rig.vehicle_config
    out.append(
        traffic_pb2.ObjectTrajectory(
            object_id="EGO",
            aabb=common_pb2.AABB(size_x=vc.aabb_x_m, size_y=vc.aabb_y_m, size_z=vc.aabb_z_m),
            trajectory=trajectory_to_grpc(trajectory_from_states(history)),
            is_static=False,
        )
    )
    for track_id, obj in scene.traffic_objects.items():
        out.append(
            traffic_pb2.ObjectTrajectory(
                object_id=str(track_id),
                aabb=common_pb2.AABB(size_x=obj.aabb.x, size_y=obj.aabb.y, size_z=obj.aabb.z),
                trajectory=trajectory_to_grpc(obj.trajectory),
                is_static=bool(obj.is_static),
            )
        )
    return out


@dataclass(frozen=True)
class RolloutResult:
    episode: dict[str, Any]
    reacted: tuple[str, ...]
    replayed: tuple[str, ...]
    queries: int


def rollout(
    servicer: Any,
    provider: EpisodeSceneProvider,
    episode: Mapping[str, Any],
    *,
    focal_agent_id: str,
    focal_trajectory: Sequence[Mapping[str, Any]],
    history_steps: int,
    seed: int = 0,
    dt_s: float = 0.1,
    prediction_steps: int = 5,
) -> RolloutResult:
    """Re-simulate ``episode`` around ``focal_trajectory``.

    Args:
        servicer: From :func:`build_servicer`.
        provider: Its scene provider, to register this episode under.
        episode: The recorded scene (data-contracts episode dict).
        focal_agent_id: Which agent the replacement trajectory belongs to.
        focal_trajectory: The replacement states for that agent, on the
            episode's time grid, history and future.
        history_steps: How many leading states are history; the handover is
            the last of them.
        seed: Forwarded to the service as ``random_seed``.

    Raises:
        RolloutError: If the service refuses the session or cannot predict.
    """
    if not 1 <= history_steps < len(focal_trajectory):
        raise RolloutError(
            f"history_steps must be in [1, {len(focal_trajectory)}) for a "
            f"{len(focal_trajectory)}-state focal trajectory, got {history_steps}"
        )
    scene = EpisodeScene(episode=episode, focal_agent_id=focal_agent_id)
    scene_id = provider.register(scene)
    session = str(uuid.uuid4())
    dt_us = int(round(dt_s * 1_000_000))
    times_us = [_us(s) for s in focal_trajectory]
    handover_us = times_us[history_steps - 1]
    ego_update = traffic_pb2.ObjectTrajectoryUpdate(
        object_id="EGO", trajectory=trajectory_to_grpc(trajectory_from_states(focal_trajectory))
    )
    try:
        ctx = _Context()
        servicer.start_session(
            traffic_pb2.TrafficSessionRequest(
                session_uuid=session,
                scene_id=scene_id,
                random_seed=int(seed),
                logged_object_trajectories=_logged(scene, focal_trajectory, handover_us),
                handover_time_us=handover_us,
            ),
            ctx,
        )
        ctx.raise_for_status("start_session")

        poses: dict[str, dict[int, tuple[float, float, float]]] = {}
        queries = 0
        query_us = handover_us
        end_us = times_us[-1]
        while query_us < end_us:
            query_us = min(query_us + prediction_steps * dt_us, end_us)
            ctx = _Context()
            reply = servicer.simulate(
                traffic_pb2.TrafficRequest(
                    session_uuid=session,
                    time_query_us=query_us,
                    object_trajectory_updates=[ego_update],
                ),
                ctx,
            )
            ctx.raise_for_status(f"simulate@{(query_us - TIME_ORIGIN_US) / 1e6:.1f}s")
            queries += 1
            for update in reply.object_trajectory_updates:
                if update.object_id == "EGO":
                    continue
                traj = trajectory_from_grpc(update.trajectory)
                store = poses.setdefault(str(update.object_id), {})
                for t, p, y in zip(
                    traj.timestamps_us, np.asarray(traj.positions), traj.yaws, strict=True
                ):
                    store[int(t)] = (float(p[0]), float(p[1]), float(y))
    finally:
        try:
            servicer.close_session(
                traffic_pb2.TrafficSessionCloseRequest(session_uuid=session), _Context()
            )
        finally:
            provider.forget(scene_id)

    return _assemble(episode, focal_agent_id, focal_trajectory, poses, handover_us, queries)


def _assemble(
    episode: Mapping[str, Any],
    focal_agent_id: str,
    focal_trajectory: Sequence[Mapping[str, Any]],
    poses: Mapping[str, Mapping[int, tuple[float, float, float]]],
    handover_us: int,
    queries: int,
) -> RolloutResult:
    """The re-simulated episode: focal replaced, reacted agents patched after handover."""
    agents: list[dict[str, Any]] = []
    reacted: list[str] = []
    replayed: list[str] = []
    for agent in episode["agents"]:
        aid = agent["agent_id"]
        if aid == focal_agent_id:
            agents.append({**agent, "states": [dict(s) for s in focal_trajectory]})
            continue
        got = poses.get(aid)
        if not got:
            agents.append(dict(agent))
            replayed.append(aid)
            continue
        agents.append({**agent, "states": _patched_states(agent["states"], got, handover_us)})
        reacted.append(aid)
    out = {**episode, "agents": agents}
    return RolloutResult(episode=out, reacted=tuple(reacted), replayed=tuple(replayed), queries=queries)
