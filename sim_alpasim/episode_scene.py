"""Adapt a data-contracts episode to AlpaSim's ``SceneDataSource``.

AlpaSim loads scenes from USDZ artifacts. The curation loop has episodes in
the platform's own schema — agents with timestamped states, and lane
centrelines — and a synthetic episode never had a USDZ. This module builds
the same in-memory objects the artifact loader would (``Rig``,
``TrafficObjects``, a trajdata ``VectorMap``), so AlpaSim's own
``CATKSceneAdapter`` runs on them unchanged.

Conventions the traffic service relies on, made explicit here:

* the focal agent is the object called ``"EGO"`` — the session factory looks
  it up by that name;
* timestamps are microseconds and must be positive, so the episode's clock is
  offset by :data:`TIME_ORIGIN_US`;
* quaternions are built about +z from the agent's heading, in the ``(x, y, z,
  w)`` order the Rust ``Pose`` stores (the proto order is ``wxyz``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from alpasim_utils.geometry import yaw_to_quat_components
from alpasim_utils.scenario import AABB, Rig, TrafficObject, TrafficObjects, VehicleConfig
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import Polyline, RoadLane
from utils_rs import Pose, Trajectory

#: Episode time is offset by this so every timestamp is a positive integer.
TIME_ORIGIN_US = 1_000_000_000

#: Data-contracts agent types -> CAT-K obstacle classes (car, truck,
#: pedestrian, cyclist). Anything else is treated as a car.
LABEL_CLASS: Mapping[str, str] = {
    "vehicle": "car",
    "bus": "truck",
    "truck": "truck",
    "pedestrian": "pedestrian",
    "cyclist": "cyclist",
    "riderless_bicycle": "cyclist",
    "motorcyclist": "cyclist",
}

#: Footprints (length, width, height) in metres by agent type, matching the
#: acceptance filters' table in motion-curation.
FOOTPRINT_M: Mapping[str, tuple[float, float, float]] = {
    "vehicle": (4.7, 2.0, 1.5),
    "bus": (10.0, 2.6, 3.2),
    "truck": (8.0, 2.5, 3.0),
    "motorcyclist": (2.2, 0.8, 1.4),
    "cyclist": (1.8, 0.6, 1.6),
    "riderless_bicycle": (1.8, 0.6, 1.0),
    "pedestrian": (0.6, 0.6, 1.8),
}

#: Below this path length an agent is static: the model should not be asked
#: to predict a parked car, and the service excludes statics from prediction.
STATIC_PATH_M = 0.5


def _quat(yaw: float) -> np.ndarray:
    """Quaternion for a heading, in the ``(x, y, z, w)`` order ``utils_rs.Pose`` stores.

    ``yaw_to_quat_components`` returns ``(w, x, y, z)`` for the *proto*, and
    ``Pose.from_proto`` reorders it; ``Pose.from_denormalized_quat`` does not.
    Passing wxyz there yields a pose whose yaw reads as 0 for every heading,
    which the first CPU rollout did.
    """
    w, x, y, z = yaw_to_quat_components(float(yaw))
    return np.asarray([x, y, z, w], dtype=np.float32)


def _heading_from_motion(states: Sequence[Mapping[str, Any]]) -> list[float]:
    """Headings from the state stream, or from travel direction when absent."""
    out: list[float] = []
    last = 0.0
    for i, s in enumerate(states):
        h = s.get("heading_rad")
        if h is None:
            j = min(i + 1, len(states) - 1)
            k = i if j != i else max(i - 1, 0)
            dx = states[j]["position"][0] - states[k]["position"][0]
            dy = states[j]["position"][1] - states[k]["position"][1]
            if math.hypot(dx, dy) > 1e-6:
                last = math.atan2(dy, dx)
            h = last
        else:
            last = float(h)
        out.append(float(h))
    return out


def trajectory_from_states(states: Sequence[Mapping[str, Any]]) -> Trajectory:
    """A ``utils_rs.Trajectory`` from a data-contracts state stream."""
    if not states:
        return Trajectory.create_empty()
    headings = _heading_from_motion(states)
    timestamps = np.asarray(
        [TIME_ORIGIN_US + int(round(float(s["time_s"]) * 1_000_000)) for s in states],
        dtype=np.uint64,
    )
    poses = [
        Pose.from_denormalized_quat(
            np.asarray([s["position"][0], s["position"][1], 0.0], dtype=np.float32),
            _quat(h),
        )
        for s, h in zip(states, headings, strict=True)
    ]
    return Trajectory.from_poses(timestamps, poses)


def states_from_trajectory(
    trajectory: Trajectory, *, template: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Data-contracts states from a ``utils_rs.Trajectory`` (velocity re-derived)."""
    times = [(int(t) - TIME_ORIGIN_US) / 1_000_000 for t in trajectory.timestamps_us]
    pos = np.asarray(trajectory.positions, dtype=np.float64)
    yaws = [float(y) for y in trajectory.yaws]
    out: list[dict[str, Any]] = []
    for i, (t, (x, y, _z), h) in enumerate(zip(times, pos, yaws, strict=True)):
        j = min(i + 1, len(times) - 1)
        k = i if j != i else max(i - 1, 0)
        dt = times[j] - times[k]
        vx = (pos[j][0] - pos[k][0]) / dt if dt > 0 else 0.0
        vy = (pos[j][1] - pos[k][1]) / dt if dt > 0 else 0.0
        out.append(
            {"time_s": round(t, 6), "position": [float(x), float(y)], "velocity": [vx, vy],
             "heading_rad": h}
        )
    return out


def _path_length(states: Sequence[Mapping[str, Any]]) -> float:
    return sum(
        math.dist(a["position"][:2], b["position"][:2])
        for a, b in zip(states, states[1:], strict=False)
    )


def _vector_map(episode: Mapping[str, Any]) -> VectorMap | None:
    """Lane centrelines as trajdata ``RoadLane`` elements; None when the episode has no map."""
    lanes = episode.get("lanes") or []
    if not lanes:
        return None
    elements: dict[Any, dict[str, Any]] = defaultdict(dict)
    for lane in lanes:
        pts = np.asarray(lane["centerline"], dtype=np.float64)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        road_lane = RoadLane(id=str(lane["lane_id"]), center=Polyline(pts[:, :2]))
        elements[road_lane.elem_type][road_lane.id] = road_lane
    if not elements:
        return None
    # Elements are passed to the constructor: VectorMap derives its ``lanes``
    # list in __post_init__, so elements added afterwards are not in it.
    return VectorMap(map_id=f"av2:{episode['episode_id']}", elements=elements)


@dataclass
class _Metadata:
    scene_id: str


@dataclass
class EpisodeScene:
    """A data-contracts episode exposed as an AlpaSim ``SceneDataSource``.

    ``focal_agent_id`` becomes the rig, published to the traffic service as
    ``"EGO"``; every other agent is a traffic object.
    """

    episode: Mapping[str, Any]
    focal_agent_id: str
    scene_id: str = ""
    source: str = "episode"
    _rig: Rig | None = field(default=None, init=False, repr=False)
    _objects: TrafficObjects | None = field(default=None, init=False, repr=False)
    _map: VectorMap | None = field(default=None, init=False, repr=False)
    _map_built: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.scene_id:
            self.scene_id = str(self.episode["episode_id"])
        ids = [a["agent_id"] for a in self.episode["agents"]]
        if self.focal_agent_id not in ids:
            raise KeyError(f"focal agent {self.focal_agent_id!r} not in {self.scene_id!r}")

    @property
    def metadata(self) -> _Metadata:
        return _Metadata(scene_id=self.scene_id)

    @property
    def rig(self) -> Rig:
        if self._rig is None:
            focal = next(a for a in self.episode["agents"] if a["agent_id"] == self.focal_agent_id)
            length, width, height = FOOTPRINT_M.get(focal["agent_type"], FOOTPRINT_M["vehicle"])
            self._rig = Rig(
                sequence_id=self.scene_id,
                trajectory=trajectory_from_states(focal["states"]),
                camera_ids=[],
                camera_frame_timestamps_us={},
                camera_frame_ranges_us={},
                world_to_nre=np.eye(4, dtype=np.float32),
                vehicle_config=VehicleConfig(aabb_x_m=length, aabb_y_m=width, aabb_z_m=height),
            )
        return self._rig

    @property
    def traffic_objects(self) -> TrafficObjects:
        if self._objects is None:
            objects = TrafficObjects()
            for agent in self.episode["agents"]:
                if agent["agent_id"] == self.focal_agent_id:
                    continue
                kind = agent["agent_type"]
                length, width, height = FOOTPRINT_M.get(kind, FOOTPRINT_M["vehicle"])
                objects[agent["agent_id"]] = TrafficObject(
                    track_id=str(agent["agent_id"]),
                    aabb=AABB(x=length, y=width, z=height),
                    trajectory=trajectory_from_states(agent["states"]),
                    is_static=_path_length(agent["states"]) < STATIC_PATH_M,
                    label_class=LABEL_CLASS.get(kind, "car"),
                )
            self._objects = objects
        return self._objects

    @property
    def map(self) -> VectorMap | None:
        if not self._map_built:
            self._map = _vector_map(self.episode)
            self._map_built = True
        return self._map


class EpisodeSceneProvider:
    """A ``SceneProvider`` whose scenes are registered at runtime, one per episode."""

    def __init__(self) -> None:
        self._scenes: dict[str, EpisodeScene] = {}

    def register(self, scene: EpisodeScene) -> str:
        self._scenes[scene.scene_id] = scene
        return scene.scene_id

    def forget(self, scene_id: str) -> None:
        self._scenes.pop(scene_id, None)

    @property
    def scene_ids(self) -> list[str]:
        return list(self._scenes)

    def has_scene(self, scene_id: str) -> bool:
        return scene_id in self._scenes

    @property
    def num_scenes(self) -> int:
        return len(self._scenes)

    def get_data_source(self, scene_id: str) -> EpisodeScene:
        try:
            return self._scenes[scene_id]
        except KeyError as exc:
            raise KeyError(f"unknown scene {scene_id!r}") from exc
