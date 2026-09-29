"""The episode-to-AlpaSim adapter, without the model.

If these are wrong, every rollout is wrong in a way the model cannot reveal:
a focal agent published under the wrong name, a pedestrian classed as a car,
a parked car asked to drive. CPU only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim_alpasim.episode_scene import (
    TIME_ORIGIN_US,
    EpisodeScene,
    EpisodeSceneProvider,
    states_from_trajectory,
    trajectory_from_states,
)
from sim_alpasim.rollout import _assemble


def _states(n: int = 20, *, speed: float = 10.0, x0: float = 0.0, y: float = 0.0, heading: float = 0.0):
    return [
        {
            "time_s": i * 0.1,
            "position": [x0 + speed * i * 0.1 * math.cos(heading), y + speed * i * 0.1 * math.sin(heading)],
            "velocity": [speed * math.cos(heading), speed * math.sin(heading)],
            "heading_rad": heading,
        }
        for i in range(n)
    ]


def _episode():
    return {
        "episode_id": "ep-1",
        "embodiment": {"name": "av"},
        "frame": "map",
        "agents": [
            {"agent_id": "focal", "agent_type": "vehicle", "states": _states()},
            {"agent_id": "walker", "agent_type": "pedestrian", "states": _states(speed=1.2, y=5.0)},
            {"agent_id": "parked", "agent_type": "vehicle", "states": _states(speed=0.0, x0=30.0)},
            {"agent_id": "bike", "agent_type": "cyclist", "states": _states(speed=4.0, y=-4.0)},
        ],
        "lanes": [
            {"lane_id": "L1", "centerline": [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]},
            {"lane_id": "L2", "centerline": [[0.0, 3.5], [100.0, 3.5]]},
        ],
    }


def test_trajectory_round_trips_positions_headings_and_time() -> None:
    states = _states(heading=0.7)
    traj = trajectory_from_states(states)
    assert len(traj) == 20
    assert int(traj.timestamps_us[0]) == TIME_ORIGIN_US
    assert int(traj.timestamps_us[1]) - int(traj.timestamps_us[0]) == 100_000
    back = states_from_trajectory(traj)
    for a, b in zip(states, back, strict=True):
        assert b["time_s"] == pytest.approx(a["time_s"], abs=1e-6)
        assert b["position"] == pytest.approx(a["position"], abs=1e-4)
        assert b["heading_rad"] == pytest.approx(a["heading_rad"], abs=1e-4)


def test_focal_becomes_the_rig_and_the_rest_traffic_objects() -> None:
    scene = EpisodeScene(episode=_episode(), focal_agent_id="focal")
    assert scene.scene_id == "ep-1"
    assert len(scene.rig.trajectory) == 20
    assert scene.rig.vehicle_config.aabb_x_m == pytest.approx(4.7)
    assert set(scene.traffic_objects) == {"walker", "parked", "bike"}


def test_labels_and_static_flags() -> None:
    objs = EpisodeScene(episode=_episode(), focal_agent_id="focal").traffic_objects
    assert objs["walker"].label_class == "pedestrian"
    assert objs["bike"].label_class == "cyclist"
    assert objs["parked"].label_class == "car"
    assert objs["parked"].is_static is True
    assert objs["walker"].is_static is False
    assert objs["walker"].aabb.x == pytest.approx(0.6)


def test_lanes_become_a_vector_map_with_lanes_populated() -> None:
    vmap = EpisodeScene(episode=_episode(), focal_agent_id="focal").map
    assert vmap is not None
    assert vmap.map_id == "av2:ep-1"
    assert vmap.lanes is not None and len(vmap.lanes) == 2
    assert vmap.lanes[0].center.points.shape[1] >= 3  # xy -> xyz (or xyzh)


def test_no_lanes_means_no_map() -> None:
    ep = _episode()
    ep["lanes"] = []
    assert EpisodeScene(episode=ep, focal_agent_id="focal").map is None


def test_unknown_focal_is_refused() -> None:
    with pytest.raises(KeyError, match="focal agent"):
        EpisodeScene(episode=_episode(), focal_agent_id="ghost")


def test_provider_registers_and_forgets() -> None:
    p = EpisodeSceneProvider()
    sid = p.register(EpisodeScene(episode=_episode(), focal_agent_id="focal"))
    assert p.has_scene(sid) and p.num_scenes == 1 and p.scene_ids == [sid]
    assert p.get_data_source(sid).scene_id == sid
    p.forget(sid)
    assert not p.has_scene(sid)
    with pytest.raises(KeyError):
        p.get_data_source(sid)


def test_assemble_patches_reacted_agents_after_handover_only() -> None:
    ep = _episode()
    focal_new = _states(speed=3.0)  # the replacement: slower
    handover_us = TIME_ORIGIN_US + 900_000  # after step 9
    # The service "moved" the walker 10 m north from step 10 on.
    moved = {
        TIME_ORIGIN_US + i * 100_000: (ep["agents"][1]["states"][i]["position"][0], 15.0, 0.0)
        for i in range(10, 20)
    }
    result = _assemble(ep, "focal", focal_new, {"walker": moved}, handover_us, queries=2)
    by_id = {a["agent_id"]: a for a in result.episode["agents"]}
    assert result.reacted == ("walker",)
    assert set(result.replayed) == {"parked", "bike"}
    assert result.queries == 2
    assert by_id["focal"]["states"][-1]["position"] == pytest.approx(focal_new[-1]["position"])
    walker = by_id["walker"]["states"]
    assert walker[5]["position"][1] == pytest.approx(5.0)   # before handover: recorded
    assert walker[15]["position"][1] == pytest.approx(15.0)  # after: simulated
    assert by_id["parked"]["states"] == ep["agents"][2]["states"]
    assert np.isfinite([s["velocity"][0] for s in walker]).all()
