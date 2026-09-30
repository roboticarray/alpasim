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


def test_pre_handover_states_keep_the_recorded_velocity() -> None:
    """Tracker velocity is filtered; re-deriving it makes an agent look synthetic."""
    ep = _episode()
    for i, s in enumerate(ep["agents"][1]["states"]):
        s["velocity"] = [1.2 + 0.001 * i, 0.0]  # a distinctive, smooth recorded value
    handover_us = TIME_ORIGIN_US + 900_000
    moved = {TIME_ORIGIN_US + i * 100_000: (float(i), 5.0, 0.0) for i in range(10, 20)}
    result = _assemble(ep, "focal", _states(), {"walker": moved}, handover_us, queries=1)
    walker = next(a for a in result.episode["agents"] if a["agent_id"] == "walker")["states"]
    for i in range(10):
        assert walker[i]["velocity"] == ep["agents"][1]["states"][i]["velocity"]


def test_simulated_velocity_is_smoothed_not_raw() -> None:
    """Jittered simulated positions must not become jittered velocities."""
    ep = _episode()
    rng = np.random.default_rng(0)
    handover_us = TIME_ORIGIN_US + 400_000
    # Continue the walker's recorded 1.2 m/s path, with tracker-scale jitter.
    moved = {
        TIME_ORIGIN_US + i * 100_000: (0.12 * i + float(rng.uniform(-0.03, 0.03)), 5.0, 0.0)
        for i in range(5, 20)
    }
    result = _assemble(ep, "focal", _states(), {"walker": moved}, handover_us, queries=1)
    walker = next(a for a in result.episode["agents"] if a["agent_id"] == "walker")["states"]
    vx = np.array([s["velocity"][0] for s in walker[8:18]])
    raw = np.diff([s["position"][0] for s in walker[7:19]]) / 0.1
    assert np.std(np.diff(vx)) < np.std(np.diff(raw)) / 2
    assert abs(float(np.mean(vx)) - 1.2) < 0.3


def test_rollout_refuses_history_longer_than_the_trajectory() -> None:
    from sim_alpasim.rollout import RolloutError, rollout

    with pytest.raises(RolloutError, match="history_steps"):
        rollout(None, EpisodeSceneProvider(), _episode(), focal_agent_id="focal",
                focal_trajectory=_states(n=10), history_steps=10)


def test_the_logged_ego_history_is_the_replacement_not_the_recording() -> None:
    """A perturbation changes the history too; the service must see the new one."""
    from sim_alpasim.rollout import _logged

    scene = EpisodeScene(episode=_episode(), focal_agent_id="focal")
    replacement = _states(speed=3.0, y=7.0)
    logged = _logged(scene, replacement, TIME_ORIGIN_US + 900_000)
    ego = next(o for o in logged if o.object_id == "EGO")
    assert len(ego.trajectory.poses) == 10
    assert ego.trajectory.poses[5].pose.vec.y == pytest.approx(7.0)


def test_a_state_the_service_did_not_simulate_is_dropped_not_spliced() -> None:
    """The real reply shape: one pose per query, every fifth step.

    Filling the four steps between from the recording made CAT-K agents jump
    back and forth between two futures -- p95 |accel| of 800 m/s^2.
    """
    ep = _episode()
    handover_us = TIME_ORIGIN_US + 400_000
    sparse = {TIME_ORIGIN_US + i * 100_000: (0.12 * i, 9.0, 0.0) for i in (9, 14, 19)}
    result = _assemble(ep, "focal", _states(), {"walker": sparse}, handover_us, queries=3)
    walker = next(a for a in result.episode["agents"] if a["agent_id"] == "walker")["states"]
    times = [round(s["time_s"], 1) for s in walker]
    assert times == [0.0, 0.1, 0.2, 0.3, 0.4, 0.9, 1.4, 1.9]
    assert all(s["position"][1] == pytest.approx(9.0) for s in walker[5:])


def test_reacted_agents_lose_their_recorded_actions() -> None:
    ep = _episode()
    ep["agents"][1]["actions"] = [[0.1, 0.0, 0.0]] * 20
    ep["agents"][3]["actions"] = [[0.2, 0.0, 0.0]] * 20
    moved = {TIME_ORIGIN_US + i * 100_000: (float(i), 5.0, 0.0) for i in range(10, 20)}
    result = _assemble(ep, "focal", _states(), {"walker": moved}, TIME_ORIGIN_US + 900_000, queries=1)
    by_id = {a["agent_id"]: a for a in result.episode["agents"]}
    assert "actions" not in by_id["walker"]
    assert by_id["bike"]["actions"] == ep["agents"][3]["actions"]  # replayed: untouched


class _Traj:
    def __init__(self, poses: dict[int, tuple[float, float, float]]) -> None:
        ts = sorted(poses)
        self.timestamps_us = np.array(ts)
        self.positions = np.array([[poses[t][0], poses[t][1], 0.0] for t in ts])
        self.yaws = np.array([poses[t][2] for t in ts])


class _Servicer:
    def __init__(self, store: dict[str, dict[int, tuple[float, float, float]]]) -> None:
        state = type("S", (), {"closed_loop_trajectories": {k: _Traj(v) for k, v in store.items()}})
        self._sessions = {"s": state}


def _us(i: int) -> int:
    return TIME_ORIGIN_US + i * 100_000


def test_forecast_poses_come_from_the_closed_loop_store() -> None:
    from sim_alpasim.rollout import _forecast_poses

    recorded = {_us(i): (float(i), 0.0, 0.0) for i in range(5)}
    simulated = {_us(i): (float(i), 1.0, 0.0) for i in range(5, 15)}
    servicer = _Servicer({"a": {**recorded, **simulated}, "never-forecast": recorded | {_us(9): (9.0, 0.0, 0.0)}})
    replied = {"a": {_us(9): simulated[_us(9)], _us(14): simulated[_us(14)]}}
    got = _forecast_poses(servicer, "s", replied, handover_us=_us(4))
    assert set(got) == {"a"}
    assert got["a"] == simulated


def test_forecast_poses_refuse_a_store_they_cannot_trust() -> None:
    from sim_alpasim.rollout import RolloutError, _forecast_poses

    simulated = {_us(i): (float(i), 1.0, 0.0) for i in range(5, 15)}
    with pytest.raises(RolloutError, match="no session state"):
        _forecast_poses(_Servicer({}), "gone", {"a": {_us(9): simulated[_us(9)]}}, _us(4))
    with pytest.raises(RolloutError, match="disagrees"):
        _forecast_poses(_Servicer({"a": simulated}), "s", {"a": {_us(9): (0.0, 0.0, 0.0)}}, _us(4))
    with pytest.raises(RolloutError, match="runs past"):
        _forecast_poses(_Servicer({"a": simulated}), "s", {"a": {_us(9): simulated[_us(9)]}}, _us(4))
    with pytest.raises(RolloutError, match="absent"):
        _forecast_poses(_Servicer({}), "s", {"a": {_us(9): simulated[_us(9)]}}, _us(4))
