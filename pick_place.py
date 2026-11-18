"""Autonomous pick-and-place for the SO-101 tabletop scene.

Six actions, one per state:

    1. MOVE_ABOVE    travel until the open gripper straddles the cube
    2. DESCEND       lower straight down to grasp height
    3. CLOSE         close the fingers on the object
    4. LIFT          raise the object clear of the table
    5. MOVE_TO_BIN   carry it over the target bin
    6. OPEN          open the fingers and let the object drop

The targets come from the scene camera, not from the simulator: each attempt
looks at one RGB frame, finds the cube (colour blob + model-based silhouette
fit) and the bin, and plans from those coordinates.  Nothing in the completion
tests uses vision -- those stay on the physics, so a bad detection shows up as a
missed grasp rather than as a state that claims to be finished.

No human input is involved: each state tests its own completion condition and
hands over to the next one.  If the grasp slips, the machine retries from
state 1 (aiming at the cube where it actually lies now), up to MAX_ATTEMPTS.

Completion is tested on the thing that actually has to happen, never on a fixed
timer, and never on a quantity that physically cannot converge:

  * a state ends when the arm joints reach their target, or when they stop
    improving (a jammed arm would otherwise wait forever);
  * the two states that carry the cube compare the gripper COMMAND, because the
    jaw is jammed on the cube and can never reach it;
  * OPEN ends when the cube has landed and stopped, not after a fixed delay.

Two facts drive the geometry, both measured in this scene:

  * the pinch centre of the fingers sits PINCH_OFFSET ahead of the tool site
    along the site z axis, so the site must be offset by that much when the
    cube has to end up between the pads;
  * the site has to sit at cube-centre height (GRASP_SITE_Z), not above it,
    or the pads close on air above the cube.

Usage:
    python pick_place.py                      # headless run + verdict
    python pick_place.py --cube red           # pick another cube
    python pick_place.py --viewer             # watch it in the viewer
    python pick_place.py --frames out/        # write a PNG per transition
    python pick_place.py --cameras            # live scene + wrist camera windows
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import mujoco
import numpy as np

import vision

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(ROOT, "mjcf", "scene.xml")

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"

# --- grasp geometry (measured, see README) ------------------------------
PINCH_OFFSET = 0.013        # m: pinch centre ahead of the site along site +z
GRASP_SITE_Z = 0.010        # m: site height so the pads straddle the cube
PREGRASP_CLEARANCE = 0.065  # m: how far above the grasp pose the approach runs
CARRY_SITE_Z = 0.065        # m: highest top-down carry the wrist can reach
RELEASE_SITE_Z = 0.060      # m: release height over the bin floor

# --- gripper commands (rad) ---------------------------------------------
GRIPPER_OPEN = 0.30         # rad: wide enough for the 20 mm cube to sit between the pads
GRIPPER_CLOSE = -0.10       # rad: squeezes the pads onto the cube
GRIPPER_RELEASE = 0.60      # opens fully so the cube drops

# --- motion limits ------------------------------------------------------
MAX_JOINT_SPEED = 4.0       # rad/s on the joint reference ramp
GRIPPER_SPEED = 1.6         # rad/s on the gripper reference ramp (jaw free)
GRIPPER_HOLD_SPEED = 2.2    # rad/s while the pads are jammed on a payload
GRIPPER_RELEASE_SPEED = 0.45  # rad/s when opening over the bin.  Opening fast
                              # makes the pads flick the cube ~13 mm off target,
                              # so this one stays deliberately slow.
GRIPPER_SETTLE = 0.06       # rad: how close the jaw, or its command, must get
SETTLE_TOL = 0.03           # rad: a state is done when every joint is this close
MIN_DWELL = 0.15            # s: ignore "already there" right after a transition
DESCEND_SPEED = 0.45        # rad/s: the final few cm down are the delicate part
DESCEND_FAST = 2.4          # rad/s: free travel down while still above the cube
DESCEND_FREE_Z = 0.032      # m: TCP height where the slow final descent starts
NUDGE_LIMIT = 0.0025        # m: cube displacement that means the fingers bumped it
STALL_WINDOW = 0.30         # s: if the joints stop making progress for this long
STALL_PROGRESS = 0.002      # rad: ...and improve by less than this, the motion
                            #      is jammed (fingers on the cube, table, ...)
                            #      and the state ends instead of waiting forever
MAX_ATTEMPTS = 3

# A single frame can have the cube hidden behind a jaw or a glare, so one missed
# detection is retried before it is believed.
VISION_RETRIES = 3


class VisionError(RuntimeError):
    """Raised when the planner needs a pose that vision never delivered."""

# Perception.  1280x960 is not cosmetic: the cube is only ~55 px across at
# 640x480, and at that size the yaw fit is off by 2.5 deg on average (8 deg
# worst case) against 1.1 deg (2.8 worst) at 1280x960.
PERCEPTION_WIDTH = 1280
PERCEPTION_HEIGHT = 960
LIFT_SUCCESS = 0.030        # m: cube rise that counts as a successful grasp
OFFSET_TRIM = (-0.004, 0.004)   # m: lateral nudges used by the retries


class State(Enum):
    MOVE_ABOVE = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    MOVE_TO_BIN = auto()
    OPEN = auto()
    DONE = auto()


@dataclass
class Report:
    cube: str
    reached_states: list[str] = field(default_factory=list)
    cube_start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cube_lifted: float = 0.0
    cube_final: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bin_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    grasp_confirmed: bool = False
    attempts: int = 0
    sim_time: float = 0.0
    frames: list[np.ndarray] = field(default_factory=list)
    vision_calls: int = 0
    vision_error_mm: list[float] = field(default_factory=list)
    bin_vision_error_mm: float = -1.0


class Arm:
    """Joint indexing, forward kinematics, damped-least-squares IK, servo ramp."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        jids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT))
        self.arm_qadr = model.jnt_qposadr[np.array(jids[:-1])]
        self.arm_dadr = model.jnt_dofadr[np.array(jids[:-1])]
        self.grip_qadr = model.jnt_qposadr[jids[-1]]
        self.lo = model.jnt_range[self.arm_qadr, 0].copy()
        self.hi = model.jnt_range[self.arm_qadr, 1].copy()
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    # -- state -----------------------------------------------------------
    def tcp(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    def site_axes(self) -> np.ndarray:
        return self.data.site_xmat[self.site].reshape(3, 3).copy()

    def joints(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    def gripper(self) -> float:
        return float(self.data.qpos[self.grip_qadr])

    # -- kinematics ------------------------------------------------------
    @staticmethod
    def _frame(approach, jaw) -> np.ndarray:
        """Rotation whose columns are (approach, jaw x approach, jaw).

        The gripper closes along column 3, so `jaw` is the closing direction.
        """
        a = np.asarray(approach, float)
        a = a / np.linalg.norm(a)
        o = np.asarray(jaw, float)
        o = o - a * np.dot(o, a)
        o = o / np.linalg.norm(o)
        return np.column_stack([a, np.cross(o, a), o])

    def grasp_site_target(self, cube_xy, site_z: float, roll: float = 0.0,
                          trim: float = 0.0) -> np.ndarray:
        """Site position that puts the pinch centre on `cube_xy` at `site_z`."""
        close_dir = np.array([np.cos(roll), np.sin(roll)])
        xy = np.asarray(cube_xy, float)[:2] - close_dir * (PINCH_OFFSET + trim)
        return np.array([xy[0], xy[1], site_z])

    def solve_ik(self, target, seed=None, roll: float = 0.0,
                 iters: int = 400) -> tuple[np.ndarray, float, float]:
        """Position+orientation IK for a top-down pose. Leaves the sim untouched."""
        r_des = self._frame([0, 0, -1], [np.cos(roll), np.sin(roll), 0.0])
        q = (self.joints() if seed is None else np.asarray(seed, float)).copy()
        save = self.data.qpos.copy()
        pos_err = rot_err = float("inf")
        try:
            for _ in range(iters):
                self.data.qpos[self.arm_qadr] = q
                mujoco.mj_forward(self.model, self.data)
                pos_err = float(np.linalg.norm(target - self.data.site_xpos[self.site]))
                cur = self.data.site_xmat[self.site].reshape(3, 3)
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (r_des @ cur.T).reshape(9))
                rot_vec = np.zeros(3)
                mujoco.mju_quat2Vel(rot_vec, quat, 1.0)
                rot_err = float(np.linalg.norm(rot_vec))
                if pos_err < 1e-4 and rot_err < 1e-2:
                    break
                mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
                jac = np.vstack([self._jp[:, self.arm_dadr], 0.35 * self._jr[:, self.arm_dadr]])
                err = np.concatenate([target - self.data.site_xpos[self.site], 0.175 * rot_vec])
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), err)
                q = np.clip(q + dq, self.lo, self.hi)
        finally:
            self.data.qpos[:] = save
            mujoco.mj_forward(self.model, self.data)
        return q, pos_err, rot_err

    def solve_ik_multi(self, target, roll: float = 0.0, tries: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        best = None
        for i in range(tries):
            start = self.joints() if i == 0 else rng.uniform(self.lo, self.hi)
            q, pe, re = self.solve_ik(target, seed=start, roll=roll)
            if best is None or pe < best[1]:
                best = (q, pe, re)
            if pe < 1e-3:
                break
        return best


class PickPlace:
    def __init__(self, model, data, cube: str, viewer=None,
                 frame_dir: str | None = None, use_vision: bool = True,
                 perception=None, cameras=None):
        self.model = model
        self.data = data
        self.arm = Arm(model, data)
        self.use_vision = use_vision
        # One GL context per process: a second mujoco.Renderer alongside a live
        # one corrupts readbacks under WSLg (measured -- every other frame came
        # back vertically mirrored).  Pass a shared Perception when looping.
        self.per = perception
        if use_vision and self.per is None:
            self.per = vision.Perception(model, data, PERCEPTION_WIDTH,
                                         PERCEPTION_HEIGHT)
        self.seen_cube = None        # vision.Pose of the target, or None
        self.seen_bin = None         # vision.Pose of the bin, or None
        self.vision_lost = False
        self.cube = cube
        self.cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{cube}")
        self.cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_cube_{cube}")
        self.bin_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bin_center")
        cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_cube_{cube}")
        self.cube_dadr = int(model.jnt_dofadr[cube_jid])
        self.finger_geoms = self._finger_geom_ids()
        self.viewer = viewer
        self.cameras = cameras           # camera_windows.CameraWindows, or None
        self.frame_dir = frame_dir
        self.renderer = None
        # With a viewer (or camera windows) open the run should be watchable, so
        # hold the wall clock to the simulation clock.  Headless runs go flat out.
        self.realtime = viewer is not None or cameras is not None
        self._wall0 = None
        self._sim0 = 0.0
        self.aborted = False
        self._last_sync = -1.0
        self.viewer_period = 1.0 / 60.0     # s of simulated time between redraws
        # Camera windows are expensive per frame (~65 ms wall for two 640x480
        # views), so they are throttled on the wall clock at a real fps instead
        # of on simulated time.  Without this the run drifts to 3x real time.
        self._cam_period = 1.0 / getattr(cameras, "fps", 10.0) if cameras else 0.0
        self._last_cam = -1e9
        # The camera windows may spend at most this share of real time drawing.
        # A fixed fps alone cannot bound that: each pair of 640x480 views costs
        # ~65 ms here, so 10 fps would be a 65% duty cycle and the run would
        # never keep up.  Both rules are enforced together -- the fps cap keeps
        # motion smooth, the duty cap keeps the run in real time.
        self._cam_budget = float(getattr(cameras, "budget", 0.60)) if cameras else 0.0
        self._cam_spent = 0.0            # seconds spent drawing
        self._cam_wall0 = None
        self.report = Report(cube=cube)

        self.q_ref = self.arm.joints()
        self.grip_ref = GRIPPER_OPEN
        self.state = State.MOVE_ABOVE
        self.clock = 0.0
        self.trim = 0.0
        self._stall = 0
        self._q_target = self.q_ref.copy()
        self._best_err = float("inf")
        self._best_time = 0.0
        self._jammed = False
        self._anchor = None
        self._replans = 0

        if frame_dir:
            os.makedirs(frame_dir, exist_ok=True)
            # Do NOT add a second mujoco.Renderer while perception holds one:
            # under WSLg that makes readbacks alternate into vertical mirrors.
            if self.per is None:
                self.renderer = mujoco.Renderer(model, 480, 640)

    # -- helpers ---------------------------------------------------------
    def _finger_geom_ids(self) -> set[int]:
        out = set()
        for name in ("gripper", "moving_jaw_so101_v1"):
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for g in range(self.model.ngeom):
                if int(self.model.geom_bodyid[g]) == b and self.model.geom_contype[g]:
                    out.add(g)
        return out

    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def cube_yaw(self) -> float:
        """Yaw of the cube about world z, folded into [0, pi/2).

        The pads are parallel plates and the cube is a box, so a rotation of
        90 degrees is the same grasp.  Squaring the pads up with the cube faces
        (instead of always closing along +x) is what makes the grip robust: on
        a corner-first approach the pads only touch two opposite corners and
        the cube squirts out sideways.
        """
        m = self.data.xmat[self.cube_body].reshape(3, 3)
        return float(np.arctan2(m[1, 0], m[0, 0]) % (np.pi / 2))

    def bin_center(self) -> np.ndarray:
        return self.data.site_xpos[self.bin_site].copy()

    # -- perception ------------------------------------------------------
    def look_cube(self) -> bool:
        """One camera frame -> where the target cube is.

        Returns False if the cube could not be seen, in which case planning
        falls back to the last pose that was seen (or to ground truth if it was
        never seen at all) -- the controller must not stall because one frame
        had a bad blob.
        """
        if self.per is None:
            return False
        t0 = time.perf_counter()
        pose = None
        for _ in range(VISION_RETRIES):
            pose = self.per.find_cube(self.cube)
            if pose is not None:
                break
        ms = 1000 * (time.perf_counter() - t0)
        if pose is None:
            self.vision_lost = True
            print(f"  [VISION     ] cube_{self.cube} not visible in "
                  f"{VISION_RETRIES} frame(s)")
            return False
        self.seen_cube = pose
        truth = self.cube_pos()
        err = 1000 * float(np.hypot(pose.xy[0] - truth[0], pose.xy[1] - truth[1]))
        dyaw = (np.degrees(pose.yaw - self.cube_yaw()) + 45.0) % 90.0 - 45.0
        print(f"  [VISION     ] cube_{self.cube} at ({pose.xy[0]:.4f}, {pose.xy[1]:.4f}) "
              f"yaw {np.degrees(pose.yaw):+6.1f} deg  IoU {pose.score:.3f}  {ms:.0f} ms"
              f"   [truth {err:.1f} mm / {dyaw:+.1f} deg]")
        self.report.vision_calls += 1
        self.report.vision_error_mm.append(err)
        return True

    def look_bin(self) -> bool:
        """One frame -> where the bin is (retried like look_cube).

        A single frame is not enough: the grey mask that outlines the bin is
        borderline, and in one run with the camera windows open a single frame
        returned nothing while an immediate re-render of the same state was
        detected cleanly (pixel difference 0.00, so not a readback fault).
        """
        if self.per is None:
            return False
        t0 = time.perf_counter()
        pose = None
        for _ in range(VISION_RETRIES):
            pose = self.per.find_bin()
            if pose is not None:
                break
        ms = 1000 * (time.perf_counter() - t0)
        if pose is None:
            print("  [VISION     ] bin not visible in %d frame(s)" % VISION_RETRIES)
            return False
        self.seen_bin = pose
        truth = self.bin_center()
        err = 1000 * float(np.hypot(pose.xy[0] - truth[0], pose.xy[1] - truth[1]))
        print(f"  [VISION     ] bin at ({pose.xy[0]:.4f}, {pose.xy[1]:.4f})  "
              f"IoU {pose.score:.3f}  {ms:.0f} ms   [truth {err:.1f} mm]")
        self.report.bin_vision_error_mm = err
        self.report.vision_calls += 1
        return True

    def target_cube(self) -> tuple[np.ndarray, float]:
        """Where the cube is, according to vision.

        Without vision (``--no-vision``) the simulator is the sensor.  With it,
        never fall back to the simulator: a silent fallback would make a missed
        detection indistinguishable from a working one.
        """
        if self.seen_cube is not None:
            return self.seen_cube.xy, float(self.seen_cube.yaw)
        if not self.use_vision:
            return self.cube_pos()[:2], self.cube_yaw()
        raise VisionError(f"cube_{self.cube} was never seen")

    def target_bin(self) -> np.ndarray:
        if self.seen_bin is not None:
            return self.seen_bin.xy
        if not self.use_vision:
            return self.bin_center()[:2]
        raise VisionError("bin was never seen")

    def touching(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.cube_geom in pair and pair & self.finger_geoms:
                return True
        return False

    def _step(self, n: int = 1) -> None:
        for _ in range(n):
            self.data.ctrl[:5] = self.q_ref
            self.data.ctrl[5] = self.grip_ref
            mujoco.mj_step(self.model, self.data)
            self.report.cube_lifted = max(self.report.cube_lifted,
                                          float(self.cube_pos()[2]))
            if self.viewer is not None or self.cameras is not None:
                if self.viewer is not None and not self.viewer.is_running():
                    self.aborted = True
                    return
                if self.cameras is not None and not self.cameras.alive():
                    self.cameras.closed = True
                    print("!! camera windows closed in %s" % self.state.name)
                    self.aborted = True
                    return
                self._pace()
                if self.viewer is not None and \
                        float(self.data.time) - self._last_sync >= self.viewer_period:
                    self._last_sync = float(self.data.time)
                    self.viewer.sync()
                if self.cameras is not None:
                    # Wall-clock throttle, not simulated time: a sim-time
                    # throttle (60 redraws per simulated second) made a 3 s run
                    # take 10 s.  The duty cap bounds the total cost instead.
                    now = time.perf_counter()
                    if self._cam_wall0 is None:
                        self._cam_wall0 = now
                    elapsed = now - self._cam_wall0
                    due = now - self._last_cam >= self._cam_period
                    affordable = self._cam_spent <= self._cam_budget * elapsed
                    if due:
                        self._last_cam = now
                        if affordable:
                            self._cam_spent += self.cameras.tick("%-11s" % self.state.name)
                        else:
                            self.cameras.dropped += 1   # budget used up
        self.report.sim_time = float(self.data.time)

    def _pace(self) -> float:
        """Sleep just enough that wall time tracks simulated time.

        Returns the signed lag in seconds: > 0 means the run is LATE (an
        expensive redraw just happened), < 0 means it is early and slept.

        The baseline is deliberately NOT re-anchored just because we fall behind:
        re-anchoring would forgive the time spent rendering, so the animation
        would never catch up and the whole run would slow to the renderer's
        pace (measured: 3.08 s of motion taking 10.2 s of wall time).  Instead
        the deficit is paid back by running the physics flat out for a moment.
        """
        if not self.realtime:
            return 0.0
        now = time.perf_counter()
        if self._wall0 is None:
            self._wall0 = now
            self._sim0 = float(self.data.time)
        behind = now - (self._wall0 + (float(self.data.time) - self._sim0))
        if behind > 2.0:            # absurd backlog (debugger, swap storm): reset
            self._wall0 = now
            self._sim0 = float(self.data.time)
            return 0.0
        if behind < 0:
            time.sleep(min(-behind, 0.05))
        return behind

    def _track(self, grip_target: float, grip_hold: bool = False,
               grip_speed: float | None = None,
               speed: float | None = None) -> bool:
        """Ramp the reference toward _q_target; True once the arm really arrives.

        The reference is ramped at a limited speed so the arm does not slam, and
        completion is measured against the FINAL target (not the moving
        reference), otherwise a state would report "arrived" on its first step.

        `grip_hold` is for the states that travel with the payload between the
        pads.  There the jaw physically cannot obey a tighter command -- it is
        jammed on the cube -- so comparing the jaw angle with the command never
        settles and the machine sits still until the cube slowly creeps.  With
        `grip_hold` only the gripper COMMAND has to arrive; the arm joints still
        have to reach the target for real.
        """
        delta = self._q_target - self.q_ref
        step = (MAX_JOINT_SPEED if speed is None else speed) * self.model.opt.timestep
        far = float(np.max(np.abs(delta)))
        self.q_ref = (self.q_ref + delta * (step / far)) if far > step else self._q_target.copy()

        if grip_speed is None:
            gspeed = GRIPPER_HOLD_SPEED if grip_hold else GRIPPER_SPEED
        else:
            gspeed = grip_speed
        gstep = gspeed * self.model.opt.timestep
        self.grip_ref += float(np.clip(grip_target - self.grip_ref, -gstep, gstep))

        self._step()
        arm_err = float(np.max(np.abs(self.arm.joints() - self._q_target)))

        # Jam watchdog: a blocked arm can sit at a constant error forever (the
        # pads resting on the cube, the wrist pushing on the table, ...).  If
        # the error stops improving the state is over -- waiting longer only
        # looks like the machine has frozen.
        now = float(self.data.time)
        if arm_err < self._best_err - STALL_PROGRESS:
            self._best_err = arm_err
            self._best_time = now
        elif now - self._best_time > STALL_WINDOW and arm_err > SETTLE_TOL:
            if not self._jammed:
                print("  [%-11s] jammed, %.3f rad short of the target -- moving on"
                      % (self.state.name, arm_err))
            self._jammed = True
        if grip_hold:
            grip_ok = abs(self.grip_ref - grip_target) < 1e-6
        else:
            grip_ok = abs(self.arm.gripper() - grip_target) < GRIPPER_SETTLE
        return (arm_err < SETTLE_TOL and grip_ok) or self._jammed

    def _capture(self, tag: str) -> None:
        if self.per is not None:
            img = self.per.render()          # share the one GL context
        elif self.renderer is not None:
            self.renderer.update_scene(self.data, camera="scene_cam")
            img = self.renderer.render()
        else:
            return
        self.report.frames.append(img)
        if self.frame_dir:
            from PIL import Image
            Image.fromarray(img).save(
                os.path.join(self.frame_dir, f"{len(self.report.frames):04d}_{tag}.png"))

    def _enter(self, state: State, plan: Callable[[], None]) -> None:
        self.state = state
        self.clock = 0.0
        self._stall = 0
        self._best_err = float("inf")
        self._best_time = float(self.data.time)
        self._jammed = False
        self._anchor = None
        self._replans = 0
        self.report.reached_states.append(state.name)
        plan()
        self._capture(state.name.lower())

    def _ik_or_warn(self, label: str, target, roll=0.0, multi=False):
        if multi:
            q, pe, re = self.arm.solve_ik_multi(target, roll=roll)
        else:
            q, pe, re = self.arm.solve_ik(target, seed=self.arm.joints(), roll=roll)
        print(f"  [{label:11s}] site {np.round(target, 3)}  "
              f"err {pe * 1000:5.2f} mm / {np.degrees(re):4.2f} deg")
        self._q_target = q

    # -- the six states --------------------------------------------------
    def _move_above(self) -> bool:
        self.clock += self.model.opt.timestep
        done = self._track(GRIPPER_OPEN)
        return done and self.clock > MIN_DWELL

    def _descend(self) -> bool:
        """Lower onto the cube, re-aiming if the fingers bump it on the way.

        The pads only clear a 20 mm cube by a couple of millimetres, so a
        descent that is a hair off-centre nudges the cube aside instead of
        straddling it.  Rather than close on air and waste a whole retry, watch
        the cube: as soon as it moves, re-plan the descent onto where it is now.
        """
        self.clock += self.model.opt.timestep
        speed = DESCEND_FAST if float(self.arm.tcp()[2]) > DESCEND_FREE_Z else DESCEND_SPEED
        done = self._track(GRIPPER_OPEN, speed=speed)

        if self._anchor is None:
            self._anchor = self.cube_pos()[:2].copy()
        moved = float(np.linalg.norm(self.cube_pos()[:2] - self._anchor))
        if moved > NUDGE_LIMIT and self._replans < 3:
            self._replans += 1
            print("  [DESCEND    ] cube nudged %.1f mm aside, re-aiming (%d)"
                  % (moved * 1000, self._replans))
            self._best_err = float("inf")
            self._best_time = float(self.data.time)
            self._jammed = False
            self._plan_descend()
            self._anchor = self.cube_pos()[:2].copy()
            return False
        return done

    def _close(self) -> bool:
        """Squeeze until the fingers stall on the cube (not merely touch it)."""
        self.clock += self.model.opt.timestep
        self._track(GRIPPER_CLOSE)

        jaw = self.arm.gripper()
        lag = self.grip_ref - jaw              # >0 while the pads are blocked
        if lag > 0.05:
            self._stall += 1
        else:
            self._stall = 0

        if self._stall > 20:                   # ~40 ms of the jaw refusing to follow
            self.report.grasp_confirmed = self.touching()
            print(f"  [CLOSE      ] pads stalled on the cube: jaw {jaw:+.3f} rad, "
                  f"cmd {self.grip_ref:+.3f} rad")
            return True

        if self.grip_ref <= GRIPPER_CLOSE + 1e-9:
            self.report.grasp_confirmed = self.touching()
            print(f"  [CLOSE      ] closed to the stop, jaw {jaw:+.3f} rad, "
                  f"contact {'yes' if self.report.grasp_confirmed else 'NO'}")
            return True

        if self.clock > 2.5:
            print(f"  [CLOSE      ] gave up: jaw {jaw:+.3f} rad, "
                  f"contact {'yes' if self.touching() else 'NO'}")
            return True
        return False

    def _lift(self) -> bool:
        self.clock += self.model.opt.timestep
        return self._track(GRIPPER_CLOSE, grip_hold=True)

    def _move_to_bin(self) -> bool:
        self.clock += self.model.opt.timestep
        return self._track(GRIPPER_CLOSE, grip_hold=True)

    def _open(self) -> bool:
        self.clock += self.model.opt.timestep
        self._track(GRIPPER_RELEASE, grip_speed=GRIPPER_RELEASE_SPEED)
        opened = self.arm.gripper() > 0.22          # pads clear of the cube
        landed = float(self.cube_pos()[2]) < RELEASE_SITE_Z
        still = float(np.linalg.norm(self.data.qvel[self.cube_dadr])) < 0.05
        return opened and landed and still and self.clock > 0.10

    # -- drivers ---------------------------------------------------------
    def _plan_move_above(self) -> None:
        if self.use_vision:
            self.look_cube()
        xy, roll = self.target_cube()
        self._ik_or_warn("MOVE_ABOVE",
                         self.arm.grasp_site_target(xy, GRASP_SITE_Z + PREGRASP_CLEARANCE,
                                                    roll=roll, trim=self.trim),
                         roll=roll, multi=True)

    def _plan_descend(self) -> None:
        if self.use_vision and (self._replans or self.seen_cube is None):
            self.look_cube()       # re-aiming after a nudge: look again
        xy, roll = self.target_cube()
        self._ik_or_warn("DESCEND",
                         self.arm.grasp_site_target(xy, GRASP_SITE_Z,
                                                    roll=roll, trim=self.trim),
                         roll=roll)

    def _plan_close(self) -> None:
        pass

    def _plan_lift(self) -> None:
        xy, roll = self.target_cube()
        self._ik_or_warn("LIFT",
                         self.arm.grasp_site_target(xy, CARRY_SITE_Z,
                                                    roll=roll, trim=self.trim),
                         roll=roll)

    def _plan_move_to_bin(self) -> None:
        if self.use_vision and self.seen_bin is None:
            self.look_bin()
        self._ik_or_warn("MOVE_TO_BIN",
                         self.arm.grasp_site_target(self.target_bin(), CARRY_SITE_Z,
                                                    trim=self.trim), multi=True)

    def _plan_open(self) -> None:
        self._ik_or_warn("OPEN",
                         self.arm.grasp_site_target(self.target_bin(), RELEASE_SITE_Z,
                                                    trim=self.trim))

    def _park(self) -> None:
        """Open the fingers and hold still, so a retry starts from a clean pose."""
        self.grip_ref = GRIPPER_OPEN
        self.q_ref = self.arm.joints().copy()
        self._q_target = self.q_ref.copy()
        self._jammed = False
        for _ in range(60):
            self._step()

    def _advance(self, state, handlers, plans, max_seconds):
        self._enter(state, plans[state])
        while not handlers[state]():
            if self.aborted:
                print("!! run stopped in %s" % state.name)
                return False
            if self.data.time > max_seconds:
                print("!! timeout in %s at t=%.1fs" % (state.name, self.data.time))
                return False
        return True

    def run(self, max_seconds: float = 40.0) -> Report:
        self.report.cube_start = self.cube_pos()
        self.report.bin_center = self.bin_center()
        print("cube_%s at %s   bin at %s"
              % (self.cube, np.round(self.report.cube_start, 4),
                 np.round(self.report.bin_center, 4)))

        handlers = {
            State.MOVE_ABOVE: self._move_above,
            State.DESCEND: self._descend,
            State.CLOSE: self._close,
            State.LIFT: self._lift,
            State.MOVE_TO_BIN: self._move_to_bin,
            State.OPEN: self._open,
        }
        plans = {
            State.MOVE_ABOVE: self._plan_move_above,
            State.DESCEND: self._plan_descend,
            State.CLOSE: self._plan_close,
            State.LIFT: self._plan_lift,
            State.MOVE_TO_BIN: self._plan_move_to_bin,
            State.OPEN: self._plan_open,
        }
        grasp_states = (State.MOVE_ABOVE, State.DESCEND, State.CLOSE, State.LIFT)

        if self.use_vision:
            print("perception   : %dx%d from %s"
                  % (PERCEPTION_WIDTH, PERCEPTION_HEIGHT, self.per.cam.name))
        else:
            print("perception   : off (using simulator ground truth)")

        lifted = False
        for attempt in range(MAX_ATTEMPTS):
            self.report.attempts = attempt + 1
            self.trim = 0.0 if attempt == 0 else OFFSET_TRIM[attempt % len(OFFSET_TRIM)]
            if attempt:
                print("")
                print("-- grasp attempt %d (lateral trim %+.0f mm)"
                      % (attempt + 1, self.trim * 1000))
            z0 = float(self.cube_pos()[2])
            self.seen_cube = None      # look again at the start of each attempt
            try:
                for state in grasp_states:
                    if not self._advance(state, handlers, plans, max_seconds):
                        self.report.cube_final = self.cube_pos()
                        self.state = State.DONE
                        return self.report
            except VisionError as exc:
                print("!! %s -- attempt %d abandoned" % (exc, attempt + 1))
                self._park()           # open the fingers before retrying
                continue
            rose = 1000 * (float(self.cube_pos()[2]) - z0)
            if rose > LIFT_SUCCESS * 1000 and float(self.cube_pos()[2]) > 0.035:
                lifted = True
                break
            print("   cube stayed on the table (rose %+.1f mm)" % rose)

        if lifted:
            try:
                for state in (State.MOVE_TO_BIN, State.OPEN):
                    if not self._advance(state, handlers, plans, max_seconds):
                        break
            except VisionError as exc:
                print("!! %s -- carrying the cube still counted" % exc)
        self.state = State.DONE
        self.report.cube_final = self.cube_pos()
        return self.report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", default="green", choices=["red", "green", "blue"])
    ap.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer")
    ap.add_argument("--cameras", action="store_true",
                    help="open two live windows: scene_cam and wrist_cam")
    ap.add_argument("--camera-size", default="640x480",
                    help="size of each camera window, WxH (default 640x480)")
    ap.add_argument("--camera-fps", type=float, default=12.0,
                    help="cap on camera-window redraw rate (each pair of 640x480 "
                         "views costs ~65 ms here, so above ~15 fps frames are "
                         "dropped and the run slows down)")
    ap.add_argument("--camera-budget", type=float, default=0.60,
                    help="max share of wall time the windows may spend drawing; "
                         "raise for smoother video, lower to keep the run brisk")
    ap.add_argument("--no-shadows", action="store_true",
                    help="skip the shadow pass in the camera windows (~2x faster)")
    ap.add_argument("--frames", default=None, help="directory for PNG frames")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--no-vision", action="store_true",
                    help="plan from the simulator's ground truth instead of the camera")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    for _ in range(500):                      # let the cubes settle on the table
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    viewer = None
    cameras = None
    try:
        if args.viewer:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(model, data)
        if args.cameras:
            import camera_windows
            w, h = (int(v) for v in args.camera_size.lower().split("x"))
            cameras = camera_windows.CameraWindows(
                model, data, width=w, height=h, fps=args.camera_fps,
                shadows=not args.no_shadows, budget=args.camera_budget)
        report = PickPlace(model, data, args.cube, viewer=viewer,
                           frame_dir=args.frames, cameras=cameras,
                           use_vision=not args.no_vision).run(max_seconds=args.seconds)
    finally:
        if cameras is not None:
            cameras.close()
        if viewer is not None:
            viewer.close()

    print()
    print("=" * 64)
    print("states     :", " -> ".join(report.reached_states))
    print("attempts   :", report.attempts)
    print("grip       :", "cube in the fingers" if report.grasp_confirmed else "no contact")
    print("cube start :", np.round(report.cube_start, 4))
    print("peak height: %.4f m" % report.cube_lifted)
    print("cube final :", np.round(report.cube_final, 4))
    print("bin centre :", np.round(report.bin_center, 4))

    if report.vision_calls:
        errs = np.array(report.vision_error_mm) if report.vision_error_mm else np.zeros(1)
        print("vision     : %d frame(s), cube error mean %.2f mm, max %.2f mm%s"
              % (report.vision_calls, errs.mean(), errs.max(),
                 "" if report.bin_vision_error_mm < 0
                 else ", bin error %.2f mm" % report.bin_vision_error_mm))

    xy_err = float(np.linalg.norm(report.cube_final[:2] - report.bin_center[:2]))
    lifted = report.cube_lifted > 0.045
    inside = xy_err < 0.062 and report.cube_final[2] < 0.05
    print("xy error   : %.1f mm   (bin half-width 62 mm)" % (xy_err * 1000))
    print()
    if lifted and inside:
        print("RESULT: PASS - the cube was grasped, carried and dropped in the bin.")
        return 0
    print("RESULT: FAIL -", "never left the table" if not lifted else "missed the bin")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
