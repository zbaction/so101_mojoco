#!/usr/bin/env python3
"""Pick a cube off the three-cube tower and put *that one* in the bin.

Loads ``mjcf/scene_stack.xml`` (the tower built by ``stack_cubes.py``), uses the
overhead camera to see which cubes are still standing on the tower, and picks
them one at a time *from the top down*.

The robot makes a per-cube decision instead of blindly binning everything:

  * the cube you named  -> goes into the bin, then the run stops;
  * any cube on top of it -> gets **parked on the table** beside the tower, out
    of the way - it is *not* thrown into the bin.

Why top-down: the cubes are stacked, so the only cube a parallel-jaw gripper can
reach without knocking the tower over is the one on top.  "Pick any cube" is
therefore "name the cube you want; the robot clears the ones above it first
(putting them aside), then takes the one you asked for".

    python pick_any.py                 # clear the whole tower into the bin
    python pick_any.py --cube red      # park blue+green aside, bin only red
    python pick_any.py --cube green    # park blue aside, bin only green
    python pick_any.py --viewer        # watch it happen
    python pick_any.py --cameras       # two live camera windows
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco

import vision
from pick_place import PickPlace, Arm, GRIPPER_OPEN

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(ROOT, "mjcf", "scene_stack.xml")

TOWER_XY = np.array([0.215, -0.045])
BIN_XY = np.array([0.215, 0.135])
CUBES = ("red", "green", "blue")

LEVEL0 = 0.010
LEVEL_STEP = 0.020

CARRY_CAP = 0.075
RELEASE_Z = 0.060
BIN_HALF_OK = 0.062

PARK_SLOTS = (np.array([0.180, -0.115]), np.array([0.250, -0.115]))
PARK_CARRY_Z = 0.065
PARK_RELEASE_Z = 0.030


def snap_level(z: float) -> float:
    """Snap a measured cube-centre height to the nearest stack level."""
    k = round((z - LEVEL0) / LEVEL_STEP)
    return LEVEL0 + k * LEVEL_STEP


def cubes_on_tower(per) -> dict:
    """Colour -> Pose for the cubes still on the tower (not parked, not in bin)."""
    poses = per.find_all_cubes_3d()
    out = {}
    for colour, pose in poses.items():
        xy = np.asarray(pose.xy, dtype=float)
        if np.linalg.norm(xy - TOWER_XY) < 0.055 and \
                np.linalg.norm(xy - BIN_XY) > 0.055:
            out[colour] = pose
    return out


def retract(pp: PickPlace, home_q: np.ndarray) -> None:
    """Drive the arm back to its home joint angles, fingers open (joint space)."""
    pp.grip_ref = GRIPPER_OPEN
    pp.q_ref = pp.arm.joints().copy()
    pp._q_target = np.asarray(home_q, dtype=float).copy()
    pp._best_err = float("inf")
    pp._best_time = float(pp.data.time)
    pp._jammed = False
    for _ in range(2000):
        if pp._track(GRIPPER_OPEN):
            break


def choose_destination(colour: str, target, park_used: list):
    """The robot's decision for the cube currently on top of the tower.

    Returns (kind, drop_xy, release_z, carry_z):
      "bin"  -> wanted cube (or no target): put it in the bin
      "park" -> blocker: set it aside on the table
    """
    if target is None or colour == target:
        return "bin", BIN_XY, RELEASE_Z, None
    slot = None
    for cand in PARK_SLOTS:
        if all(np.linalg.norm(cand - u) > 0.05 for u in park_used):
            slot = cand
            break
    if slot is None:
        slot = PARK_SLOTS[-1]
    return "park", slot, PARK_RELEASE_Z, PARK_CARRY_Z


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", default=None, choices=list(CUBES),
                    help="cube to put in the bin; the ones above it are parked "
                         "on the table, not binned (default: clear all into the bin)")
    ap.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer")
    ap.add_argument("--cameras", action="store_true",
                    help="open the two live camera windows")
    ap.add_argument("--camera-size", default="640x480")
    ap.add_argument("--camera-fps", type=float, default=12.0)
    ap.add_argument("--camera-budget", type=float, default=0.60)
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="per-pick simulated-time budget")
    ap.add_argument("--max-picks", type=int, default=4)
    ap.add_argument("--frames", default=None,
                    help="directory to save scene_cam PNGs of each stage")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    for _ in range(800):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    per = vision.Perception(model, data)
    arm = Arm(model, data)
    home_q = arm.joints().copy()

    def save_frame(name: str) -> None:
        if not args.frames:
            return
        os.makedirs(args.frames, exist_ok=True)
        from PIL import Image
        Image.fromarray(per.render()).save(os.path.join(args.frames, name + ".png"))

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

        save_frame("00_tower")
        placed = []
        park_used = []
        target_done = False
        for pick_no in range(args.max_picks):
            on_tower = cubes_on_tower(per)
            if not on_tower:
                print("\n== the tower is clear ==")
                break
            top = max(on_tower, key=lambda c: on_tower[c].z)
            pose = on_tower[top]
            grasp_z = snap_level(pose.z)
            kind, drop_xy, release_z, carry_override = choose_destination(
                top, args.cube, park_used)
            carry_z = (carry_override if carry_override is not None
                       else min(CARRY_CAP, max(0.065, grasp_z + 0.020)))

            print("\n" + "=" * 68)
            print("pick %d: cube_%s on top at xy (%.4f, %.4f) z=%.4f -> grasp %.3f, carry %.3f"
                  % (pick_no + 1, top, pose.xy[0], pose.xy[1], pose.z, grasp_z, carry_z))
            if kind == "bin":
                print("        decision: this is the target -> put it IN THE BIN")
            else:
                print("        decision: this is NOT the target (%s wanted) -> PARK aside at "
                      "(%.3f, %.3f)" % (args.cube, drop_xy[0], drop_xy[1]))
            print("        still on the tower: %s"
                  % ", ".join(sorted(on_tower, key=lambda c: -on_tower[c].z)))

            if kind == "park":
                park_used.append(np.array(drop_xy, dtype=float))

            pp = PickPlace(model, data, top, viewer=viewer, cameras=cameras,
                           use_vision=True, perception=per,
                           grasp_z=grasp_z, carry_z=carry_z, release_z=release_z,
                           drop_xy=(None if kind == "bin" else drop_xy))
            rep = pp.run(max_seconds=args.seconds)
            placed.append((top, kind, rep))
            retract(pp, home_q)
            save_frame("1%d_%s_%s" % (pick_no + 1, kind, top))

            if kind == "bin" and args.cube is not None:
                final = np.asarray(rep.cube_final, dtype=float)
                if top == args.cube and np.linalg.norm(final[:2] - BIN_XY) < BIN_HALF_OK \
                        and final[2] < 0.05:
                    target_done = True
                    break
    finally:
        if cameras is not None:
            cameras.close()
        if viewer is not None:
            viewer.close()

    save_frame("99_final")

    print("\n" + "=" * 68)
    print("SUMMARY")
    all_ok = True
    for colour, kind, rep in placed:
        final = np.asarray(rep.cube_final, dtype=float)
        if kind == "bin":
            xy = float(np.linalg.norm(final[:2] - BIN_XY))
            inside = xy < BIN_HALF_OK and final[2] < 0.05
            all_ok = all_ok and inside
            print("  cube_%-5s  final %s  xy-to-bin %5.1f mm  %s"
                  % (colour, np.round(final, 4), xy * 1000,
                     "IN BIN" if inside else "MISSED BIN"))
        else:
            slot = min(PARK_SLOTS, key=lambda s: np.linalg.norm(final[:2] - s))
            off = float(np.linalg.norm(final[:2] - slot))
            print("  cube_%-5s  final %s  PARKED on table (%.0f mm from its slot)"
                  % (colour, np.round(final, 4), off * 1000))

    if args.cube is not None:
        if not target_done:
            print("\nRESULT: FAIL - cube_%s never reached the bin" % args.cube)
            return 1
        parked = [c for (c, k, _r) in placed if k == "park"]
        if parked:
            print("\nRESULT: PASS - only cube_%s is in the bin; %s parked on the table."
                  % (args.cube, ", ".join("cube_" + c for c in parked)))
        else:
            print("\nRESULT: PASS - cube_%s is in the bin." % args.cube)
        return 0
    if all_ok and placed:
        print("\nRESULT: PASS - all %d cube(s) cleared into the bin." % len(placed))
        return 0
    print("\nRESULT: FAIL - at least one cube missed the bin.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
