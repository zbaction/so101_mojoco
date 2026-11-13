"""Render the two cameras of the SO-101 pick scene to PNG files.

Usage:
    python capture_cameras.py [output_dir]

Writes scene_cam.png and wrist_cam.png (640x480).  Set ARM_POSE=grasp in the
environment to render the arm at the verified top-down grasp pose instead of
the default home pose.
"""
import os
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(ROOT, "mjcf", "scene.xml")
WIDTH, HEIGHT = 640, 480
SETTLE_STEPS = 600

# Verified top-down grasp pose for cube_green (see mjcf/README notes).
GRASP_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRASP_Q = [0.215, 0.179, 0.107, 1.284, 0.264]


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else ROOT
    os.makedirs(out_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    # Let the free-floating cubes settle onto the tabletop.
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    if os.environ.get("ARM_POSE", "").lower() == "grasp":
        for name, value in zip(GRASP_JOINTS, GRASP_Q):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    for cam in ("scene_cam", "wrist_cam"):
        renderer.update_scene(data, camera=cam)
        image = renderer.render()
        path = os.path.join(out_dir, f"{cam}.png")
        Image.fromarray(image).save(path)
        print(f"wrote {path}")
    renderer.close()


if __name__ == "__main__":
    main()
