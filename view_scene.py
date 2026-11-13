"""Open the SO-101 tabletop pick scene in the interactive MuJoCo viewer.

Usage:
    python view_scene.py

Controls (MuJoCo passive viewer):
    - Left drag: rotate, Right drag / scroll: zoom, Middle drag: pan
    - Tab / [ ]: cycle the scene_cam and wrist_cam fixed views
    - Double-click a body: track it
"""
import os
import time

import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(ROOT, "mjcf", "scene.xml")
REALTIME = 1.0


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"loaded {SCENE}")
    print(f"  bodies={model.nbody} geoms={model.ngeom} cameras={model.ncam} actuators={model.nu}")
    print(f"  cameras: {[model.camera(i).name for i in range(model.ncam)]}")
    print("  close the viewer window to quit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            mujoco.mj_step(model, data)
            viewer.sync()

            # Keep the simulation close to real time.
            wait = model.opt.timestep / REALTIME - (time.perf_counter() - step_start)
            if wait > 0:
                time.sleep(wait)


if __name__ == "__main__":
    main()
