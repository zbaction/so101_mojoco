import os

import mujoco
import mujoco.viewer
import numpy as np

# 加载模型
ROOT = os.path.dirname(os.path.abspath(__file__))
model = mujoco.MjModel.from_xml_path(
    os.path.join(ROOT, "SO-ARM100/Simulation/SO101/scene.xml")
)

# 创建仿真数据
data = mujoco.MjData(model)

# 末端目标位置（米），可以直接修改为想要到达的位置。
target_position = np.array([0.2, 0.2, 0.3])
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
joint_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
joint_ids = np.array(
    [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
)
dof_ids = model.jnt_dofadr[joint_ids]
qpos_ids = model.jnt_qposadr[joint_ids]


def solve_position_ik(
    target: np.ndarray,
    initial_qpos: np.ndarray,
    iterations: int = 100,
    damping: float = 1e-3,
) -> np.ndarray:
    """Use damped least squares and mj_jacSite to solve position-only IK."""
    qpos = initial_qpos.copy()
    jacobian = np.zeros((3, model.nv))

    for _ in range(iterations):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        position_error = target - data.site_xpos[site_id]
        if np.linalg.norm(position_error) < 1e-4:
            break

        mujoco.mj_jacSite(model, data, jacobian, None, site_id)
        position_jacobian = jacobian[:, dof_ids]
        update = position_jacobian.T @ np.linalg.solve(
            position_jacobian @ position_jacobian.T
            + damping**2 * np.eye(3),
            position_error,
        )
        qpos[qpos_ids] += update
        qpos[qpos_ids] = np.clip(
            qpos[qpos_ids],
            model.jnt_range[joint_ids, 0],
            model.jnt_range[joint_ids, 1],
        )

    return qpos


mujoco.mj_forward(model, data)
ik_qpos = solve_position_ik(target_position, data.qpos)
target_ctrl = data.ctrl.copy()
for actuator_id in range(model.nu):
    joint_id = model.actuator_trnid[actuator_id, 0]
    target_ctrl[actuator_id] = ik_qpos[model.jnt_qposadr[joint_id]]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = target_ctrl

        mujoco.mj_step(model, data)
        viewer.sync()