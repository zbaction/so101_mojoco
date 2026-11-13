# SO-101 桌面抓取场景 (MuJoCo)

桌面抓取场景，包含 SO-101 六轴机械臂、三色方块、目标收纳盒，以及场景相机与腕部相机。
另附一个**六状态自主抓取-放置状态机**：不依赖人工干预，机器人自己把方块夹起来放进收纳盒。

## 快速开始

```bash
cd ~/so101_mujoco
source .venv/bin/activate

python view_scene.py            # 打开交互式 MuJoCo 查看器
python capture_cameras.py       # 把两个相机渲染成 scene_cam.png / wrist_cam.png
ARM_POSE=grasp python capture_cameras.py    # 在抓取姿态下渲染

python pick_place.py                     # 自主抓取+放置（默认绿色方块）
python pick_place.py --cube red          # 换红色方块
python pick_place.py --viewer            # 边跑边看
python pick_place.py --frames out/       # 每个状态导出一张 PNG
```

## 文件

| 文件 | 说明 |
| --- | --- |
| `mjcf/scene.xml` | 场景主文件：桌面、三色方块、收纳盒、灯光、`scene_cam` |
| `mjcf/so101_arm.xml` | SO-101 机器人（含腕部相机模块），内含 `wrist_cam` |
| `pick_place.py` | 六状态自主抓取-放置控制器 |
| `view_scene.py` | 交互式查看器入口 |
| `capture_cameras.py` | 离屏渲染两个相机到 PNG |

## 场景内容

- **桌面**：0.6 × 0.6 m 木纹台面，台面位于世界坐标 `z = 0`，中心 `(0.18, 0, 0)`。
- **方块**：边长 20 mm、质量 20 g、`condim=4`（带扭转摩擦，便于夹取）。
  - `cube_red`   `(0.200, -0.120)`
  - `cube_green` `(0.235, -0.045)`
  - `cube_blue`  `(0.215,  0.030)`
- **收纳盒** `bin`：130 × 130 mm、内壁高 35 mm，中心 `(0.215, 0.135)`，内部有 `bin_center` 站点。
- **场景相机** `scene_cam`：`pos="0.5360 -0.3660 0.4110"`，`fovy="40"`，分辨率 640×480，俯视整个工作区。
- **腕部相机** `wrist_cam`：装在 `wrist_camera` body 上（镜头面外移 23.5 mm），光轴沿手指方向，`fovy="62"`。

> ⚠️ **`fovy` 的单位永远是度**，不受 `<compiler angle="radian"/>` 影响。写成 `0.65` 会得到 0.65° 的视场，画面几乎全空。

## 机械臂

6 个位置执行器（`class="sts3215"`，`kp=998.22`、`kv=2.731`）：

```
shoulder_pan  shoulder_lift  elbow_flex  wrist_flex  wrist_roll  gripper
```

工作空间实测（以 `gripperframe` 站点为 TCP，垂直向下抓取姿态）：

- 可达范围主要落在 `x ≈ 0.14–0.30`、`y ≈ ±0.15`、`z < 0.09`（世界坐标）；
- `z > 0.10` 时姿态可达性急剧变差，`x > 0.30` 基本无法竖直抓取；
- 夹爪命令 `gripper ≈ 0.2–0.5` 时开口约 24–29 mm，适合夹 20 mm 方块。

已验证的抓取姿态（绿色方块，TCP 误差 < 2 mm）：

```
shoulder_pan 0.215, shoulder_lift 0.179, elbow_flex 0.107, wrist_flex 1.284, wrist_roll 0.264
```

## 自主抓取-放置 (`pick_place.py`)

### 六个状态

| 状态 | 动作 | 完成条件 |
| --- | --- | --- |
| `MOVE_ABOVE` | 张开夹爪移到方块正上方 65 mm | 关节到达目标并稳定 |
| `DESCEND` | 垂直下降到抓取高度 | 关节到达目标 |
| `CLOSE` | 闭合夹爪 | 夹爪被物体顶住（指令与实际差 > 0.05 rad 持续 40 ms）或到机械限位 |
| `LIFT` | 垂直抬起 | 关节到达目标 |
| `MOVE_TO_BIN` | 水平搬到收纳盒上方 | 关节到达目标 |
| `OPEN` | 张开夹爪放下 | 张到位且方块已落回低处 |

若 `LIFT` 结束后方块没有真的离开桌面（高度上升 < 30 mm），会带着 ±4 mm 的横向偏移重试，最多 3 次。

### 几何要点（这两个数不对就一定抓空）

1. **`PINCH_OFFSET = 0.013`**：手指真正的捏合点不在 `gripperframe` 站点原点上，而在站点 **+z 轴前方 13 mm** 处。所以 IK 的目标必须是「方块中心往回退 13 mm」，否则手指伸出去旁边夹空气。
2. **`GRASP_SITE_Z = 0.010`**：站点必须降到**方块中心高度**。手指尖在站点下方约 7 mm，站点若停在方块顶面（z≈0.02），夹爪闭合的是方块上方的空气。

### 实测结果

```
python pick_place.py --cube red      RESULT: PASS   峰值 64.7 mm  落点误差 4.2 mm
python pick_place.py --cube green    RESULT: PASS   峰值 65.4 mm  落点误差 5.1 mm
python pick_place.py --cube blue     RESULT: PASS   峰值 64.6 mm  落点误差 7.9 mm
```

随机位置鲁棒性测试（绿色方块在 `x ∈ [0.16, 0.26]`、`y ∈ [-0.10, 0.03]`、随机偏航内取 12 个位置）：**12/12 一次成功**，峰值高度 64–73 mm，落点误差 5–8 mm（收纳盒半宽 62 mm）。

> 注意：方块之间如果贴得太近（间距 < 25 mm），抬起时会把相邻方块一起带起来，这属于场景摆放问题，不是控制器问题。

### 实现要点

- **IK**：阻尼最小二乘 + `mj_jacSite`，同时约束位置与姿态（approach 沿世界 −Z，夹爪闭合轴沿世界 +X），多随机种子重试。
- **运动**：关节参考以 1.2 rad/s 斜坡推进，夹爪 0.5 rad/s，避免瞬跳造成冲击。
- **状态完成判据**：与实际关节角度比较**最终目标**（不是移动中的参考），否则状态会在第一帧就误判完成。
- **抓取确认**：靠「夹爪指令领先实际 > 0.05 rad 且持续 40 ms」判断顶到物体，而不是只看接触对数量（接触会瞬时出现又消失）。
