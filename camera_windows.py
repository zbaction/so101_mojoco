"""Two live camera windows: the scene camera and the wrist camera.

Launched from `pick_place.py --cameras`, or standalone:

    python camera_windows.py                 # idle scene, both windows
    python camera_windows.py --size 800x600  # bigger windows

Why a separate renderer instead of reusing the perception one: `mujoco.Renderer`
fixes its size at construction and has no resize, so showing 640x480 frames out
of the 1280x960 perception renderer would mean rendering 1280x960 and throwing
three quarters of the pixels away -- 70 ms instead of 44 ms for the pair.

Measured cost on this machine (WSLg, software GL), for the pair of views:
    two 1280x960 -> 135 ms      two 640x480 -> 70 ms
    two 640x480 with the shadow pass skipped -> 25 ms
    two 512x384 with shadows off -> 47 ms (pair), 21 fps
Because each frame pair costs real milliseconds, redrawing has to be budgeted:
the owner throttles it by both an fps cap and a share of wall time (see
`budget`), otherwise a 3 s animation takes 10 s.

One WSLg quirk is guarded against, the same one that bit perception: a readback
can come back as an exact vertical mirror.  `_sanitise` catches it two ways --
an absolute check for the scene camera (the wooden table must project into the
lower half of the frame) and a relative check for either camera (a frame that
matches the previous frame *upside down* better than upright is mirrored).
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk

import vision

SCENE = "/home/zhangb/so101_mujoco/mjcf/scene.xml"

# (camera name in the MJCF, window title)
DEFAULT_CAMERAS = (("scene_cam", "全景相机  scene_cam"),
                   ("wrist_cam", "腕部相机  wrist_cam"))


class CameraWindows:
    """Two tkinter windows showing two MuJoCo cameras, refreshed by tick()."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 cameras=DEFAULT_CAMERAS, width: int = 640, height: int = 480,
                 fps: float = 12.0, shadows: bool = True, budget: float = 0.60):
        self.model, self.data = model, data
        self.cameras = list(cameras)
        self.width, self.height = width, height
        self.fps = fps
        # Share of real time this display is allowed to consume.  The owner reads
        # it to decide when to skip a redraw; kept here so both sides agree.
        self.budget = budget
        self.closed = False
        self.frames_shown = 0
        self.dropped = 0            # frames skipped to keep up with real time
        self.flips = 0
        self._prev: dict[str, np.ndarray] = {}
        self._font = ImageFont.load_default(size=18)

        self.renderer = mujoco.Renderer(model, height, width)
        if not shadows:
            # The shadow pass is most of the raster time here (44.8 -> 14.8 ms
            # at 640x480), so it is worth turning off for a display-only view.
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.renderer.update_scene(data, camera=self.cameras[0][0])
        self.renderer.render()          # burn the first frame (WSLg, see vision.py)

        self._root = tk.Tk()
        self._root.withdraw()
        self._root.title("SO-101 cameras")
        self._labels: dict[str, tk.Label] = {}
        self._windows: dict[str, tk.Toplevel] = {}
        self._photos: dict[str, ImageTk.PhotoImage] = {}
        self._shown: dict[str, Image.Image] = {}   # exactly what the label got

        for i, (cam, title) in enumerate(self.cameras):
            win = tk.Toplevel(self._root)
            win.title(title)
            win.geometry("%dx%d+%d+%d" % (width, height, 40 + i * (width + 40), 40))
            win.protocol("WM_DELETE_WINDOW", self.close)
            label = tk.Label(win, bg="black", borderwidth=0, highlightthickness=0)
            label.pack(fill="both", expand=True)
            self._windows[cam] = win
            self._labels[cam] = label
        self._root.bind_all("<Escape>", lambda _e: self.close())

        # Absolute orientation reference for the scene camera: the table centre
        # must land in the lower half of the image.
        self._table_row = None
        if any(cam == "scene_cam" for cam, _ in self.cameras):
            try:
                camobj = vision.Camera(model, data, "scene_cam", width, height)
                self._table_row = float(camobj.project(np.array(
                    [[vision.TABLE_CENTRE_XY[0], vision.TABLE_CENTRE_XY[1], 0.0]]))[0, 1])
            except Exception:
                self._table_row = None
        print("[cameras] %d window(s) %dx%d, <=%.0f fps, <=%.0f%% of wall time"
              % (len(self.cameras), width, height, fps, 100 * budget))

    # -- frame plumbing ---------------------------------------------------
    def _rendered(self, cam: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=cam)
        return np.ascontiguousarray(self.renderer.render())

    def _mirrored_scene_cam(self, img: np.ndarray) -> bool:
        """Absolute test: where does the wooden table sit in a scene_cam frame?"""
        if self._table_row is None or img.shape[1] < 8:
            return False
        # Half resolution: 2 ms instead of 10 ms on every displayed frame, and
        # the two candidate rows are hundreds of pixels apart, so the test is
        # just as unambiguous (same tradeoff as Perception._upside_down).
        q = 2
        hue, sat, val = vision.hsv_planes(img[::q, ::q])
        wood = (hue >= 12) & (hue <= 52) & (sat > 0.25) & (val > 0.12)
        ys, _ = np.nonzero(wood)
        if len(ys) < 60:
            return False
        v_obs = q * (float(ys.mean()) + 0.5) - 0.5
        return abs(v_obs - (self.height - 1 - self._table_row)) < abs(v_obs - self._table_row)

    def _sanitise(self, cam: str, img: np.ndarray) -> np.ndarray:
        """Reject black readbacks and undo vertically mirrored ones."""
        for _ in range(2):
            if img.mean() > 40.0:
                break
            img = self._rendered(cam)

        if cam == "scene_cam" and self._mirrored_scene_cam(img):
            img = np.ascontiguousarray(img[::-1])
            self.flips += 1
            return img

        prev = self._prev.get(cam)
        if prev is None:
            # Bootstrap: two renders of the same frozen state must agree.
            again = self._rendered(cam)
            same = float(np.abs(again - img).mean())
            flipped = float(np.abs(again[::-1] - img).mean())
            if flipped < 0.5 * same:
                img = np.ascontiguousarray(img[::-1])
                self.flips += 1
        else:
            same = float(np.abs(img - prev).mean())
            flipped = float(np.abs(img[::-1] - prev).mean())
            if flipped < 0.5 * same and same > 12.0:
                img = np.ascontiguousarray(img[::-1])
                self.flips += 1
        return img

    def grab(self, cam: str) -> np.ndarray:
        """One sanitised display frame, for saving or for --frames captures."""
        return self._sanitise(cam, self._rendered(cam))

    # -- the loop ---------------------------------------------------------
    def alive(self) -> bool:
        if self.closed:
            return False
        for win in self._windows.values():
            try:
                if not win.winfo_exists():
                    return False
            except tk.TclError:
                return False
        return True

    def tick(self, caption: str = "") -> float:
        """Redraw both windows once.  Returns the wall time it took, in seconds."""
        if not self.alive():
            self.closed = True
            return 0.0
        t0 = time.perf_counter()
        stamp = "%s   t = %.2f s" % (caption, float(self.data.time)) if caption else \
                "t = %.2f s" % float(self.data.time)
        for cam, _title in self.cameras:
            img = self._sanitise(cam, self._rendered(cam))
            self._prev[cam] = img
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            tw = draw.textlength(stamp, font=self._font)
            draw.rectangle([8, pil.height - 34, 8 + tw + 16, pil.height - 8], fill=(0, 0, 0))
            draw.text((16, pil.height - 31), stamp, fill=(255, 255, 255), font=self._font)
            photo = ImageTk.PhotoImage(pil)
            self._labels[cam].configure(image=photo)
            self._photos[cam] = photo         # keep a reference or it is collected
            self._shown[cam] = pil
        try:
            self._root.update()
        except tk.TclError:
            self.closed = True
            return time.perf_counter() - t0
        self.frames_shown += 1
        return time.perf_counter() - t0

    def save_shown(self, directory: str) -> list[str]:
        """Write the last displayed image of each window (overlay included)."""
        import os
        os.makedirs(directory, exist_ok=True)
        out = []
        for cam, img in self._shown.items():
            path = os.path.join(directory, "%s.png" % cam)
            img.save(path)
            out.append(path)
        return out

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._root.destroy()
        except tk.TclError:
            pass
        try:
            self.renderer.close()
        except Exception:
            pass
        extra = ""
        if self.dropped:
            extra += ", %d frame(s) dropped to keep real time" % self.dropped
        if self.flips:
            extra += ", %d mirrored frame(s) corrected" % self.flips
        print("[cameras] closed after %d frame(s)%s" % (self.frames_shown, extra))


def _demo(args) -> int:
    w, h = (int(v) for v in args.size.lower().split("x"))
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    wins = CameraWindows(model, data, width=w, height=h, fps=args.fps,
                         shadows=not args.no_shadows)
    period = 1.0 / max(args.fps, 1.0)
    t_start = time.perf_counter()
    costs = []
    try:
        while wins.alive():
            t0 = time.perf_counter()
            wins.tick(caption="idle scene")
            costs.append(time.perf_counter() - t0)
            rest = period - costs[-1]
            if rest > 0:
                time.sleep(min(rest, 0.05))
            if args.seconds and time.perf_counter() - t_start >= args.seconds:
                break
    finally:
        if args.save:
            for path in wins.save_shown(args.save):
                print("[cameras] saved %s" % path)
        wins.close()
    if costs:
        costs = sorted(costs)
        print("[cameras] %d frame pair(s), median %.1f ms, max %.1f ms"
              % (len(costs), 1000 * costs[len(costs) // 2], 1000 * costs[-1]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="640x480", help="window size, WxH")
    ap.add_argument("--fps", type=float, default=12.0,
                    help="redraw cap; two 640x480 views cost ~65 ms here, so "
                         "much above 15 fps the refresh starts lagging")
    ap.add_argument("--no-shadows", action="store_true",
                    help="skip the shadow pass: ~2x faster here (7.5 -> 15 fps "
                         "for two 640x480 windows), slightly flatter image")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="auto-close after N seconds (0 = run until closed)")
    ap.add_argument("--save", default=None,
                    help="write the last frame of each window to this directory")
    return _demo(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
