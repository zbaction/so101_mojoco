"""Visual perception for the SO-101 pick-and-place scene.

Everything here works from rendered pixels only -- no body position is ever read
out of the physics state.  The objects are known solids on a known plane, which
is what makes a single RGB image enough:

  * the cube is a 20 mm box resting on the table, so (x, y, yaw) fully determine
    its silhouette.  We recover those three numbers by maximising the IoU
    between the projected box hull and the observed colour mask (exact convex
    polygon clipping, no rasterisation, no gradients);
  * yaw is folded into [0, pi/2) because a square box is unchanged by a quarter
    turn -- the same symmetry the parallel-jaw gripper wants;
  * the working surface is found from the wood texture, and every colour blob
    must lie inside it.  Without that region of interest the blue sky is easily
    mistaken for the blue cube.

Nothing in this module imports cv2; only numpy / scipy / mujoco.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import ndimage, optimize

# cube colour -> hue range in degrees (red wraps through 0)
COLOUR_HUE = {
    "red": (345.0, 15.0),
    "green": (110.0, 180.0),
    "blue": (195.0, 260.0),
}

CUBE_HALF = 0.010        # m: half edge of the 20 mm cube
CUBE_CENTRE_Z = 0.0102   # m: cube centre height above the table top
CUBE_TOP_Z = 0.0202      # m: top face height

BIN_HALF = 0.065         # m: half width of the bin's outer wall
BIN_TOP_Z = 0.035        # m: height of the bin's rim above the table

# Table centre in world xy.  Used only to sanity-check the orientation of a
# rendered frame: the table top is the one large object whose image position is
# known in advance, so it is the natural anchor for "is this frame upside down".
TABLE_CENTRE_XY = (0.18, 0.0)

_CUBE_CORNERS = np.array([[sx, sy, sz]
                          for sx in (-CUBE_HALF, CUBE_HALF)
                          for sy in (-CUBE_HALF, CUBE_HALF)
                          for sz in (-CUBE_HALF, CUBE_HALF)], float)
_TOP_CORNERS = np.array([[-CUBE_HALF, -CUBE_HALF],
                         [CUBE_HALF, -CUBE_HALF],
                         [CUBE_HALF, CUBE_HALF],
                         [-CUBE_HALF, CUBE_HALF]], float)


@dataclass
class Pose:
    """A planar pose recovered from pixels."""
    xy: np.ndarray            # m, world frame
    yaw: float                # rad, folded into [0, pi/2)
    score: float              # IoU of the fitted model against the mask
    pixels: int               # mask area in px
    colour: str = ""

    def __repr__(self) -> str:
        return ("Pose(%s xy=(%.4f, %.4f) yaw=%.1f deg IoU=%.3f px=%d)"
                % (self.colour, self.xy[0], self.xy[1], np.degrees(self.yaw),
                   self.score, self.pixels))


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------
def hsv_planes(img: np.ndarray):
    """Hue (deg), saturation, value as float32 planes.  Pure numpy.

    Written with whole-array operations: the obvious "compute a mask, then
    assign into an index" version costs 2.4x more on a 1280x960 frame, which
    matters because this runs on every control cycle.
    """
    f = img.astype(np.float32) * (1.0 / 255.0)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    inv = 1.0 / np.maximum(d, 1e-6)
    rc = (mx - r) * inv
    gc = (mx - g) * inv
    bc = (mx - b) * inv
    hue = np.where(mx == r, bc - gc,
                   np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc)) * 60.0
    hue = np.where(hue < 0.0, hue + 360.0, hue)
    hue = np.where(d > 1e-6, hue, 0.0).astype(np.float32)
    sat = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0).astype(np.float32)
    return hue, sat, mx


def hull2d(points: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain), counter-clockwise, no repeated endpoint."""
    p = np.unique(np.round(np.asarray(points, float), 9), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in p[::-1]:
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])


def silhouette_outline(mask: np.ndarray) -> np.ndarray:
    """Convex outline of a binary mask, without touching every set pixel.

    A connected blob's hull is spanned by the leftmost and rightmost pixel of
    each row plus the topmost and bottommost of each column, so four `argmax`
    passes give the same hull as feeding all 50 000 pixels to `hull2d` -- 40x
    faster (1.2 ms vs 161 ms at 1280x960) and bit-identical in practice.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return np.zeros((0, 2))
    left = np.argmax(mask[rows], axis=1)
    right = mask.shape[1] - 1 - np.argmax(mask[rows][:, ::-1], axis=1)
    cols = np.flatnonzero(mask.any(axis=0))
    top = np.argmax(mask[:, cols], axis=0)
    bottom = mask.shape[0] - 1 - np.argmax(mask[::-1, cols], axis=0)
    pts = np.column_stack([
        np.concatenate([left, right, cols, cols]).astype(float),
        np.concatenate([rows, rows, top, bottom]).astype(float),
    ])
    return hull2d(pts)


def polygon_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_signed_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ccw(p: np.ndarray) -> np.ndarray:
    """Same polygon, wound counter-clockwise (in the abstract xy plane)."""
    p = np.asarray(p, float)
    return p[::-1].copy() if polygon_signed_area(p) < 0 else p


def cw(p: np.ndarray) -> np.ndarray:
    """Same polygon, wound clockwise (in the abstract xy plane)."""
    p = np.asarray(p, float)
    return p[::-1].copy() if polygon_signed_area(p) > 0 else p


def clip_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: the part of `subject` inside convex `clipper`."""
    out = [tuple(p) for p in subject]
    n = len(clipper)
    for i in range(n):
        a = clipper[i]
        b = clipper[(i + 1) % n]
        if not out:
            return np.zeros((0, 2))
        ex, ey = b[0] - a[0], b[1] - a[1]

        def side(p):
            return ex * (p[1] - a[1]) - ey * (p[0] - a[0])

        inp = out
        out = []
        s = inp[-1]
        ss = side(s)
        for e in inp:
            es = side(e)
            if es >= 0:
                if ss < 0:
                    t = ss / (ss - es)
                    out.append((s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])))
                out.append(e)
            elif ss >= 0:
                t = ss / (ss - es)
                out.append((s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])))
            s, ss = e, es
    return np.array(out) if out else np.zeros((0, 2))


def _kmeans1d(v: np.ndarray, k: int = 3, iters: int = 25):
    """Deterministic 1-D k-means; used to split a cube into its bright top face
    and its darker side faces."""
    c = np.quantile(v, np.linspace(0.5 / k, 1.0 - 0.5 / k, k))
    lab = np.zeros(len(v), int)
    for _ in range(iters):
        lab = np.abs(v[:, None] - c[None, :]).argmin(1)
        for j in range(k):
            if np.any(lab == j):
                c[j] = v[lab == j].mean()
    return lab, c


# --------------------------------------------------------------------------
# the camera
# --------------------------------------------------------------------------
class Camera:
    """Pinhole projection for one MuJoCo camera at an arbitrary render size."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 name: str = "scene_cam", width: int = 1280, height: int = 960):
        self.model, self.data = model, data
        self.name = name
        self.width, self.height = width, height
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.id < 0:
            raise ValueError("no camera named %r" % name)
        fovy = np.radians(float(model.cam_fovy[self.id]))
        self.focal = (height / 2.0) / np.tan(fovy / 2.0)
        self.fovy = fovy

    # -- intrinsics follow the current data state (cameras can be attached) --
    @property
    def pos(self) -> np.ndarray:
        return self.data.cam_xpos[self.id].copy()

    @property
    def mat(self) -> np.ndarray:
        return self.data.cam_xmat[self.id].reshape(3, 3).copy()

    def project(self, pts: np.ndarray) -> np.ndarray:
        """World points (N,3) -> pixels (N,2).  Behind-camera points are clamped."""
        d = np.asarray(pts, float) - self.pos
        m = self.mat
        zc = d @ m[:, 2]
        zc = np.minimum(zc, -1e-6)
        u = self.width / 2.0 + self.focal * ((d @ m[:, 0]) / -zc)
        v = self.height / 2.0 - self.focal * ((d @ m[:, 1]) / -zc)
        return np.stack([u, v], 1)

    def unproject(self, u, v, z: float) -> np.ndarray:
        """Pixel(s) at known world height z -> world xy (N,2)."""
        u = np.atleast_1d(np.asarray(u, float))
        v = np.atleast_1d(np.asarray(v, float))
        xc = (u - self.width / 2.0) / self.focal
        yc = -(v - self.height / 2.0) / self.focal
        d = np.stack([xc, yc, -np.ones_like(xc)], 1) @ self.mat.T
        t = (z - self.pos[2]) / d[:, 2]
        p = self.pos[None, :] + t[:, None] * d
        return p[:, :2]


# --------------------------------------------------------------------------
# perception
# --------------------------------------------------------------------------
class Perception:
    """Colour-based detection of the cubes and the bin, from one RGB frame."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 width: int = 1280, height: int = 960, camera: str = "scene_cam"):
        self.model, self.data = model, data
        self.cam = Camera(model, data, camera, width, height)
        self.renderer = mujoco.Renderer(model, height, width)
        self.table_z = 0.0
        self.bin_floor_z = 0.005
        self.bin_top_z = BIN_TOP_Z
        # The first frame drawn by a fresh GL context is occasionally garbage
        # under WSLg (nearly black, mean ~15 instead of ~107).  Burning one
        # frame here makes every later call reliable; render() also re-tries.
        self.render()

    # -- I/O -------------------------------------------------------------
    def _upside_down(self, img: np.ndarray) -> bool:
        """True if the frame is row-reversed, judged by where the table sits.

        Half-resolution HSV is plenty for one centroid, and cheap enough to run
        on every frame.
        """
        # Quarter resolution: 3.9 ms instead of 85 ms, and the measured
        # centroid is identical (upright 631, mirrored 329 at 1/2, 1/4 and 1/8).
        q = 4
        hue, sat, val = hsv_planes(img[::q, ::q])
        wood = (hue >= 12) & (hue <= 52) & (sat > 0.25) & (val > 0.12)
        ys, _ = np.nonzero(wood)
        if len(ys) < 60:                  # too little table to judge: trust it
            return False
        v_obs = q * (ys.mean() + 0.5) - 0.5
        v_ref = self.cam.project(
            np.array([[TABLE_CENTRE_XY[0], TABLE_CENTRE_XY[1], 0.0]]))[0, 1]
        # Compare against where the table centre should be, upright vs mirrored.
        return abs(v_obs - (self.cam.height - 1 - v_ref)) < abs(v_obs - v_ref)

    def render(self, camera: str | None = None, retries: int = 2) -> np.ndarray:
        """Grab one RGB frame, always upright.

        Two WSLg quirks are handled here, both measured rather than guessed:
        a fresh GL context can drop its first frame (nearly black), and every
        other readback can come back as an exact vertical mirror.  Either way the
        caller gets a frame in which image row 0 is the top of the scene.
        """
        name = camera or self.cam.name
        img = None
        for _ in range(retries + 1):
            self.renderer.update_scene(self.data, camera=name)
            img = self.renderer.render()
            if img.mean() > 40.0:
                break
        if self._upside_down(img):
            img = np.ascontiguousarray(img[::-1])
        return img

    # -- masks -----------------------------------------------------------
    def working_surface(self, hue, sat, val) -> np.ndarray:
        """The wooden table top, i.e. the region where objects can be.

        Without this the blue sky (a bright, saturated blue) is detected as the
        blue cube whenever the cube happens to be far from the camera.
        """
        m = (hue >= 12) & (hue <= 52) & (sat > 0.25) & (val > 0.12)
        # Morphology at half resolution: the table is one huge smooth blob, so
        # the coarse version finds the same component four times faster.  Do
        # not go below half -- at quarter resolution the coarse closing bridges
        # the table edge into the sky and hole filling then swallows the sky,
        # which is exactly how the blue cube used to be "found" in the sky.
        q = 2
        sm = m[::q, ::q]
        sm = ndimage.binary_closing(sm, np.ones((3, 3)))
        sm = ndimage.binary_fill_holes(sm)
        lab, n = ndimage.label(sm)
        if n == 0:
            return np.ones(m.shape, bool)
        sizes = ndimage.sum(sm, lab, range(1, n + 1))
        sm = lab == (int(np.argmax(sizes)) + 1)
        big = np.repeat(np.repeat(sm, q, 0), q, 1)[:m.shape[0], :m.shape[1]]
        return ndimage.binary_erosion(big, np.ones((5, 5)))

    def colour_blob(self, hue, sat, val, colour: str,
                    roi: np.ndarray | None = None,
                    min_px: int = 120) -> np.ndarray | None:
        lo, hi = COLOUR_HUE[colour]
        if lo > hi:
            m = (hue >= lo) | (hue <= hi)
        else:
            m = (hue >= lo) & (hue <= hi)
        m = m & (sat > 0.45) & (val > 0.30)
        if roi is not None:
            m = m & roi
        m = ndimage.binary_opening(m, np.ones((3, 3)))
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        if n == 0:
            return None
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        if sizes.max() < min_px:
            return None
        return lab == (int(np.argmax(sizes)) + 1)

    def grey_blob(self, hue, sat, val, roi=None, min_px: int = 400):
        """The bin: dark, unsaturated grey -- clearly separated from the wood."""
        m = (sat < 0.25) & (val > 0.08) & (val < 0.60)
        if roi is not None:
            m = m & roi
        m = ndimage.binary_opening(m, np.ones((5, 5)))
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        if n == 0:
            return None
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        if sizes.max() < min_px:
            return None
        return lab == (int(np.argmax(sizes)) + 1)

    def top_face_mask(self, blob: np.ndarray, val: np.ndarray) -> np.ndarray | None:
        """Brightest cluster of a cube = the face pointing at the ceiling.

        Splitting the cube's pixels by brightness is what removes the side faces'
        contribution: seen from above, the top face is a clean 20x20 square.
        """
        v = val[blob]
        if len(v) < 45:
            return None
        lab, centres = _kmeans1d(v, 3)
        idx = np.nonzero(blob)
        keep = lab == int(np.argmax(centres))
        m = np.zeros_like(blob)
        m[idx[0][keep], idx[1][keep]] = True
        lab2, n2 = ndimage.label(m, structure=np.ones((3, 3)))
        if n2 == 0:
            return None
        sizes = ndimage.sum(m, lab2, range(1, n2 + 1))
        return lab2 == (int(np.argmax(sizes)) + 1)

    # -- model-based fitting ---------------------------------------------
    def box_outline(self, xy, yaw: float, top_face: bool = False) -> np.ndarray:
        """Projected outline of the cube model, wound counter-clockwise.

        The winding matters: clip_polygon treats "inside" as the left of every
        edge, so a clockwise outline clips everything away and every IoU comes
        back as zero.  The projection flips handedness (image y points down),
        so the raw projected points must be re-wound, not assumed correct.
        """
        c, s = np.cos(yaw), np.sin(yaw)
        if top_face:
            wx = c * _TOP_CORNERS[:, 0] - s * _TOP_CORNERS[:, 1] + xy[0]
            wy = s * _TOP_CORNERS[:, 0] + c * _TOP_CORNERS[:, 1] + xy[1]
            pts = np.column_stack([wx, wy, np.full(4, CUBE_TOP_Z)])
            return ccw(self.cam.project(pts))
        wx = c * _CUBE_CORNERS[:, 0] - s * _CUBE_CORNERS[:, 1] + xy[0]
        wy = s * _CUBE_CORNERS[:, 0] + c * _CUBE_CORNERS[:, 1] + xy[1]
        wz = _CUBE_CORNERS[:, 2] + CUBE_CENTRE_Z
        return ccw(hull2d(self.cam.project(np.column_stack([wx, wy, wz]))))

    def fit_pose(self, mask: np.ndarray, colour: str = "",
                 top_face: bool = False) -> Pose | None:
        """Recover (x, y, yaw) by maximising model/mask overlap."""
        ys, xs = np.nonzero(mask)
        if len(xs) < 40:
            return None
        obs = silhouette_outline(mask)
        obs_area = polygon_area(obs)
        if obs_area <= 0:
            return None
        plane_z = CUBE_TOP_Z if top_face else CUBE_CENTRE_Z
        seed = self.cam.unproject(xs.mean(), ys.mean(), plane_z)[0]

        def iou(p) -> float:
            if not (0.10 < p[0] < 0.40 and -0.25 < p[1] < 0.25):
                return 0.0                    # keep the search where the table is
            poly = self.box_outline((p[0], p[1]), p[2] % (np.pi / 2), top_face)
            if len(poly) < 3:
                return 0.0
            inter = polygon_area(clip_polygon(obs, poly))
            return inter / (polygon_area(poly) + obs_area - inter + 1e-12)

        # Coarse yaw sweep, then refine the best distinct candidates: a square
        # silhouette is weak at some angles, so a single start can lock onto a
        # 45-degree-wrong minimum.  Two starts and a 5-degree grid give the same
        # answer as five starts and a 2-degree grid, 2.4x faster (measured).
        grid = np.radians(np.arange(0.0, 90.0, 5.0))
        scored = [(iou([seed[0], seed[1], yaw]), float(yaw)) for yaw in grid]
        scored.sort(reverse=True)

        starts = []
        for score, yaw in scored:
            if score <= 0:
                continue
            if all(abs(((yaw - y2 + np.pi / 2) % np.pi) - np.pi / 2) > np.radians(20)
                   for y2 in starts):
                starts.append(yaw)
            if len(starts) == 2:
                break

        best = (0.0, None)
        h = 0.002
        for yaw in starts:
            x0 = [seed[0], seed[1], yaw]
            simplex = np.array([x0,
                                [x0[0] + h, x0[1], x0[2]],
                                [x0[0], x0[1] + h, x0[2]],
                                [x0[0], x0[1], x0[2] + 0.02]])
            res = optimize.minimize(lambda p: -iou(p), x0, method="Nelder-Mead",
                                    options={"xatol": 1e-5, "fatol": 1e-7,
                                             "maxiter": 300,
                                             "initial_simplex": simplex})
            score = -float(res.fun)
            if score > best[0]:
                best = (score, res.x)
        if best[1] is None:
            return None
        p = best[1]
        return Pose(xy=np.array([p[0], p[1]]), yaw=float(p[2] % (np.pi / 2)),
                    score=float(best[0]), pixels=int(len(xs)), colour=colour)

    # -- the two queries the controller needs ----------------------------
    def find_cube(self, colour: str, img: np.ndarray | None = None,
                  top_face: bool = False) -> Pose | None:
        """Locate one cube.

        The full silhouette is the accurate mode: measured over random poses it
        gives 0.5 mm / 1.1 deg mean error against 2.3 mm / 3.8 deg for the
        top-face-only model.  The top face is small and partly occluded, so its
        rectangle is a poor fit; the whole cube is a clean 20 mm box.
        """
        if img is None:
            img = self.render()
        hue, sat, val = hsv_planes(img)
        roi = self.working_surface(hue, sat, val)
        blob = self.colour_blob(hue, sat, val, colour, roi)
        if blob is None:
            return None
        mask = self.top_face_mask(blob, val) if top_face else blob
        if mask is None or mask.sum() < 40:
            mask = blob
        return self.fit_pose(mask, colour, top_face=top_face and mask is not blob)

    def find_all_cubes(self, img: np.ndarray | None = None) -> dict[str, Pose]:
        if img is None:
            img = self.render()
        hue, sat, val = hsv_planes(img)
        roi = self.working_surface(hue, sat, val)
        out = {}
        for colour in COLOUR_HUE:
            blob = self.colour_blob(hue, sat, val, colour, roi)
            if blob is None:
                continue
            pose = self.fit_pose(blob, colour, top_face=False)
            if pose is not None:
                out[colour] = pose
        return out

    def bin_outline(self, xy, yaw: float) -> np.ndarray:
        """Projected outline of the open box that is the bin, wound CCW."""
        c, s = np.cos(yaw), np.sin(yaw)
        pts = []
        for z in (self.bin_floor_z - 0.005, self.bin_top_z):
            for sx, sy in ((-BIN_HALF, -BIN_HALF), (BIN_HALF, -BIN_HALF),
                           (BIN_HALF, BIN_HALF), (-BIN_HALF, BIN_HALF)):
                pts.append([c * sx - s * sy + xy[0], s * sx + c * sy + xy[1], z])
        return ccw(hull2d(self.cam.project(np.array(pts))))

    def find_bin(self, img: np.ndarray | None = None) -> Pose | None:
        """Centre of the bin, by fitting its box model to its grey silhouette.

        Back-projecting the silhouette centroid instead lands 13 mm off: the
        walls are seen from one side only, so the visible area is not centre-
        symmetric.  Fitting the known box (130 x 130 x 35 mm) removes that bias.
        """
        if img is None:
            img = self.render()
        hue, sat, val = hsv_planes(img)
        roi = self.working_surface(hue, sat, val)
        blob = self.grey_blob(hue, sat, val, roi)
        if blob is None:
            return None
        ys, xs = np.nonzero(blob)
        if len(xs) < 400:
            return None
        obs = silhouette_outline(blob)
        obs_area = polygon_area(obs)
        if obs_area <= 0:
            return None
        seed = self.cam.unproject(xs.mean(), ys.mean(), 0.5 * self.bin_top_z)[0]

        def iou(q) -> float:
            if not (0.05 < q[0] < 0.40 and -0.20 < q[1] < 0.35):
                return 0.0
            poly = self.bin_outline((q[0], q[1]), q[2])
            inter = polygon_area(clip_polygon(obs, poly))
            return inter / (polygon_area(poly) + obs_area - inter + 1e-12)

        best = (0.0, None)
        for yaw in np.radians(np.arange(-12.0, 12.1, 2.0)):
            sc = iou([seed[0], seed[1], yaw])
            if sc > best[0]:
                best = (sc, [seed[0], seed[1], yaw])
        if best[1] is None:
            return None
        res = optimize.minimize(lambda q: -iou(q), best[1], method="Nelder-Mead",
                                options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 400})
        q = res.x
        return Pose(xy=np.array([q[0], q[1]]), yaw=float(q[2]), score=float(-res.fun),
                    pixels=int(blob.sum()), colour="bin")
