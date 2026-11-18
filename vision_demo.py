"""Render the scene, detect everything, and save an annotated picture.

This is the "show me it works" script: it draws the fitted box outlines on the
camera frame together with the estimated pose, so the numbers in the log can be
checked against the picture by eye.

    python vision_demo.py [--out DIR] [--no-arm]
"""
from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import vision

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(ROOT, "mjcf", "scene.xml")

CUBE_RGB = {"red": (255, 70, 70), "green": (70, 220, 90), "blue": (90, 150, 255)}
BIN_RGB = (240, 240, 240)


FONT = ImageFont.load_default(size=18)


def draw_poly(draw, pts, colour, width=3):
    pts = [tuple(p) for p in pts]
    draw.line(pts + [pts[0]], fill=colour, width=width)


def draw_label(draw, u, v, text, colour):
    """Text with a dark plate behind it, so it stays readable on any background."""
    box = draw.textbbox((u, v), text, font=FONT)
    pad = 5
    draw.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad],
                   fill=(12, 12, 16))
    draw.text((u, v), text, fill=colour, font=FONT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "vision_demo.png"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    for _ in range(500):                 # let the cubes settle on the table
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    per = vision.Perception(model, data, args.width, args.height)
    img = per.render()
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    print("frame: %dx%d, mean %.1f" % (args.width, args.height, img.mean()))

    poses = per.find_all_cubes(img)
    for colour in ("red", "green", "blue"):
        pose = poses.get(colour)
        if pose is None:
            print("  %-5s not visible" % colour)
            continue
        draw_poly(draw, pose.score and per.box_outline(pose.xy, pose.yaw), CUBE_RGB[colour])
        label = "%s (%.3f, %.3f) yaw %+.1f  IoU %.3f" % (
            colour, pose.xy[0], pose.xy[1], np.degrees(pose.yaw), pose.score)
        u, v = per.cam.project(np.array([[pose.xy[0], pose.xy[1], vision.CUBE_TOP_Z]]))[0]
        draw_label(draw, u - 60, v - 34, label, CUBE_RGB[colour])
        print("  %-5s xy (%.4f, %.4f) yaw %+6.1f deg  IoU %.3f  %d px"
              % (colour, pose.xy[0], pose.xy[1], np.degrees(pose.yaw),
                 pose.score, pose.pixels))

    b = per.find_bin(img)
    if b is not None:
        draw_poly(draw, per.bin_outline(b.xy, b.yaw), BIN_RGB)
        u, v = per.cam.project(np.array([[b.xy[0], b.xy[1], vision.BIN_TOP_Z]]))[0]
        draw_label(draw, u - 80, v - 26,
                   "bin (%.3f, %.3f) IoU %.3f" % (b.xy[0], b.xy[1], b.score),
                   BIN_RGB)
        print("  bin   xy (%.4f, %.4f) IoU %.3f  %d px"
              % (b.xy[0], b.xy[1], b.score, b.pixels))
    else:
        print("  bin   not visible")

    pil.save(args.out)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
