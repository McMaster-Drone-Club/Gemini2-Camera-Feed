#!/usr/bin/env python3
"""
point_distance_3d.py

Click two points on the COLOR window to measure:
- Distance of each point from the camera (m)
- Full 3D distance between the two points (m), using camera FOV.
"""

from pyorbbecsdk import *
import numpy as np
import cv2
from collections import deque
import math

# ---------------- FOVs from datasheet ----------------

# Depth FOV
HFOV_DEPTH_DEG = 91.0
VFOV_DEPTH_DEG = 66.0

# RGB/Color FOV
HFOV_COLOR_DEG = 86.0
VFOV_COLOR_DEG = 55.0

# Precompute radians
HFOV_DEPTH_RAD = math.radians(HFOV_DEPTH_DEG)
VFOV_DEPTH_RAD = math.radians(VFOV_DEPTH_DEG)
HFOV_COLOR_RAD = math.radians(HFOV_COLOR_DEG)
VFOV_COLOR_RAD = math.radians(VFOV_COLOR_DEG)

# Store clicked points in COLOR coordinates (max 2)
click_state = {"pts_color": []}


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback for the COLOR window.
    Left-click to select up to 2 points.
    On the 3rd click, it resets and starts over.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(click_state["pts_color"]) >= 2:
            click_state["pts_color"].clear()
        click_state["pts_color"].append((x, y))
        print(f"Selected point {len(click_state['pts_color'])}: ({x}, {y})")


def get_depth_at_point(depth_data, x, y, window=2):
    """
    Returns the median depth (in meters) around (x, y) in a (2*window+1)^2 region.
    depth_data is a 2D uint16 array in millimeters.
    """
    h, w = depth_data.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    y1 = max(y - window, 0)
    y2 = min(y + window + 1, h)
    x1 = max(x - window, 0)
    x2 = min(x + window + 1, w)

    region = depth_data[y1:y2, x1:x2]
    valid = region[region > 0]

    if valid.size == 0:
        return None

    depth_mm = int(np.median(valid))
    depth_m = depth_mm / 1000.0  # meters

    if depth_m <= 0 or not math.isfinite(depth_m):
        return None

    return depth_m


def map_color_to_depth(cx, cy, c_w, c_h, d_w, d_h):
    """
    Map a pixel from COLOR image (cx, cy) to a pixel on the DEPTH image (dx, dy)
    using angular mapping (color FOV -> depth FOV).

    Returns (dx, dy) as integers, always clamped within depth bounds.
    """

    # Centered normalized coords in color image in [-1, 1]
    cx_c = c_w / 2.0
    cy_c = c_h / 2.0
    nx_c = (cx - cx_c) / (c_w / 2.0)
    ny_c = (cy - cy_c) / (c_h / 2.0)

    # Angles in color camera
    theta_x = nx_c * (HFOV_COLOR_RAD / 2.0)
    theta_y = ny_c * (VFOV_COLOR_RAD / 2.0)

    # Corresponding normalized coords in depth camera
    # angle = nx_d * (HFOV_DEPTH/2) => nx_d = angle / (HFOV_DEPTH/2)
    if HFOV_DEPTH_RAD == 0 or VFOV_DEPTH_RAD == 0:
        return None

    nx_d = theta_x / (HFOV_DEPTH_RAD / 2.0)
    ny_d = theta_y / (VFOV_DEPTH_RAD / 2.0)

    # Ideally |nx_d|,|ny_d| <= 1. If not, clamp to edge of depth FoV.
    nx_d = max(-1.0, min(1.0, nx_d))
    ny_d = max(-1.0, min(1.0, ny_d))

    # Back to pixel coords in depth image
    dx = nx_d * (d_w / 2.0) + d_w / 2.0
    dy = ny_d * (d_h / 2.0) + d_h / 2.0

    # Clamp to valid integer pixel range
    dx_i = int(np.clip(dx, 0, d_w - 1))
    dy_i = int(np.clip(dy, 0, d_h - 1))

    return dx_i, dy_i


def pixel_to_camera_xyz(dx, dy, depth_m, d_w, d_h):
    """
    Map depth pixel (dx, dy, depth_m) to camera coordinates (X, Y, Z),
    using the DEPTH FOV.
    """
    if depth_m is None or depth_m <= 0 or not math.isfinite(depth_m):
        return None

    cx_d = d_w / 2.0
    cy_d = d_h / 2.0

    nx = (dx - cx_d) / (d_w / 2.0)
    ny = (dy - cy_d) / (d_h / 2.0)

    theta_x = nx * (HFOV_DEPTH_RAD / 2.0)
    theta_y = ny * (VFOV_DEPTH_RAD / 2.0)

    X = depth_m * math.tan(theta_x)
    Y = depth_m * math.tan(theta_y)
    Z = depth_m

    return np.array([X, Y, Z], dtype=np.float32)


def safe_mean(history):
    """Return float mean or None if history empty or all invalid."""
    if not history:
        return None
    arr = np.array(history, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def main():
    # ------------------------------------------------------------
    #  Orbbec pipeline + config
    # ------------------------------------------------------------
    config = Config()
    pipeline = Pipeline()

    # Get stream profiles
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    # COLOR: 1280x720 YUYV @ 30
    color_profile = color_profiles.get_video_stream_profile(
        1280, 720, OBFormat.YUYV, 30
    )

    # DEPTH: 640x400 Y16 @ 30
    depth_profile = depth_profiles.get_video_stream_profile(
        640, 400, OBFormat.Y16, 30
    )

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    pipeline.start(config)
    print("Pipeline started – press 'q' or ESC to quit")
    print("Left-click on the COLOR window to pick two points.")

    # Create windows once + bind callback to COLOR window
    cv2.namedWindow("Gemini2 Color (YUYV)")
    cv2.namedWindow("Gemini2 Depth (Y16)")
    cv2.setMouseCallback("Gemini2 Color (YUYV)", mouse_callback)

    history_p1 = deque(maxlen=10)
    history_p2 = deque(maxlen=10)

    try:
        timeout_count = 0
        MAX_TIMEOUTS = 50  # e.g. ~5s if timeout=100ms

        while True:
            try:
                frames = pipeline.wait_for_frames(100)  # 100 ms timeout
            except Exception as e:
                print(f"\n[ERROR] wait_for_frames failed: {e}")
                break

            if not frames:
                timeout_count += 1
                if timeout_count >= MAX_TIMEOUTS:
                    print("\n[ERROR] No frames for a while – device likely disconnected. Exiting.")
                    break
                continue

            timeout_count = 0

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if depth_frame is None or color_frame is None:
                continue

            # ---------------- COLOR: YUYV → BGR ----------------
            c_w = color_frame.get_width()
            c_h = color_frame.get_height()
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

            try:
                color_yuyv = color_data.reshape((c_h, c_w, 2))
            except ValueError:
                continue

            color_bgr = cv2.cvtColor(color_yuyv, cv2.COLOR_YUV2BGR_YUY2)

            # ---------------- DEPTH: Y16 → uint16 ---------------
            d_w = depth_frame.get_width()
            d_h = depth_frame.get_height()
            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)

            try:
                depth_data = depth_raw.reshape((d_h, d_w))
            except ValueError:
                continue

            # ---------------- Compute depths --------------------
            d1 = d2 = None
            p1_3d = p2_3d = None
            depth_pts = []  # list of (dx, dy) for drawing on depth view

            # First point
            if len(click_state["pts_color"]) >= 1:
                cx1, cy1 = click_state["pts_color"][0]
                mapped = map_color_to_depth(cx1, cy1, c_w, c_h, d_w, d_h)
                if mapped is not None:
                    dx1, dy1 = mapped
                    depth_pts.append((dx1, dy1))
                    d1 = get_depth_at_point(depth_data, dx1, dy1)
                    if d1 is not None:
                        history_p1.append(d1)

            # Second point
            if len(click_state["pts_color"]) >= 2:
                cx2, cy2 = click_state["pts_color"][1]
                mapped = map_color_to_depth(cx2, cy2, c_w, c_h, d_w, d_h)
                if mapped is not None:
                    dx2, dy2 = mapped
                    depth_pts.append((dx2, dy2))
                    d2 = get_depth_at_point(depth_data, dx2, dy2)
                    if d2 is not None:
                        history_p2.append(d2)

            # Smoothed depths
            d1_s = safe_mean(history_p1)
            d2_s = safe_mean(history_p2)

            # ---------------- 3D coordinates & distance ---------
            dist_3d = None
            if len(depth_pts) >= 1 and d1_s is not None:
                dx1, dy1 = depth_pts[0]
                p1_3d = pixel_to_camera_xyz(dx1, dy1, d1_s, d_w, d_h)

            if len(depth_pts) >= 2 and d2_s is not None:
                dx2, dy2 = depth_pts[1]
                p2_3d = pixel_to_camera_xyz(dx2, dy2, d2_s, d_w, d_h)

            if p1_3d is not None and p2_3d is not None:
                try:
                    dist_3d = float(np.linalg.norm(p1_3d - p2_3d))
                except Exception as e:
                    print(f"\n[WARN] Failed to compute 3D dist: {e}")
                    dist_3d = None

            # ---------------- Text strings ----------------------
            txt1 = f"P1: {d1_s:.2f} m" if d1_s is not None else "P1: N/A"
            txt2 = f"P2: {d2_s:.2f} m" if d2_s is not None else "P2: N/A"
            txt_3d = (
                f"3D dist: {dist_3d:.2f} m" if dist_3d is not None else "3D dist: N/A"
            )

            print(txt1, "|", txt2, "|", txt_3d, end="\r")

            # ---------------- Depth visualization ---------------
            depth_norm = cv2.normalize(
                depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            # Draw mapped depth points
            for idx, (dx, dy) in enumerate(depth_pts):
                color = (255, 255, 255) if idx == 0 else (0, 0, 0)
                cv2.circle(depth_colormap, (dx, dy), 6, color, 2)

            cv2.putText(depth_colormap, txt1, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_colormap, txt2, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_colormap, txt_3d, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ---------------- Draw markers on COLOR -------------
            for idx, (cx, cy) in enumerate(click_state["pts_color"]):
                col = (0, 255, 0) if idx == 0 else (0, 200, 255)
                cv2.circle(color_bgr, (cx, cy), 8, col, 2)

            cv2.putText(color_bgr, txt_3d, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ---------------- Show windows ----------------------
            cv2.imshow("Gemini2 Color (YUYV)", color_bgr)
            cv2.imshow("Gemini2 Depth (Y16)", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nPipeline stopped")


if __name__ == "__main__":
    main()