#!/usr/bin/env python3
from pyorbbecsdk import *
import numpy as np
import cv2
from collections import deque


def main():
    # Create an Orbbec configuration + processing pipeline
    config = Config()
    pipeline = Pipeline()

    # ================================================================
    #  COLOR STREAM CONFIGURATION
    #  We choose YUYV (uncompressed) @ 1280x720 @ 30 FPS
    # ================================================================
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_video_stream_profile(
        1280, 720, OBFormat.YUYV, 30
    )

    # ================================================================
    #  DEPTH STREAM CONFIGURATION
    #  We choose RAW_DEPTH16 (Y16) @ width=640 @ 30 FPS
    #  Height=0 → SDK automatically picks matching height for this mode
    # ================================================================
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_video_stream_profile(
        640, 0, OBFormat.Y16, 30
    )

    # Enable both color + depth streams in the pipeline
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    # Start the pipeline and begin streaming frames
    pipeline.start(config)
    print("Pipeline started - press 'q' or ESC to quit")

    # A rolling buffer storing last 15 valid center distance readings (meters)
    history = deque(maxlen=15)

    try:
        while True:
            # Wait for a synchronized frame set (depth + color)
            frames = pipeline.wait_for_frames(100)
            if not frames:
                continue

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Safety check — sometimes frames drop
            if depth_frame is None or color_frame is None:
                continue

            # ================================================================
            #  PROCESS COLOR FRAME (Convert YUYV → BGR for display)
            # ================================================================
            c_w = color_frame.get_width()
            c_h = color_frame.get_height()
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

            try:
                # YUYV = 2 bytes per pixel → reshape to (H, W, 2)
                color_yuyv = color_data.reshape((c_h, c_w, 2))
            except ValueError:
                continue  # Skip bad frames

            # Convert YUYV → BGR so OpenCV can display it
            color_bgr = cv2.cvtColor(color_yuyv, cv2.COLOR_YUV2BGR_YUY2)

            # ================================================================
            #  PROCESS DEPTH FRAME (Y16 → uint16 distance data)
            # ================================================================
            d_w = depth_frame.get_width()
            d_h = depth_frame.get_height()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)

            try:
                depth_data = depth_data.reshape((d_h, d_w))
            except ValueError:
                continue  # Skip bad frames

            # ================================================================
            #  GET CENTER DEPTH VALUE (median of 5×5 region)
            # ================================================================
            cx = d_w // 2
            cy = d_h // 2
            win = 2  # 2 px on each side → 5×5 window

            y1 = max(cy - win, 0)
            y2 = min(cy + win + 1, d_h)
            x1 = max(cx - win, 0)
            x2 = min(cx + win + 1, d_w)

            # Extract 5×5 region around center
            center_region = depth_data[y1:y2, x1:x2]

            # Ignore invalid zeros
            valid = center_region[center_region > 0]

            current_m = None
            if valid.size > 0:
                # Depth values come in millimeters
                depth_mm = int(np.median(valid))
                current_m = depth_mm / 1000.0
                history.append(current_m)  # Add to rolling average buffer

            # ================================================================
            #  SMOOTHED DISTANCE (Average of last N frames)
            # ================================================================
            avg_m = float(np.mean(history)) if len(history) > 0 else None

            # Build display text
            curr_text = f"Curr: {current_m:.1f} m" if current_m else "Curr: N/A"
            avg_text  = f"Avg (last {len(history)}): {avg_m:.1f} m" if avg_m else "Avg: N/A"

            # Print distance to terminal (debugging)
            if current_m is not None:
                print(curr_text, "|", avg_text)

            # ================================================================
            #  DEPTH VISUALIZATION (Normalize + color map)
            # ================================================================
            depth_norm = cv2.normalize(
                depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            # Draw marker + text on depth image
            cv2.circle(depth_colormap, (cx, cy), 5, (255, 255, 255), 2)
            cv2.putText(depth_colormap, curr_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(depth_colormap, avg_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Draw marker + text on color image
            ccx = c_w // 2
            ccy = c_h // 2
            cv2.circle(color_bgr, (ccx, ccy), 8, (0, 255, 0), 2)
            cv2.putText(color_bgr, curr_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(color_bgr, avg_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ================================================================
            #  DISPLAY WINDOWS
            # ================================================================
            cv2.imshow("Gemini2 Color (YUYV)", color_bgr)
            cv2.imshow("Gemini2 Depth (Y16)", depth_colormap)

            # Exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        # Clean shut-down
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped")


if __name__ == "__main__":
    main()