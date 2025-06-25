import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


def find_template_in_frame(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.6,  # left, top, width, height
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi: Optional[Tuple[int, int, int, int]] = (
        739,
        0,
        1481,
        501,
    )

    x_start, y_start, roi_w, roi_h = roi
    x_end = x_start + roi_w
    y_end = y_start + roi_h
    frame_gray_roi = frame_gray[y_start:y_end, x_start:x_end]

    result = cv2.matchTemplate(frame_gray_roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        h, w = template.shape
        bbox = (max_loc[0] + x_start, max_loc[1] + y_start, w, h)
        return bbox, max_val

    return None, max_val


def calculate_visible_coordinates(
    bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
) -> Dict[str, int]:
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h

    height, width = frame_shape

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return {
        "width": x2 - x,
        "height": y2 - y,
        "top": y,
        "right": x2,
        "bottom": y2,
        "left": x,
    }


def track_template_all_frames(
    video_path: str, template_path: str, threshold: float = 0.7, sample_rate: int = 1
) -> pd.DataFrame:
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(
        f"Video: {fps:.2f} FPS, {total_frames} frames, {total_frames/fps:.2f}s duration"
    )
    print(f"Frame size: {frame_width}x{frame_height}")

    template = cv2.imread(template_path, 0)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")

    print(f"Template size: {template.shape[1]}x{template.shape[0]}")

    results = []
    frame_idx = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        bbox, confidence = find_template_in_frame(frame, template, threshold)

        if bbox:
            print(bbox, confidence)
            relative_time = frame_idx / fps
            frame_data = {
                "frame_number": frame_idx,
                "timestamp_relative": relative_time,
                "confidence": confidence,
            }

            coords = calculate_visible_coordinates(bbox, (frame_height, frame_width))
            frame_data.update(coords)
            results.append(frame_data)

        frame_idx += 1

    video.release()
    return pd.DataFrame(results)


def process_video_template(
    video_filename: str,
    template_filename: str = "banner_template.png",
    threshold: float = 0.6,
    sample_rate: int = 1,
) -> pd.DataFrame:
    tracking_data = track_template_all_frames(
        video_filename, template_filename, threshold, sample_rate  # roi
    )

    print(f"\nProcessed {len(tracking_data)} frames with template detected")
    print(f"Average confidence: {tracking_data['confidence'].mean():.3f}")
    print("\nSample tracking data:")
    print(
        tracking_data[
            [
                "timestamp_relative",
                "confidence",
                "width",
                "height",
                "top",
                "right",
                "bottom",
                "left",
            ]
        ].head(10)
    )

    if len(tracking_data) > 0:
        print(
            f"\nTemplate visible from {tracking_data['timestamp_relative'].min():.2f}s to {tracking_data['timestamp_relative'].max():.2f}s"
        )

    return tracking_data
