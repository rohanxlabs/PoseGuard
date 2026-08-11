"""
PoseGuard Annotation Tool
Extract frames from a video and display detected keypoints for manual review.
Useful for creating ground-truth labels for training an ML classifier.

Usage:
    python scripts/annotate_tool.py --input data/raw/video.mp4 --output data/annotations/frames/

Controls:
    s       - save current frame + keypoints JSON
    n       - next frame (skip forward)
    q       - quit
"""
import argparse
import cv2
import json
import os
import yaml
from pathlib import Path

from src.detector import PoseDetector, KEYPOINTS, SKELETON_PAIRS


def main():
    parser = argparse.ArgumentParser(description="PoseGuard annotation helper")
    parser.add_argument("--input",  type=str, required=True,
                        help="Input video path")
    parser.add_argument("--output", type=str, default="data/annotations/frames/",
                        help="Output directory for saved frames")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--skip",   type=int, default=10,
                        help="Skip N frames between displays")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    detector = PoseDetector(args.config)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Cannot open video: {args.input}")
        return

    frame_idx = 0
    saved_count = 0

    print("\n=== PoseGuard Annotation Tool ===")
    print("Controls: [s] save · [n] next · [q] quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.skip != 0:
            continue

        # Detect poses
        persons = detector.detect(frame)

        # Draw skeletons on display frame
        display = frame.copy()
        for person in persons:
            kps = person["keypoints"]
            kp_list = [kps.get(name) for name in KEYPOINTS.keys()]

            # Draw skeleton
            for (i, j) in SKELETON_PAIRS:
                if kp_list[i] and kp_list[j]:
                    pt1 = (int(kp_list[i][0]), int(kp_list[i][1]))
                    pt2 = (int(kp_list[j][0]), int(kp_list[j][1]))
                    cv2.line(display, pt1, pt2, (0, 255, 120), 2, cv2.LINE_AA)

            # Draw keypoints
            for kp in kp_list:
                if kp:
                    cv2.circle(display, (int(kp[0]), int(kp[1])), 5, (255, 255, 255), -1)
                    cv2.circle(display, (int(kp[0]), int(kp[1])), 5, (0, 255, 120), 1)

        # HUD
        cv2.putText(display, f"Frame: {frame_idx} | Persons: {len(persons)} | Saved: {saved_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, "[s] save  [n] next  [q] quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Annotation Tool", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n'):
            continue
        elif key == ord('s'):
            # Save frame + JSON metadata
            frame_name = f"frame_{frame_idx:06d}.jpg"
            json_name  = f"frame_{frame_idx:06d}.json"

            cv2.imwrite(os.path.join(args.output, frame_name), frame)

            # Export keypoints in a simple format
            export_data = {
                "frame_index": frame_idx,
                "image_file":  frame_name,
                "persons": []
            }
            for person in persons:
                kps_export = {}
                for name, idx in KEYPOINTS.items():
                    kp = person["keypoints"].get(name)
                    kps_export[name] = {
                        "x": kp[0], "y": kp[1], "confidence": kp[2]
                    } if kp else None
                export_data["persons"].append({
                    "bbox": person["bbox"],
                    "keypoints": kps_export,
                })

            with open(os.path.join(args.output, json_name), "w") as f:
                json.dump(export_data, f, indent=2)

            saved_count += 1
            print(f"✓ Saved {frame_name} + {json_name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Saved {saved_count} annotated frames to {args.output}")


if __name__ == "__main__":
    main()
