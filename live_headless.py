#!/usr/bin/env python3
"""Headless live face-swap: reads camera frames from a network stream,
swaps the face, and streams the result back out. No GUI required.

Usage (on the GPU server):
    python live_headless.py --source /tmp/face1.jpg --output-host 192.168.10.107

Then on the PC, send the camera (same as before):
    ffmpeg -f v4l2 -input_format h264 -video_size 1280x720 -framerate 30 -i /dev/video0 \\
      -c:v copy -f mpegts "tcp://192.168.10.148:12345"

And view the result on the PC:
    ffplay -fflags nobuffer -flags low_delay "tcp://0.0.0.0:12346?listen=1"
"""

import argparse
import subprocess
import sys

import cv2

import modules.globals
from modules import imread_unicode
from modules.face_analyser import get_one_face
from modules.processors.frame.face_swapper import (
    get_face_swapper,
    swap_face,
    apply_post_processing,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="source face image path")
    parser.add_argument("--input-url", default="tcp://0.0.0.0:12345?listen",
                         help="incoming camera stream URL")
    parser.add_argument("--output-host", required=True,
                         help="your PC's LAN IP, to stream the swapped video back to")
    parser.add_argument("--output-port", type=int, default=12346)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--execution-provider", default="cuda",
                         choices=["cuda", "cpu"])
    args = parser.parse_args()

    modules.globals.execution_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.execution_provider == "cuda"
        else ["CPUExecutionProvider"]
    )

    print(f"Loading source face from {args.source}...")
    source_img = imread_unicode(args.source)
    if source_img is None:
        print(f"ERROR: could not read {args.source}")
        sys.exit(1)
    source_face = get_one_face(source_img)
    if source_face is None:
        print("ERROR: no face found in source image")
        sys.exit(1)
    print("Source face loaded.")

    print("Loading face swapper model...")
    if get_face_swapper() is None:
        print("ERROR: face swapper model failed to load")
        sys.exit(1)

    print(f"Opening input stream: {args.input_url}")
    print("Waiting for your PC's ffmpeg sender to connect...")
    cap = cv2.VideoCapture(args.input_url)
    if not cap.isOpened():
        print("ERROR: failed to open input stream")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: input stream opened but no frame received")
        sys.exit(1)
    height, width = frame.shape[:2]
    print(f"Connected! Incoming frame size: {width}x{height}")

    out_url = f"tcp://{args.output_host}:{args.output_port}"
    writer_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(args.fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-f", "mpegts", out_url,
    ]
    print(f"Starting output stream to {out_url}")
    print("(Start your ffplay viewer on the PC now if it's not already running)")
    writer = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE)

    frame_count = 0
    try:
        while True:
            target_face = get_one_face(frame)
            if target_face is not None:
                frame = swap_face(source_face, target_face, frame)
                frame = apply_post_processing(frame, [target_face.bbox.astype(int)])
            try:
                writer.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("Output pipe broken (viewer disconnected?). Stopping.")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

            ret, frame = cap.read()
            if not ret:
                print("Input stream ended.")
                break
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        try:
            writer.stdin.close()
        except Exception:
            pass
        writer.wait()


if __name__ == "__main__":
    main()
