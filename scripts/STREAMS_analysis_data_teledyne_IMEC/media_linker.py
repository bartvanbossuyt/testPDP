"""
media_linker.py — Media Linker for IMEC configuration metadata
===============================================================

Reads JSON sidecar files produced by config_generator_IMEC.py and uses
OpenCV (cv2) to open the corresponding video at the exact frame where
the configuration starts.  Also supports static images.

Features
--------
* Opens a video and seeks to the configuration's start frame.
* Plays the frame range that corresponds to the configuration (loops).
* On-screen overlay showing frame counter, config name, pause state.
* Keyboard controls for interactive navigation.
* --list mode to show all available configs in a directory.

Keyboard controls (during playback)
------------------------------------
  SPACE     : pause / resume
  Q / Esc   : quit
  →  or D   : step forward one frame  (paused only)
  ←  or A   : step backward one frame (paused only)
  R         : reset to start frame

Usage examples
--------------
    # Play a single configuration
    python media_linker.py --json output/configs/tracks_df_750_cfg0000.json

    # List all configs in a directory
    python media_linker.py --json_dir output/configs --list

    # Play the 5th configuration from a directory
    python media_linker.py --json_dir output/configs --index 4

Requirements
------------
    pip install opencv-python
"""

import os
import sys
import json
import argparse
import glob

# ── Attempt to import OpenCV ──────────────────────────────────────────
# cv2 is the only non-stdlib dependency of this script.
try:
    import cv2
except ImportError:
    print(
        "Error: OpenCV (cv2) is required.\n"
        "Install with:  pip install opencv-python"
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
#  METADATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────────────

def load_metadata(json_path: str) -> dict:
    """
    Load and validate a configuration sidecar JSON file.

    Expected keys (set by config_generator_IMEC.py):
      - source_media_path : absolute path to the video / image file
      - media_type        : "video" or "image"
      - start_time        : frame index (or seconds) to seek to
      - time_unit         : "frames" or "seconds"

    Args:
        json_path : Path to the JSON sidecar file.

    Returns:
        Parsed metadata dictionary.

    Raises:
        FileNotFoundError : JSON file does not exist.
        ValueError        : Required fields are missing or None.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # Check that the four essential fields are present and non-None.
    required = ["source_media_path", "media_type", "start_time", "time_unit"]
    missing = [f for f in required if f not in meta or meta[f] is None]

    if missing:
        raise ValueError(
            f"JSON metadata missing required fields: {missing}\n"
            "This usually means no --video_dir was given during config "
            "generation.  Re-run config_generator_IMEC.py with "
            "--video_dir to link media files."
        )

    return meta


def display_metadata_summary(meta: dict):
    """
    Print a concise, human-readable summary of the config metadata.
    """
    print(f"\n{'-' * 50}")
    print(f"Configuration : {meta.get('config_name', 'unknown')}")
    print(f"{'-' * 50}")
    print(f"  Source CSV   : {meta.get('source_csv',            'N/A')}")
    print(f"  Media path   : {meta.get('source_media_path',     'N/A')}")
    print(f"  Media type   : {meta.get('media_type',            'N/A')}")
    print(f"  Start frame  : {meta.get('start_time',            'N/A')}")
    print(f"  End frame    : {meta.get('end_time',              'N/A')}")
    print(f"  Duration     : {meta.get('target_length',         'N/A')} frames")
    print(f"  FPS          : {meta.get('fps',                   'N/A')}")
    print(f"  Start (sec)  : {meta.get('start_time_seconds',    'N/A')}s")

    # List the tracks and their classes
    classes = meta.get("track_classes", {})
    if classes:
        parts = [f"Track {tid} ({cls})" for tid, cls in classes.items()]
        print(f"  Tracks       : {', '.join(parts)}")
    print()


# ─────────────────────────────────────────────────────────────────────
#  VIDEO PLAYBACK
# ─────────────────────────────────────────────────────────────────────

def open_video_at_frame(meta: dict):
    """
    Open a video and play the segment that corresponds to the config.

    The video is opened with cv2.VideoCapture.  Seeking is done via
    CAP_PROP_POS_FRAMES (frame-based) or CAP_PROP_POS_MSEC (seconds).

    Overlay text is drawn on each frame to show:
      - current / end frame counter
      - configuration name
      - PAUSED indicator when paused

    Args:
        meta : Configuration metadata dictionary (from load_metadata).
    """
    media_path = meta["source_media_path"]

    # ── 1. Verify the media file exists ────────────────────────────
    if not os.path.exists(media_path):
        print(f"Error: Media file not found: {media_path}")
        print("The video may have been moved or deleted since "
              "the configuration was generated.")
        return

    # ── 2. Open video capture ──────────────────────────────────────
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {media_path}")
        return

    # Read basic video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video : {width}x{height}, {total_frames} frames, "
          f"{video_fps:.1f} fps")

    # ── 3. Compute seek window ─────────────────────────────────────
    start_frame   = meta["start_time"]
    target_length = meta.get("target_length", 30)
    time_unit     = meta["time_unit"]

    # If the time unit is seconds, convert to a frame number
    if time_unit == "seconds":
        start_frame = int(start_frame * video_fps)

    # Clamp to valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame   = min(start_frame + target_length, total_frames)

    print(f"Seeking to frame {start_frame} of {total_frames}")
    print(f"Playing frames {start_frame} -> {end_frame} "
          f"({target_length} frames)")
    print()
    print("Controls: SPACE=pause/play  Q=quit  "
          "LEFT/RIGHT=step  R=reset")

    # ── 4. Seek to start ──────────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ── 5. Playback loop ──────────────────────────────────────────
    paused        = False
    current_frame = start_frame
    window_name   = f"Config: {meta.get('config_name', 'video')}"
    frame         = None   # will be set on first successful read

    # Delay between frames for real-time playback speed
    delay_ms = int(1000 / video_fps) if video_fps > 0 else 33

    while True:
        # ── Read next frame (unless paused) ────────────────────────
        if not paused:
            ret, frame = cap.read()
            if not ret or current_frame >= end_frame:
                # End of segment → loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_frame = start_frame
                continue
            current_frame += 1

        # Skip drawing if we haven't read a frame yet
        if frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1

        # ── Draw overlay ───────────────────────────────────────────
        display = frame.copy()

        # Frame counter (top-left)
        cv2.putText(
            display,
            f"Frame {current_frame}/{end_frame}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        # Config name (below counter)
        cv2.putText(
            display,
            meta.get("config_name", ""),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
        )

        # Paused indicator (top-right)
        if paused:
            cv2.putText(
                display,
                "PAUSED",
                (width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        # ── Show the frame ─────────────────────────────────────────
        cv2.imshow(window_name, display)

        # ── Handle keyboard input ──────────────────────────────────
        key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF

        if key == ord("q") or key == 27:                    # Q / Esc
            break

        elif key == ord(" "):                               # Space
            paused = not paused

        elif key == ord("r"):                               # R → reset
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            ret, frame = cap.read()
            if ret:
                current_frame += 1

        elif key in (83, ord("d")):                         # → or D
            if paused:
                ret, frame = cap.read()
                if ret:
                    current_frame += 1

        elif key in (81, ord("a")):                         # ← or A
            if paused and current_frame > start_frame + 1:
                current_frame -= 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if ret:
                    current_frame += 1

    # ── Cleanup ────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────
#  IMAGE DISPLAY
# ─────────────────────────────────────────────────────────────────────

def display_image(meta: dict):
    """
    Display an image file linked to a configuration.

    Opens the image with cv2.imread and shows it with an overlay
    of the configuration name.  Press Q to close.

    Args:
        meta : Configuration metadata dictionary.
    """
    media_path = meta["source_media_path"]

    if not os.path.exists(media_path):
        print(f"Error: Image file not found: {media_path}")
        return

    img = cv2.imread(media_path)
    if img is None:
        print(f"Error: Could not read image: {media_path}")
        return

    # Overlay config name
    cv2.putText(
        img,
        meta.get("config_name", ""),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    window_name = f"Config: {meta.get('config_name', 'image')}"
    print("Press Q to close the image window.")

    cv2.imshow(window_name, img)

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Media Linker — open video/image at the frame "
                    "specified in a configuration JSON sidecar.",
    )

    # Accept either a single JSON file or a directory of JSON files
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--json",
        help="Path to a single configuration JSON file")
    group.add_argument(
        "--json_dir",
        help="Directory containing JSON files; use --index to pick one")

    parser.add_argument(
        "--index", type=int, default=0,
        help="When using --json_dir, select the Nth JSON file (0-based, "
             "default: 0)")
    parser.add_argument(
        "--list", action="store_true",
        help="When using --json_dir, list all available configs and exit")

    args = parser.parse_args()

    # ── Resolve JSON path ──────────────────────────────────────────
    if args.json:
        json_path = args.json
    else:
        # Discover JSON files in the directory
        json_files = sorted(
            glob.glob(os.path.join(args.json_dir, "*.json"))
        )

        if not json_files:
            print(f"No JSON files found in: {args.json_dir}")
            sys.exit(1)

        # ── List mode ──────────────────────────────────────────────
        if args.list:
            print(f"\nAvailable configurations in {args.json_dir}:")
            print(f"{'-' * 60}")
            for i, jf in enumerate(json_files):
                try:
                    with open(jf, "r", encoding="utf-8") as fh:
                        m = json.load(fh)
                    name = m.get("config_name", os.path.basename(jf))
                    cls_items = m.get("track_classes", {})
                    cls_str = ", ".join(cls_items.values()) if cls_items else "?"
                    print(f"  [{i:3d}] {name}  ({cls_str})")
                except Exception:
                    print(f"  [{i:3d}] {os.path.basename(jf)}  "
                          "(could not parse)")
            print()
            sys.exit(0)

        # ── Index mode ─────────────────────────────────────────────
        if args.index < 0 or args.index >= len(json_files):
            print(
                f"Error: --index {args.index} out of range. "
                f"Found {len(json_files)} JSON files "
                f"(0..{len(json_files) - 1})"
            )
            sys.exit(1)

        json_path = json_files[args.index]

    # ── Load metadata ──────────────────────────────────────────────
    print(f"Loading: {json_path}")
    meta = load_metadata(json_path)
    display_metadata_summary(meta)

    # ── Dispatch to the correct handler ────────────────────────────
    media_type = meta["media_type"].lower()

    if media_type == "video":
        open_video_at_frame(meta)
    elif media_type == "image":
        display_image(meta)
    else:
        print(f"Error: Unsupported media type: '{media_type}'")
        print("Supported types: video, image")
        sys.exit(1)


if __name__ == "__main__":
    main()
