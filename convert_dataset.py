#!/usr/bin/env python3
"""
Convert image sequences to MP4 videos for ControlVideo dataset.

This script processes image sequences in the assets directory and converts
them to MP4 format using ffmpeg.

Directory structure:
    assets/
    ├── o2o/
    │   ├── 3ball/ (1.jpg, 2.jpg, ..., 15.jpg)
    │   └── ...
    ├── o2p/
    └── p2p/

Output:
    assets/
    ├── o2o/
    │   ├── 3ball.mp4
    │   └── ...
"""

import os
import subprocess
from pathlib import Path


def convert_sequence_to_video(input_dir, output_path, framerate=2, crf=18):
    """
    Convert image sequence to MP4 video using ffmpeg.

    Args:
        input_dir: Directory containing numbered images (1.jpg, 2.jpg, ...)
        output_path: Output MP4 file path
        framerate: Output video framerate (default: 2)
        crf: Constant Rate Factor for quality (default: 18, range 0-51, lower is better)

    Returns:
        tuple: (success: bool, message: str)
    """
    # Check if images exist
    first_image = input_dir / "1.jpg"
    if not first_image.exists():
        return False, f"First image not found: {first_image}"

    # Count frames
    frame_count = len(list(input_dir.glob("*.jpg")))

    # Build ffmpeg command
    input_pattern = str(input_dir / "%d.jpg")
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-framerate", str(framerate),
        "-start_number", "1",
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        str(output_path)
    ]

    try:
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Verify output file
        if output_path.exists() and output_path.stat().st_size > 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            return True, f"{frame_count} frames, {size_mb:.2f} MB"
        else:
            return False, "Output file not created or empty"

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "Unknown error"
        return False, f"ffmpeg error: {error_msg[-200:]}"  # Last 200 chars
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def main():
    # Get script directory and assets path
    script_dir = Path(__file__).parent
    assets_dir = script_dir / "assets"

    if not assets_dir.exists():
        print(f"Error: Assets directory not found: {assets_dir}")
        return 1

    print("=" * 60)
    print("ControlVideo Dataset Converter")
    print("Converting image sequences to MP4 videos")
    print("=" * 60)
    print()

    # Categories to process
    categories = ["o2o", "o2p", "p2p"]

    # Statistics
    total = 0
    success = 0
    failed = 0

    # Process each category
    for category in categories:
        category_path = assets_dir / category

        if not category_path.exists() or not category_path.is_dir():
            print(f"Skipping {category}: directory not found")
            continue

        print(f"Processing category: {category}")
        print("-" * 60)

        # Get all subdirectories
        subdirs = sorted([d for d in category_path.iterdir() if d.is_dir()])

        for subdir in subdirs:
            dirname = subdir.name
            output_file = category_path / f"{dirname}.mp4"

            print(f"  {dirname}...", end=" ", flush=True)

            # Convert
            is_success, message = convert_sequence_to_video(subdir, output_file)

            total += 1
            if is_success:
                success += 1
                print(f"✓ {message}")
            else:
                failed += 1
                print(f"✗ {message}")

        print()

    # Print summary
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total:   {total}")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print()

    if failed == 0:
        print("All videos converted successfully!")
        return 0
    else:
        print(f"{failed} video(s) failed to convert. Check error messages above.")
        return 1


if __name__ == "__main__":
    exit(main())
