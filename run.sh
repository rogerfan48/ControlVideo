#!/bin/bash

# ControlVideo Inference Script for Assets Dataset
#
# About parameters:
# - video_length: Number of frames to process (15 for most videos, 12 for n_u_c_s)
# - frame_rate: Frame sampling rate from input video (optional, auto-calculated if not set)
#   * If not specified, it's computed as: total_frames / video_length
#   * Example: 15-frame video with video_length=15 will use all frames
# - condition: Type of control signal (depth, canny, openpose)
# - width/height: Output resolution (must be multiple of 32)
#
# Note: The input MP4's playback FPS (set during conversion) doesn't affect processing.
# ControlVideo reads frames based on video_length and frame_rate parameters.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
source /home/roger/miniconda3/bin/activate controlvideo

# Common parameters
# Note: Use depth_midas instead of depth (controlnet_aux 0.0.10+ requirement)
CONDITION="depth_midas"  # Options: depth_midas, depth_zoe, depth_leres, canny, openpose
WIDTH=512
HEIGHT=512
VERSION="v10"
SEED=42

# Output base directory
OUTPUT_BASE="outputs"

echo "======================================"
echo "ControlVideo Inference - Batch Processing"
echo "======================================"
echo ""

# ============================================
# Category: o2o (Object to Object)
# ============================================
echo "Processing o2o videos..."

# o2o/3ball
python inference.py \
    --prompt "A blue bowl and a yellow bowl are moving on the table." \
    --condition "$CONDITION" \
    --video_path "assets/o2o/3ball.mp4" \
    --output_path "$OUTPUT_BASE/o2o/3ball" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# o2o/car_truck
python inference.py \
    --prompt "A firetruck and a taxi are on the road" \
    --condition "$CONDITION" \
    --video_path "assets/o2o/car_truck.mp4" \
    --output_path "$OUTPUT_BASE/o2o/car_truck" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# o2o/cloth-bag
python inference.py \
    --prompt "A man with a leather jacket and a woman with a red school bag walking across each other." \
    --condition "$CONDITION" \
    --video_path "assets/o2o/cloth-bag.mp4" \
    --output_path "$OUTPUT_BASE/o2o/cloth-bag" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# o2o/dog-cat
python inference.py \
    --prompt "A Golden Retriever and a giant monkey cross each other." \
    --condition "$CONDITION" \
    --video_path "assets/o2o/dog-cat.mp4" \
    --output_path "$OUTPUT_BASE/o2o/dog-cat" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# o2o/elephant
python inference.py \
    --prompt "A curly apricot fur toy poodle walks across a giant panda." \
    --condition "$CONDITION" \
    --video_path "assets/o2o/elephant.mp4" \
    --output_path "$OUTPUT_BASE/o2o/elephant" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# o2o/hair
python inference.py \
    --prompt "A girl with wavy brown curly hair walk pass another girl with orange hair." \
    --condition "$CONDITION" \
    --video_path "assets/o2o/hair.mp4" \
    --output_path "$OUTPUT_BASE/o2o/hair" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# ============================================
# Category: o2p (Object to Person)
# ============================================
echo "Processing o2p videos..."

# o2p/Charlie-pillar
python inference.py \
    --prompt "A spider man walks across a tree." \
    --condition "$CONDITION" \
    --video_path "assets/o2p/Charlie-pillar.mp4" \
    --output_path "$OUTPUT_BASE/o2p/Charlie-pillar" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# o2p/man-obj-horizontal
python inference.py \
    --prompt "A Super man and a TV pass each other." \
    --condition "$CONDITION" \
    --video_path "assets/o2p/man-obj-horizontal.mp4" \
    --output_path "$OUTPUT_BASE/o2p/man-obj-horizontal" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# o2p/woman_car
python inference.py \
    --prompt "A school bus and an iron man riding a scooter crossing on the road." \
    --condition "$CONDITION" \
    --video_path "assets/o2p/woman_car.mp4" \
    --output_path "$OUTPUT_BASE/o2p/woman_car" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# ============================================
# Category: p2p (Person to Person)
# ============================================
echo "Processing p2p videos..."

# p2p/02_walk_walk
python inference.py \
    --prompt "Spider man and Polar bear walking in opposite directions." \
    --condition "$CONDITION" \
    --video_path "assets/p2p/02_walk_walk.mp4" \
    --output_path "$OUTPUT_BASE/p2p/02_walk_walk" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# p2p/03_withwalk
python inference.py \
    --prompt "Spiderman walks from behind a super man." \
    --condition "$CONDITION" \
    --video_path "assets/p2p/03_withwalk.mp4" \
    --output_path "$OUTPUT_BASE/p2p/03_withwalk" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# p2p/chinese_palace
python inference.py \
    --prompt "An iron man and a police walk in the garden." \
    --condition "$CONDITION" \
    --video_path "assets/p2p/chinese_palace.mp4" \
    --output_path "$OUTPUT_BASE/p2p/chinese_palace" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# p2p/lalaland
python inference.py \
    --prompt "Astronaut and Iron man walking in opposite directions." \
    --condition "$CONDITION" \
    --video_path "assets/p2p/lalaland.mp4" \
    --output_path "$OUTPUT_BASE/p2p/lalaland" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# p2p/n_u_c_s (Note: Only 12 frames)
python inference.py \
    --prompt "A police's back and an iron man's back move inside a room." \
    --condition "$CONDITION" \
    --video_path "assets/p2p/n_u_c_s.mp4" \
    --output_path "$OUTPUT_BASE/p2p/n_u_c_s" \
    --video_length 12 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

# p2p/u_name
python inference.py \
    --prompt "A grandpa and a grandma walking opposite on the stairs." \
    --condition "$CONDITION" \
    --video_path "assets/p2p/u_name.mp4" \
    --output_path "$OUTPUT_BASE/p2p/u_name" \
    --video_length 15 \
    --width $WIDTH \
    --height $HEIGHT \
    --version $VERSION \
    --seed $SEED

echo ""
echo "======================================"
echo "All inference tasks completed!"
echo "======================================"
echo "Results saved to: $OUTPUT_BASE/"
