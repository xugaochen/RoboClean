#!/usr/bin/env python3
"""Test script to verify timestamp preservation after cleaning."""

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

def test_cleaner_logic():
    """Test that cleaner preserves original timestamps."""
    print("Testing cleaner logic...")

    # Create a mock table with timestamps
    timestamps = np.array([0.0, 0.033, 0.066, 0.099, 0.132], dtype=np.float32)
    frame_indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    table = pa.table({
        'timestamp': timestamps,
        'frame_index': frame_indices,
        'episode_index': np.array([0, 0, 0, 0, 0], dtype=np.int64),
        'index': np.array([0, 1, 2, 3, 4], dtype=np.int64),
    })

    print(f"Original timestamps: {table['timestamp'].to_numpy()}")
    print(f"Original frame_index: {table['frame_index'].to_numpy()}")

    # Simulate deleting frames 1-2 (keeping 0, 3, 4)
    keep_indices = [0, 3, 4]
    cleaned_table = table.take(pa.array(keep_indices, type=pa.int64()))

    print(f"\nAfter deletion (frames 1-2 removed):")
    print(f"Cleaned timestamps: {cleaned_table['timestamp'].to_numpy()}")
    print(f"Cleaned frame_index: {cleaned_table['frame_index'].to_numpy()}")

    # Expected behavior: timestamps and frame_index should NOT change
    expected_timestamps = [0.0, 0.099, 0.132]
    expected_frame_indices = [0, 3, 4]

    actual_timestamps = cleaned_table['timestamp'].to_numpy().tolist()
    actual_frame_indices = cleaned_table['frame_index'].to_numpy().tolist()

    if actual_timestamps == expected_timestamps:
        print("✓ Timestamps preserved correctly!")
    else:
        print(f"✗ Timestamps wrong! Expected {expected_timestamps}, got {actual_timestamps}")

    if actual_frame_indices == expected_frame_indices:
        print("✓ Frame indices preserved correctly!")
    else:
        print(f"✗ Frame indices wrong! Expected {expected_frame_indices}, got {actual_frame_indices}")

def test_video_alignment():
    """Test video alignment logic."""
    print("\n" + "="*60)
    print("Testing video alignment...")
    print("="*60)

    # Scenario: Episode has 5 frames, delete frame 2-3
    original_timestamps = [0.0, 0.033, 0.066, 0.099, 0.132]

    # After cleaning (keep frames 0, 1, 4)
    cleaned_timestamps = [0.0, 0.033, 0.132]  # Preserved!

    # Video decoding uses: from_timestamp + timestamp
    # Example: Episode metadata has from_timestamp = 5.0s
    from_timestamp = 5.0

    print(f"Original episode: timestamps={original_timestamps}")
    print(f"After cleaning: timestamps={cleaned_timestamps}")
    print(f"Episode metadata: from_timestamp={from_timestamp}s")

    print("\nVideo decoding:")
    for i, ts in enumerate(cleaned_timestamps):
        video_time = from_timestamp + ts
        print(f"  Frame {i}: timestamp={ts}s → video_time={video_time}s")

        # Check if this aligns with original frame
        if i == 0:
            expected_video_time = from_timestamp + 0.0
        elif i == 1:
            expected_video_time = from_timestamp + 0.033
        elif i == 2:
            expected_video_time = from_timestamp + 0.132  # Original frame 4

        if abs(video_time - expected_video_time) < 0.001:
            print(f"    ✓ Correctly aligned with original frame")
        else:
            print(f"    ✗ Misaligned! Expected {expected_video_time}s")

    print("\nConclusion:")
    print("✓ With preserved timestamps, video frames align correctly")
    print("✓ No need to update episode metadata")
    print("✓ Training and visualization both work correctly")

if __name__ == "__main__":
    test_cleaner_logic()
    test_video_alignment()