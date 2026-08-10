#!/usr/bin/env python3
"""Test the modified cleaner with real dataset."""

import sys
sys.path.insert(0, '/data/chenxugao/RoboClean')

from pathlib import Path
from roboclean.core.cleaner import DataCleaner
import pyarrow.parquet as pq
import pandas as pd

def test_cleaner():
    """Test cleaner with pickbread dataset."""
    input_path = Path("/data/chenxugao/RoboClean/demo_data/pickbread/00")
    output_path = Path("/data/chenxugao/RoboClean/demo_data/pickbread/00_cleaned_v2")

    print("="*80)
    print("Testing Modified Cleaner Logic")
    print("="*80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Load dataset
    from roboclean.core.loader import LeRobotDatasetLoader
    loader = LeRobotDatasetLoader(str(input_path))

    # Clean with default threshold (should drop some static frames)
    cleaner = DataCleaner(loader)
    cleaned_path = cleaner.clean_dataset(
        output_root=str(output_path),
        threshold=0.01,  # Small threshold to drop near-static frames
        norm="l2"
    )

    print(f"\nCleaning completed: {cleaned_path}")
    print()

    # Verify the results
    print("="*80)
    print("Verifying Cleaned Data")
    print("="*80)

    # Load original data
    original_parquet = input_path / "data" / "chunk-000" / "file-000.parquet"
    original_table = pq.read_table(original_parquet)
    original_df = original_table.to_pandas()

    print(f"\nOriginal data:")
    print(f"  Total frames: {len(original_df)}")
    print(f"  Columns: {list(original_df.columns)}")
    print(f"\n  Sample timestamps (first 10):")
    print(f"  {original_df['timestamp'].head(10).tolist()}")
    print(f"\n  Sample frame_index (first 10):")
    print(f"  {original_df['frame_index'].head(10).tolist()}")

    # Load cleaned data
    cleaned_parquet = Path(cleaned_path) / "data" / "chunk-000" / "file-000.parquet"
    if cleaned_parquet.exists():
        cleaned_table = pq.read_table(cleaned_parquet)
        cleaned_df = cleaned_table.to_pandas()

        print(f"\nCleaned data:")
        print(f"  Total frames: {len(cleaned_df)}")
        print(f"  Columns: {list(cleaned_df.columns)}")
        print(f"\n  Sample timestamps (first 10):")
        print(f"  {cleaned_df['timestamp'].head(10).tolist()}")
        print(f"\n  Sample frame_index (first 10):")
        print(f"  {cleaned_df['frame_index'].head(10).tolist()}")

        # Check if timestamps are preserved (not recalculated)
        print("\n" + "="*80)
        print("Validation: Checking if timestamps are preserved")
        print("="*80)

        # Find which original frames were kept
        if len(cleaned_df) > 0:
            # Check first episode
            first_ep_cleaned = cleaned_df[cleaned_df['episode_index'] == 0]
            first_ep_original = original_df[original_df['episode_index'] == 0]

            print(f"\nEpisode 0:")
            print(f"  Original frames: {len(first_ep_original)}")
            print(f"  Cleaned frames:  {len(first_ep_cleaned)}")
            print(f"  Dropped frames:  {len(first_ep_original) - len(first_ep_cleaned)}")

            # Check if timestamps are original values
            if len(first_ep_cleaned) > 5:
                print(f"\n  Timestamp comparison (first 5 cleaned frames):")
                for i in range(min(5, len(first_ep_cleaned))):
                    cleaned_ts = first_ep_cleaned.iloc[i]['timestamp']
                    cleaned_fi = first_ep_cleaned.iloc[i]['frame_index']

                    # Check if these values exist in original data
                    matching_original = first_ep_original[
                        (first_ep_original['timestamp'] == cleaned_ts) |
                        (abs(first_ep_original['timestamp'] - cleaned_ts) < 0.0001)
                    ]

                    if len(matching_original) > 0:
                        print(f"    Frame {i}: ts={cleaned_ts:.6f}, fi={cleaned_fi} ✓ (found in original)")
                    else:
                        print(f"    Frame {i}: ts={cleaned_ts:.6f}, fi={cleaned_fi} ✗ (NOT in original!)")

            # Check if frame_index values are preserved (not 0,1,2,...)
            frame_indices = first_ep_cleaned['frame_index'].values
            expected_sequential = list(range(len(frame_indices)))

            if list(frame_indices) == expected_sequential:
                print(f"\n  ⚠️  WARNING: frame_index was recalculated to be sequential!")
                print(f"     This means the preservation logic is NOT working.")
            else:
                print(f"\n  ✓ frame_index values are preserved (not recalculated)")
                print(f"    Example: {frame_indices[:10]}")

            # Same for timestamps
            timestamps = first_ep_cleaned['timestamp'].values
            if len(timestamps) > 1:
                # Check if timestamps are evenly spaced (would indicate recalculation)
                time_diffs = [timestamps[i+1] - timestamps[i] for i in range(min(5, len(timestamps)-1))]
                avg_diff = sum(time_diffs) / len(time_diffs)
                expected_diff = 1.0 / 30.0  # Assuming 30 fps

                if all(abs(d - expected_diff) < 0.001 for d in time_diffs):
                    print(f"\n  ⚠️  WARNING: timestamps are evenly spaced!")
                    print(f"     This suggests they were recalculated.")
                else:
                    print(f"\n  ✓ timestamps are preserved (not evenly spaced)")
                    print(f"    Time diffs: {[f'{d:.6f}' for d in time_diffs[:5]]}")

if __name__ == "__main__":
    test_cleaner()