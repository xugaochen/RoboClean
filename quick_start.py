#!/usr/bin/env python3
"""Quick start example for RoboClean."""

from pathlib import Path

from roboclean.core.loader import LeRobotDatasetLoader
from roboclean.core.cleaner import DataCleaner
from roboclean.viz.visualizer import DatasetVisualizer


def main():
    # Example dataset path
    dataset_path = Path("demo_data/empty_table_reset/empty_table_reset_lerobot_v2.0")

    print("=" * 60)
    print("RoboClean - LeRobot Data Cleaning Example")
    print("=" * 60)

    # 1. Load dataset
    print("\n1. Loading dataset...")
    loader = LeRobotDatasetLoader(dataset_path)
    print(f"   Dataset: {dataset_path}")
    print(f"   Version: {loader.version}")
    print(f"   Episodes: {loader.info.get('total_episodes', 0)}")
    print(f"   Total frames: {loader.info.get('total_frames', 0)}")

    # 2. Show motion columns
    print("\n2. Motion columns detected:")
    motion_columns = loader.get_motion_columns()
    for col in motion_columns:
        print(f"   - {col}")

    # 3. Load sample episode
    print("\n3. Loading episode 0...")
    if loader.episodes:
        episode_data = loader.load_episode_as_numpy(0, motion_columns)
        for col, data in episode_data.items():
            print(f"   {col}: shape={data.shape}, dtype={data.dtype}")

    # 4. Visualize
    print("\n4. Generating visualization...")
    viz = DatasetVisualizer(loader)
    report_dir = Path("quick_start_report")
    report_dir.mkdir(exist_ok=True)

    # Generate episode length distribution
    fig1 = viz.plot_episode_length_distribution()
    fig1.savefig(report_dir / "episode_lengths.png", dpi=150, bbox_inches="tight")
    print(f"   Saved: {report_dir / 'episode_lengths.png'}")

    # Generate feature statistics
    fig2 = viz.plot_feature_statistics(max_episodes=5)
    fig2.savefig(report_dir / "feature_stats.png", dpi=150, bbox_inches="tight")
    print(f"   Saved: {report_dir / 'feature_stats.png'}")

    # 5. Clean dataset
    print("\n5. Cleaning dataset...")
    cleaner = DataCleaner(loader)
    output_path = Path("demo_data_cleaned_example")

    if output_path.exists():
        print(f"   Skipping cleaning (output exists: {output_path})")
        print("   Remove the output directory to run cleaning")
    else:
        cleaned_path = cleaner.clean_dataset(
            output_root=output_path,
            threshold=0.001,
            norm="max",
            compare_to="previous",
            overwrite=False,
        )
        print(f"   Cleaned dataset saved to: {cleaned_path}")

    print("\n" + "=" * 60)
    print("Quick start completed!")
    print(f"Report saved to: {report_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()