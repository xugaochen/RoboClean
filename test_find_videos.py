#!/usr/bin/env python3
"""Test script to verify video finding logic."""

from pathlib import Path
from roboclean.core.loader import LeRobotDatasetLoader
from roboclean.web.app import find_episode_videos

def test_find_videos():
    """Test finding videos for both v2.1 and v3.0 datasets."""

    # Test v2.1 dataset
    print("=" * 80)
    print("Testing LeRobot v2.1 Dataset")
    print("=" * 80)

    dataset_path = "demo_data/empty_table_reset/empty_table_reset_lerobot_v21_video"
    loader = LeRobotDatasetLoader(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"Version: {loader.version}")
    print(f"Total episodes: {len(loader.episodes)}")

    # Test finding videos for episode 0
    episode_idx = 0
    print(f"\nFinding videos for episode {episode_idx}...")
    print("-" * 60)

    videos = find_episode_videos(loader, episode_idx)

    print(f"Found {len(videos)} video(s):")
    for camera, info in videos.items():
        print(f"  - {camera}:")
        print(f"    Path: {info['path']}")
        print(f"    Type: {info['type']}")
        if 'start_time' in info:
            print(f"    Time: {info['start_time']:.2f}s - {info['end_time']:.2f}s")

    # Verify files exist
    print("\nVerifying files exist:")
    dataset_root = loader.dataset_root
    for camera, info in videos.items():
        full_path = dataset_root / info['path']
        exists = "✓" if full_path.exists() else "✗"
        print(f"  {exists} {camera}")

    # Test v3.0 dataset
    print("\n" + "=" * 80)
    print("Testing LeRobot v3.0 Dataset")
    print("=" * 80)

    dataset_path = "demo_data/empty_table_reset/empty_table_reset_lerobot_v30"
    loader = LeRobotDatasetLoader(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"Version: {loader.version}")
    print(f"Total episodes: {len(loader.episodes)}")
    print(f"FPS: {loader.info.get('fps', 30)}")

    # Test finding videos for episode 0
    episode_idx = 0
    print(f"\nFinding videos for episode {episode_idx}...")
    print("-" * 60)

    videos = find_episode_videos(loader, episode_idx)

    print(f"Found {len(videos)} video(s):")
    for camera, info in videos.items():
        print(f"  - {camera}:")
        print(f"    Path: {info['path']}")
        print(f"    Type: {info['type']}")
        if 'start_time' in info:
            print(f"    Frames: {info['start_frame']} - {info['end_frame']}")
            print(f"    Time: {info['start_time']:.2f}s - {info['end_time']:.2f}s")

    # Verify files exist
    print("\nVerifying files exist:")
    dataset_root = loader.dataset_root
    for camera, info in videos.items():
        full_path = dataset_root / info['path']
        exists = "✓" if full_path.exists() else "✗"
        print(f"  {exists} {camera}")

    # Test multiple episodes for v3.0
    print("\n" + "=" * 80)
    print("Testing v3.0 episodes 0-2:")
    print("=" * 80)

    for ep_idx in range(min(3, len(loader.episodes))):
        videos = find_episode_videos(loader, ep_idx)
        episode_length = loader.episodes[ep_idx].get('length', 0)

        print(f"\nEpisode {ep_idx}: {episode_length} frames")
        for camera, info in sorted(videos.items()):
            if 'start_time' in info:
                print(f"  {camera}: {info['start_time']:.2f}s - {info['end_time']:.2f}s")

if __name__ == "__main__":
    test_find_videos()