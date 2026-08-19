"""Universal data loader for different LeRobot dataset versions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyarrow.parquet as pq


class LeRobotDatasetLoader:
    """Load and manage LeRobot datasets across different versions.

    Supports v2.0, v2.1, and v3.0 formats with automatic version detection.
    """

    VERSION_MAPPINGS = {
        "v2.0": {
            "state_columns": ["observation_joint_position_left", "observation_joint_position_right"],
            "action_column": "actions",
            "image_columns_pattern": "observation_.*_image",
        },
        "v2.1": {
            "state_column": "observation.state",
            "action_column": "action",
            "image_columns_pattern": "observation\\.images\\..*",
        },
        "v3.0": {
            "state_column": "observation.state",
            "action_column": "action",
            "video_columns_pattern": "observation\\.images\\..*",
        },
    }

    def __init__(self, dataset_root: Union[str, Path]):
        self.dataset_root = Path(dataset_root).resolve()
        self.info: Dict[str, Any] = {}
        self.episodes: List[Dict[str, Any]] = []
        self.version: str = ""
        self._data_cache: Optional[pq.Table] = None  # Cache for v3.0 data
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load dataset metadata from meta/ directory."""
        meta_dir = self.dataset_root / "meta"
        if not meta_dir.exists():
            raise FileNotFoundError(f"Meta directory not found: {meta_dir}")

        info_path = meta_dir / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Info file not found: {info_path}")

        self.info = json.loads(info_path.read_text())
        self.version = self.info.get("codebase_version", "v2.1")

        # Load episodes based on version
        if self.version == "v3.0":
            # v3.0 stores episodes in parquet files
            episodes_files = list(meta_dir.glob("episodes/**/*.parquet"))
            if episodes_files:
                # Load episodes from parquet
                for episodes_file in episodes_files:
                    table = pq.read_table(episodes_file)
                    for i in range(table.num_rows):
                        episode = {}
                        for col in table.column_names:
                            episode[col] = table[col][i].as_py()
                        self.episodes.append(episode)
            else:
                # Fallback: create episodes from total_episodes
                total_episodes = self.info.get("total_episodes", 0)
                for i in range(total_episodes):
                    self.episodes.append({"episode_index": i, "length": 0})
        else:
            # v2.1 and earlier use jsonl
            episodes_path = meta_dir / "episodes.jsonl"
            if episodes_path.exists():
                self.episodes = [
                    json.loads(line) for line in episodes_path.read_text().strip().split("\n") if line
                ]

    def get_feature_columns(self, dtype_filter: Optional[str] = None) -> List[str]:
        """Get all feature column names, optionally filtered by dtype.

        Args:
            dtype_filter: Filter features by dtype (e.g., 'float32', 'image', 'video')

        Returns:
            List of column names matching the filter
        """
        features = self.info.get("features", {})
        columns = []

        for name, feature in features.items():
            if dtype_filter is None:
                columns.append(name)
            elif feature.get("dtype") == dtype_filter:
                columns.append(name)

        return columns

    def get_motion_columns(self) -> List[str]:
        """Get columns suitable for motion detection (state and action)."""
        version_config = self.VERSION_MAPPINGS.get(self.version, {})
        features = self.info.get("features", {})

        motion_columns = []

        # Try to find state columns
        if "state_column" in version_config:
            col = version_config["state_column"]
            if col in features:
                motion_columns.append(col)
        elif "state_columns" in version_config:
            for col in version_config["state_columns"]:
                if col in features:
                    motion_columns.append(col)

        # Auto-detect state columns if not found
        if not motion_columns:
            for name, feat in features.items():
                if feat.get("dtype") in ["float32", "float64"]:
                    # Look for state-related columns
                    if any(keyword in name.lower() for keyword in ["state", "joint", "position"]):
                        # Skip scalar columns (like timestamp)
                        shape = feat.get("shape", [])
                        if isinstance(shape, list) and len(shape) == 1 and shape[0] > 1:
                            motion_columns.append(name)

        # Try to find action column
        action_col = version_config.get("action_column", "action")
        if action_col in features:
            motion_columns.append(action_col)
        else:
            # Auto-detect action columns
            for name, feat in features.items():
                if feat.get("dtype") in ["float32", "float64"]:
                    if "action" in name.lower() or name.lower() == "actions":
                        shape = feat.get("shape", [])
                        if isinstance(shape, list) and len(shape) == 1 and shape[0] > 1:
                            motion_columns.append(name)

        return motion_columns

    def get_episode_path(self, episode_index: int) -> Path:
        """Get the parquet file path for a specific episode."""
        if self.version == "v3.0":
            # For v3.0, we need to find the data file from episodes table
            # or scan all data files to find the one containing this episode
            data_path_template = self.info.get(
                "data_path",
                "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
            )

            # Try to get chunk/file info from episodes table
            meta_dir = self.dataset_root / "meta"
            episodes_files = list(meta_dir.glob("episodes/**/*.parquet"))

            if episodes_files:
                for episodes_file in episodes_files:
                    table = pq.read_table(episodes_file)
                    for i in range(table.num_rows):
                        ep_idx = table['episode_index'][i].as_py()
                        if ep_idx == episode_index:
                            chunk_idx = table['data/chunk_index'][i].as_py()
                            file_idx = table['data/file_index'][i].as_py()

                            rel_path = data_path_template.format(
                                chunk_index=chunk_idx,
                                file_index=file_idx
                            )
                            return self.dataset_root / rel_path

            # Fallback: scan all data files to find the episode
            # This is less efficient but works as a backup
            data_files = list((self.dataset_root / "data").glob("**/*.parquet"))
            for data_file in data_files:
                table = pq.read_table(data_file, columns=['episode_index'])
                if episode_index in [x.as_py() for x in table['episode_index']]:
                    return data_file

            raise FileNotFoundError(f"Episode {episode_index} not found in any data file")
        else:
            # For v2.x, use the old format
            data_path_template = self.info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            )

            chunks_size = self.info.get("chunks_size", 1000)
            episode_chunk = episode_index // chunks_size

            rel_path = data_path_template.format(episode_chunk=episode_chunk, episode_index=episode_index)
            return self.dataset_root / rel_path

    def get_episode_frame_timestamps(self, episode_index: int) -> List[float]:
        """Get original timestamps for each frame in an episode.

        For cleaned datasets, this returns the original_timestamp column.
        For original datasets, this returns the timestamp column.

        Args:
            episode_index: Index of the episode

        Returns:
            List of timestamps in seconds
        """
        # Try to load with original_timestamp first (cleaned data)
        try:
            table = self.load_episode(episode_index, columns=['original_timestamp', 'timestamp'])
            if 'original_timestamp' in table.column_names:
                return [t.as_py() for t in table['original_timestamp']]
            else:
                return [t.as_py() for t in table['timestamp']]
        except Exception:
            # Fallback: load all columns
            table = self.load_episode(episode_index)
            if 'original_timestamp' in table.column_names:
                return [t.as_py() for t in table['original_timestamp']]
            else:
                return [t.as_py() for t in table['timestamp']]

    def load_episode(self, episode_index: int, columns: Optional[List[str]] = None) -> pq.Table:
        """Load a single episode as a PyArrow table.

        Args:
            episode_index: Index of the episode to load
            columns: Optional list of columns to load (loads all if None)

        Returns:
            PyArrow Table containing the episode data
        """
        episode_path = self.get_episode_path(episode_index)
        if not episode_path.exists():
            raise FileNotFoundError(f"Episode file not found: {episode_path}")

        # For v3.0, the file may contain multiple episodes, so we need to filter
        if self.version == "v3.0":
            # Use cached data if available
            if self._data_cache is None:
                # Load and cache the full table
                self._data_cache = pq.read_table(episode_path)

            full_table = self._data_cache

            # If specific columns requested, select from cached table
            if columns:
                import pyarrow as pa
                # Check if episode_index is in requested columns
                if 'episode_index' not in columns:
                    columns_with_idx = columns + ['episode_index']
                    temp_table = full_table.select(columns_with_idx)
                else:
                    temp_table = full_table.select(columns)
                    columns_with_idx = columns

                # Filter by episode_index
                mask = pa.compute.equal(temp_table['episode_index'], pa.scalar(episode_index))
                result = temp_table.filter(mask)

                # Remove episode_index from result if not requested
                if 'episode_index' not in columns:
                    result = result.select(columns)

                return result
            else:
                # Load all columns, filter by episode_index
                import pyarrow as pa
                mask = pa.compute.equal(full_table['episode_index'], pa.scalar(episode_index))
                return full_table.filter(mask)
        else:
            # For v2.x, each file contains exactly one episode
            return pq.read_table(episode_path, columns=columns)

    def get_episode_lengths_batch(self) -> Dict[int, int]:
        """Get frame counts for all episodes efficiently (v3.0 only).

        Returns:
            Dictionary mapping episode_index to frame count
        """
        if self.version != "v3.0":
            # Fallback for v2.x: use metadata
            return {i: ep.get("length", 0) for i, ep in enumerate(self.episodes)}

        # For v3.0, read episode_index column once
        episode_path = self.get_episode_path(0)
        if not episode_path.exists():
            return {i: ep.get("length", 0) for i, ep in enumerate(self.episodes)}

        table = pq.read_table(episode_path, columns=['episode_index'])
        ep_indices = table['episode_index'].to_pylist()

        # Count frames per episode
        lengths = {}
        for ep_idx in range(len(self.episodes)):
            lengths[ep_idx] = sum(1 for idx in ep_indices if idx == ep_idx)

        return lengths

    def load_episode_as_numpy(
        self, episode_index: int, columns: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Load a single episode and convert to numpy arrays.

        Args:
            episode_index: Index of the episode to load
            columns: Optional list of columns to load

        Returns:
            Dictionary mapping column names to numpy arrays
        """
        table = self.load_episode(episode_index, columns)
        return {col: np.asarray(table[col].to_pylist(), dtype=np.float64) for col in table.column_names}

    def get_image_columns(self) -> List[str]:
        """Get all image/video columns based on version."""
        features = self.info.get("features", {})
        image_columns = []

        for name, feat in features.items():
            dtype = feat.get("dtype", "")
            if dtype in ["image", "video"]:
                image_columns.append(name)

        return image_columns

    def get_episode_video_info(self, episode_index: int) -> Dict[str, Dict[str, Any]]:
        """Get video file information for a specific episode (v3.0 only).

        Args:
            episode_index: Index of the episode

        Returns:
            Dictionary mapping video column names to video info dicts
        """
        if self.version != "v3.0":
            return {}

        if episode_index >= len(self.episodes):
            return {}

        # Load episodes parquet file to get video info
        meta_dir = self.dataset_root / "meta"
        episodes_files = list(meta_dir.glob("episodes/**/*.parquet"))

        if not episodes_files:
            return {}

        # Load all episodes to find the one we need
        for episodes_file in episodes_files:
            table = pq.read_table(episodes_file)
            for i in range(table.num_rows):
                ep_idx = table['episode_index'][i].as_py()
                if ep_idx == episode_index:
                    # Found the episode, extract video info
                    video_info = {}
                    video_columns = self.get_image_columns()

                    for video_col in video_columns:
                        # Check if video column exists in episodes table
                        chunk_col = f"videos/{video_col}/chunk_index"
                        file_col = f"videos/{video_col}/file_index"
                        from_col = f"videos/{video_col}/from_timestamp"
                        to_col = f"videos/{video_col}/to_timestamp"

                        if chunk_col in table.column_names:
                            chunk_idx = table[chunk_col][i].as_py()
                            file_idx = table[file_col][i].as_py()
                            from_ts = table[from_col][i].as_py()
                            to_ts = table[to_col][i].as_py()

                            video_path_template = self.info.get(
                                "video_path",
                                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
                            )
                            video_path = video_path_template.format(
                                video_key=video_col,
                                chunk_index=chunk_idx,
                                file_index=file_idx
                            )

                            video_info[video_col] = {
                                "path": video_path,
                                "chunk_index": chunk_idx,
                                "file_index": file_idx,
                                "from_timestamp": float(from_ts),
                                "to_timestamp": float(to_ts)
                            }

                    return video_info

        return {}

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        return {
            "version": self.version,
            "robot_type": self.info.get("robot_type", "unknown"),
            "total_episodes": self.info.get("total_episodes", 0),
            "total_frames": self.info.get("total_frames", 0),
            "fps": self.info.get("fps", 30),
            "features": list(self.info.get("features", {}).keys()),
        }

    def __repr__(self) -> str:
        stats = self.get_dataset_stats()
        return (
            f"LeRobotDatasetLoader(\n"
            f"  root={self.dataset_root},\n"
            f"  version={stats['version']},\n"
            f"  episodes={stats['total_episodes']},\n"
            f"  frames={stats['total_frames']}\n"
            f")"
        )