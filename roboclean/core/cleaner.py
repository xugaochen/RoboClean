"""Data cleaning operations for LeRobot datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from .loader import LeRobotDatasetLoader


MotionNorm = Literal["max", "l2"]
CompareTo = Literal["previous", "last-kept"]


class DataCleaner:
    """Universal cleaner for LeRobot datasets.

    Supports cleaning operations like:
    - Removing static frames
    - Removing frames with zero dimensions
    - Filtering by motion threshold
    - Episode statistics calculation
    """

    def __init__(self, loader: LeRobotDatasetLoader):
        self.loader = loader

    def _motion_delta(self, a: np.ndarray, b: np.ndarray, norm: MotionNorm) -> float:
        """Calculate motion delta between two arrays."""
        diff = np.abs(a - b)
        if norm == "max":
            return float(np.max(diff))
        if norm == "l2":
            return float(np.linalg.norm(diff))
        raise ValueError(f"Unsupported norm: {norm}")

    def build_keep_mask(
        self,
        table: pa.Table,
        motion_columns: List[str],
        threshold: float = 1e-3,
        norm: MotionNorm = "max",
        compare_to: CompareTo = "previous",
        drop_zero_rows: bool = False,
        zero_dim_specs: Optional[Dict[str, List[int]]] = None,
        zero_atol: float = 1e-8,
        keep_empty_fallback: bool = True,
    ) -> Tuple[np.ndarray, List[float]]:
        """Build a mask for frames to keep based on motion criteria.

        Args:
            table: PyArrow table containing episode data
            motion_columns: Columns to use for motion detection
            threshold: Motion threshold for keeping frames
            norm: Normalization method for motion delta
            compare_to: Frame comparison method
            drop_zero_rows: Whether to drop rows with all-zero motion columns
            zero_dim_specs: Dimensions that should be zero to drop a row
            zero_atol: Tolerance for zero detection
            keep_empty_fallback: Keep at least one frame if all are dropped

        Returns:
            Tuple of (keep_mask, motion_scores)
        """
        num_rows = table.num_rows
        if num_rows == 0:
            return np.zeros(0, dtype=bool), []

        zero_dim_specs = zero_dim_specs or {}
        needed_columns = sorted(set(motion_columns) | set(zero_dim_specs.keys()))

        # Load arrays
        arrays = {}
        for column in needed_columns:
            if table.schema.get_field_index(column) >= 0:
                arrays[column] = np.asarray(table[column].to_pylist(), dtype=np.float64)

        keep = np.zeros(num_rows, dtype=bool)
        keep[0] = True
        last_kept = 0
        motion_scores = [float("inf")]

        # Check first frame
        if drop_zero_rows or zero_dim_specs:
            first_is_zero = drop_zero_rows and any(
                np.allclose(arrays[col][0], 0.0) for col in motion_columns if col in arrays
            )
            first_has_zero_dim = self._has_zero_dim(arrays, zero_dim_specs, 0, zero_atol)
            if first_is_zero or first_has_zero_dim:
                keep[0] = False
                motion_scores[0] = 0.0

        # Process remaining frames
        for i in range(1, num_rows):
            row_is_zero = drop_zero_rows and any(
                np.allclose(arrays[col][i], 0.0) for col in motion_columns if col in arrays
            )
            if row_is_zero or self._has_zero_dim(arrays, zero_dim_specs, i, zero_atol):
                motion_scores.append(0.0)
                continue

            ref = i - 1 if compare_to == "previous" else last_kept
            # Only calculate score for columns that exist in the data
            available_columns = [col for col in motion_columns if col in arrays]
            if available_columns:
                score = max(
                    self._motion_delta(arrays[col][i], arrays[col][ref], norm)
                    for col in available_columns
                )
            else:
                score = 0.0
            motion_scores.append(score)
            if score > threshold:
                keep[i] = True
                last_kept = i

        # Avoid empty episodes
        if keep_empty_fallback and not np.any(keep):
            fallback = 0
            if drop_zero_rows or zero_dim_specs:
                valid_rows = [
                    i
                    for i in range(num_rows)
                    if not (
                        drop_zero_rows
                        and any(np.allclose(arrays[col][i], 0.0) for col in motion_columns if col in arrays)
                    )
                    and not self._has_zero_dim(arrays, zero_dim_specs, i, zero_atol)
                ]
                fallback = valid_rows[0] if valid_rows else 0
            keep[fallback] = True

        return keep, motion_scores

    def _has_zero_dim(
        self,
        arrays: Dict[str, np.ndarray],
        zero_dim_specs: Dict[str, List[int]],
        row: int,
        zero_atol: float,
    ) -> bool:
        """Check if specified dimensions are zero for a row."""
        for column, dims in zero_dim_specs.items():
            if column in arrays:
                values = arrays[column][row, dims]
                if np.any(np.isclose(values, 0.0, atol=zero_atol)):
                    return True
        return False

    def clean_dataset(
        self,
        output_root: Optional[Union[str, Path]] = None,
        motion_columns: Optional[List[str]] = None,
        threshold: float = 1e-3,
        norm: MotionNorm = "max",
        compare_to: CompareTo = "previous",
        drop_zero_rows: bool = False,
        zero_dim_specs: Optional[Dict[str, List[int]]] = None,
        zero_atol: float = 1e-8,
        drop_empty_episodes: bool = False,
        overwrite: bool = False,
        in_place: bool = False,
        backup_root: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Clean the entire dataset and save to output directory.

        Args:
            output_root: Output directory path
            motion_columns: Columns for motion detection
            threshold: Motion threshold
            norm: Normalization method
            compare_to: Comparison method
            drop_zero_rows: Drop all-zero rows
            zero_dim_specs: Zero dimension specifications
            zero_atol: Zero detection tolerance
            drop_empty_episodes: Drop empty episodes
            overwrite: Overwrite existing output
            in_place: Modify dataset in place
            backup_root: Backup directory for in-place cleaning

        Returns:
            Path to cleaned dataset
        """
        import shutil

        dataset_root = self.loader.dataset_root
        output_root = Path(output_root) if output_root else None

        if motion_columns is None:
            motion_columns = self.loader.get_motion_columns()

        if in_place:
            output_root = dataset_root
            backup_root = Path(backup_root) if backup_root else dataset_root.with_name(
                f"{dataset_root.name}_backup"
            )
            if backup_root.exists() and not overwrite:
                raise FileExistsError(f"Backup already exists: {backup_root}")
            shutil.copytree(dataset_root, backup_root)
            print(f"Backed up to: {backup_root}")
        else:
            output_root = output_root or dataset_root.with_name(f"{dataset_root.name}_cleaned")
            if output_root.exists() and not overwrite:
                raise FileExistsError(f"Output exists: {output_root}. Use --overwrite")

            # Copy non-data directories
            shutil.copytree(dataset_root, output_root, ignore=shutil.ignore_patterns("data", ".clean_*"))
            (output_root / "data").mkdir(parents=True, exist_ok=True)

        # Process episodes
        cleaned_episodes = []
        global_index = 0
        total_in = 0
        total_out = 0

        fps = self.loader.info.get("fps", 30)

        # For v3.0, we need to group episodes by data file
        if self.loader.version == "v3.0":
            # Group episodes by their data file
            episodes_by_file = {}
            for episode in self.loader.episodes:
                episode_index = int(episode["episode_index"])
                episode_path = self.loader.get_episode_path(episode_index)
                file_key = str(episode_path.relative_to(dataset_root))

                if file_key not in episodes_by_file:
                    episodes_by_file[file_key] = []
                episodes_by_file[file_key].append((episode_index, episode))

            # Process each data file
            for file_key, episodes_in_file in tqdm.tqdm(episodes_by_file.items(), desc="Cleaning data files"):
                # Load the full data file
                data_file = dataset_root / file_key
                full_table = pq.read_table(data_file)

                # Process each episode in this file
                cleaned_tables = []
                for episode_index, episode in episodes_in_file:
                    # Extract episode data
                    import pyarrow as pa
                    mask = pa.compute.equal(full_table['episode_index'], pa.scalar(episode_index))
                    table = full_table.filter(mask)

                    # Clean this episode
                    keep_mask, scores = self.build_keep_mask(
                        table,
                        motion_columns=motion_columns,
                        threshold=threshold,
                        norm=norm,
                        compare_to=compare_to,
                        drop_zero_rows=drop_zero_rows,
                        zero_dim_specs=zero_dim_specs,
                        zero_atol=zero_atol,
                        keep_empty_fallback=not drop_empty_episodes,
                    )

                    indices = np.flatnonzero(keep_mask)
                    length = len(indices)
                    total_in += table.num_rows

                    if length == 0:
                        print(f"Episode {episode_index}: dropped (0 frames kept)")
                        continue

                    # Clean and rewrite
                    cleaned = table.take(pa.array(indices, type=pa.int64()))
                    cleaned = self._rewrite_indices(cleaned, episode_index, global_index, fps)
                    cleaned_tables.append(cleaned)

                    global_index += length
                    total_out += length

                    new_episode = dict(episode)
                    new_episode["length"] = length
                    cleaned_episodes.append(new_episode)

                    print(f"Episode {episode_index}: {table.num_rows} -> {length} frames")

                # Combine all cleaned episodes for this file
                if cleaned_tables:
                    combined = pa.concat_tables(cleaned_tables)
                    output_path = output_root / file_key
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(combined, output_path)
        else:
            # For v2.x, process each episode separately
            for episode in tqdm.tqdm(self.loader.episodes, desc="Cleaning episodes"):
                episode_index = int(episode["episode_index"])
                table = self.loader.load_episode(episode_index)

                keep_mask, scores = self.build_keep_mask(
                    table,
                    motion_columns=motion_columns,
                    threshold=threshold,
                    norm=norm,
                    compare_to=compare_to,
                    drop_zero_rows=drop_zero_rows,
                    zero_dim_specs=zero_dim_specs,
                    zero_atol=zero_atol,
                    keep_empty_fallback=not drop_empty_episodes,
                )

                indices = np.flatnonzero(keep_mask)
                length = len(indices)
                total_in += table.num_rows

                if length == 0:
                    print(f"Episode {episode_index}: dropped (0 frames kept)")
                    continue

                # Clean and rewrite
                cleaned = table.take(pa.array(indices, type=pa.int64()))
                cleaned = self._rewrite_indices(cleaned, episode_index, global_index, fps)

                # Save
                output_path = self.loader.get_episode_path(episode_index)
                output_path = output_root / output_path.relative_to(dataset_root)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(cleaned, output_path)

                global_index += length
                total_out += length

                new_episode = dict(episode)
                new_episode["length"] = length
                cleaned_episodes.append(new_episode)

                print(f"Episode {episode_index}: {table.num_rows} -> {length} frames")

        # Update metadata
        self._update_metadata(output_root, cleaned_episodes, total_out)

        print(f"\nTotal: {total_in} -> {total_out} frames (dropped {total_in - total_out})")
        print(f"Cleaned dataset: {output_root}")

        return output_root

    def _rewrite_indices(
        self, table: pa.Table, episode_index: int, global_start: int, fps: float
    ) -> pa.Table:
        """Update episode_index and global index after cleaning, preserving original timestamps.

        Note: We keep original timestamp and frame_index values unchanged to maintain
        video synchronization without needing to update episode metadata.
        """
        length = table.num_rows

        # Keep original timestamp and frame_index unchanged for video sync
        # Only update episode_index and global index
        table = self._replace_column(table, "episode_index", np.full(length, episode_index, dtype=np.int64))
        table = self._replace_column(
            table, "index", np.arange(global_start, global_start + length, dtype=np.int64)
        )

        return table

    def _replace_column(self, table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
        """Replace a column in the table."""
        idx = table.schema.get_field_index(name)
        if idx < 0:
            return table
        field = table.schema.field(idx)
        return table.set_column(idx, field, pa.array(values, type=field.type))

    def _update_metadata(
        self, output_root: Path, cleaned_episodes: List[Dict], total_frames: int
    ) -> None:
        """Update metadata files after cleaning."""
        info = dict(self.loader.info)
        info["total_frames"] = total_frames
        info["total_episodes"] = len(cleaned_episodes)

        if "splits" in info and "train" in info["splits"]:
            info["splits"]["train"] = f"0:{len(cleaned_episodes)}"

        meta_dir = output_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        (meta_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

        # Write episodes.jsonl
        episodes_jsonl = "\n".join(json.dumps(ep, ensure_ascii=False) for ep in cleaned_episodes)
        (meta_dir / "episodes.jsonl").write_text(episodes_jsonl + "\n")