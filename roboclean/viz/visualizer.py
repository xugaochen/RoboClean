"""Visualization tools for LeRobot datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Optional: seaborn for better plot styles
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ..core.loader import LeRobotDatasetLoader


class DatasetVisualizer:
    """Visualize LeRobot datasets for analysis and debugging."""

    def __init__(self, loader: LeRobotDatasetLoader):
        self.loader = loader
        # Use seaborn style if available, otherwise use default
        if HAS_SEABORN:
            try:
                plt.style.use("seaborn-v0_8-darkgrid")
            except:
                pass  # Use default style if seaborn style not available

    def plot_motion_over_episode(
        self,
        episode_index: int,
        motion_columns: Optional[List[str]] = None,
        figsize: tuple = (12, 6),
    ) -> Figure:
        """Plot motion magnitude over time for a single episode.

        Args:
            episode_index: Episode to visualize
            motion_columns: Columns to use for motion calculation
            figsize: Figure size

        Returns:
            Matplotlib Figure object
        """
        if motion_columns is None:
            motion_columns = self.loader.get_motion_columns()

        data = self.loader.load_episode_as_numpy(episode_index, motion_columns)

        fig, axes = plt.subplots(len(motion_columns), 1, figsize=figsize, sharex=True)
        if len(motion_columns) == 1:
            axes = [axes]

        for ax, col in zip(axes, motion_columns):
            if col in data:
                values = data[col]
                # Plot each dimension
                for dim in range(values.shape[1]):
                    ax.plot(values[:, dim], alpha=0.7, label=f"{col}[{dim}]")
                ax.set_ylabel(col)
                ax.legend(loc="upper right", fontsize="small")
                ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Frame")
        fig.suptitle(f"Episode {episode_index} Motion Analysis")
        fig.tight_layout()

        return fig

    def get_episode_lengths(self) -> Dict[int, int]:
        """Get length of each episode.

        Returns:
            Dictionary mapping episode index to its length (number of frames)
        """
        lengths = {}
        for i, ep in enumerate(self.loader.episodes):
            lengths[i] = ep.get("length", 0)
        return lengths

    def print_episode_lengths(self) -> None:
        """Print each episode's length to console."""
        lengths = self.get_episode_lengths()
        print("\nEpisode Lengths:")
        print("-" * 40)
        for ep_idx, length in lengths.items():
            print(f"Episode {ep_idx:04d}: {length:5d} frames")
        print("-" * 40)
        print(f"Total: {len(lengths)} episodes, {sum(lengths.values())} frames")
        print(f"Average: {np.mean(list(lengths.values())):.1f} frames")

    def plot_episode_length_distribution(self, figsize: tuple = (10, 6)) -> Figure:
        """Plot distribution of episode lengths.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib Figure object
        """
        lengths = [ep.get("length", 0) for ep in self.loader.episodes]

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(lengths, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(lengths), color="red", linestyle="--", label=f"Mean: {np.mean(lengths):.1f}")
        ax.axvline(
            np.median(lengths), color="green", linestyle="--", label=f"Median: {np.median(lengths):.1f}"
        )
        ax.set_xlabel("Episode Length (frames)")
        ax.set_ylabel("Count")
        ax.set_title("Episode Length Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_feature_statistics(
        self, episode_indices: Optional[List[int]] = None, max_episodes: int = 10, figsize: tuple = (15, 8)
    ) -> Figure:
        """Plot statistics for numerical features across episodes.

        Args:
            episode_indices: Specific episodes to analyze (None for first max_episodes)
            max_episodes: Maximum number of episodes to analyze
            figsize: Figure size

        Returns:
            Matplotlib Figure object
        """
        if episode_indices is None:
            episode_indices = list(range(min(max_episodes, len(self.loader.episodes))))

        motion_columns = self.loader.get_motion_columns()
        all_data = {col: [] for col in motion_columns}

        for ep_idx in episode_indices:
            data = self.loader.load_episode_as_numpy(ep_idx, motion_columns)
            for col in motion_columns:
                if col in data:
                    all_data[col].append(data[col])

        # Concatenate all episodes
        for col in motion_columns:
            if all_data[col]:
                all_data[col] = np.concatenate(all_data[col], axis=0)

        # Plot
        n_cols = len(motion_columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=figsize, sharex=True)
        if n_cols == 1:
            axes = [axes]

        for ax, col in zip(axes, motion_columns):
            if col in all_data and len(all_data[col]) > 0:
                data = all_data[col]
                # Box plot for each dimension
                ax.boxplot([data[:, i] for i in range(min(data.shape[1], 20))], vert=False)
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Dimension Index")
        fig.suptitle("Feature Statistics Across Episodes")
        fig.tight_layout()

        return fig

    def plot_motion_heatmap(
        self, episode_indices: Optional[List[int]] = None, motion_column: str = "action", figsize: tuple = (12, 8)
    ) -> Figure:
        """Plot heatmap of motion values for selected episodes.

        Args:
            episode_indices: Episodes to visualize
            motion_column: Column to visualize
            figsize: Figure size

        Returns:
            Matplotlib Figure object
        """
        if episode_indices is None:
            episode_indices = list(range(min(5, len(self.loader.episodes))))

        # Collect data
        episode_data = []
        for ep_idx in episode_indices:
            data = self.loader.load_episode_as_numpy(ep_idx, [motion_column])
            if motion_column in data:
                episode_data.append(data[motion_column])

        # Create combined heatmap
        max_len = max(len(d) for d in episode_data)
        n_dims = episode_data[0].shape[1] if episode_data else 0

        heatmap_data = np.zeros((len(episode_indices) * n_dims, max_len))
        for i, d in enumerate(episode_data):
            for dim in range(n_dims):
                row = i * n_dims + dim
                heatmap_data[row, : len(d)] = d[:, dim]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(heatmap_data, aspect="auto", cmap="viridis")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Episode × Dimension")
        ax.set_title(f"{motion_column} Heatmap Across Episodes")
        fig.colorbar(im, ax=ax)

        return fig

    def generate_report(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Generate a comprehensive visual report of the dataset.

        Args:
            output_dir: Directory to save plots (None for no saving)

        Returns:
            Dictionary with report information
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "dataset_stats": self.loader.get_dataset_stats(),
            "figures": {},
        }

        # Generate plots
        plots = [
            ("episode_lengths", self.plot_episode_length_distribution()),
            ("feature_stats", self.plot_feature_statistics()),
        ]

        for name, fig in plots:
            if output_dir:
                fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
            report["figures"][name] = fig

        return report

    def show(self) -> None:
        """Display all generated plots."""
        plt.show()

    def close(self) -> None:
        """Close all figures to free memory."""
        plt.close("all")