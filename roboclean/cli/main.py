"""Command-line interface for RoboClean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

from ..core.cleaner import DataCleaner
from ..core.loader import LeRobotDatasetLoader
from ..viz.visualizer import DatasetVisualizer

try:
    from ..web import run_web_interface
    HAS_WEB = True
except ImportError:
    HAS_WEB = False


def parse_zero_dim_specs(specs: List[str]) -> Dict[str, List[int]]:
    """Parse zero dimension specifications from command line.

    Args:
        specs: List of specs like 'action:14,15' or 'observation.state:0,1'

    Returns:
        Dictionary mapping column names to dimension indices
    """
    parsed = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid zero-dim spec: {spec}. Expected format: column:dim1,dim2,...")
        column, dims = spec.split(":", 1)
        parsed.setdefault(column, []).extend(int(dim) for dim in dims.split(",") if dim)
    return parsed


def cmd_info(args: argparse.Namespace) -> None:
    """Show dataset information."""
    console = Console()

    try:
        loader = LeRobotDatasetLoader(args.dataset_path)
        stats = loader.get_dataset_stats()

        # Display basic info
        console.print(f"\n[bold cyan]Dataset: {args.dataset_path}[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            if isinstance(value, list):
                value = f"{len(value)} features"
            table.add_row(key, str(value))

        console.print(table)

        # Show motion columns
        motion_columns = loader.get_motion_columns()
        console.print(f"\n[bold]Motion columns:[/bold] {', '.join(motion_columns)}")

        # Show image/video columns
        image_columns = loader.get_image_columns()
        if image_columns:
            console.print(f"[bold]Image/Video columns:[/bold] {', '.join(image_columns)}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def cmd_clean(args: argparse.Namespace) -> None:
    """Clean dataset."""
    console = Console()

    try:
        loader = LeRobotDatasetLoader(args.dataset_path)
        cleaner = DataCleaner(loader)

        zero_dim_specs = parse_zero_dim_specs(args.drop_zero_dims) if args.drop_zero_dims else None

        output_path = cleaner.clean_dataset(
            output_root=args.output_path,
            motion_columns=args.motion_columns if args.motion_columns else None,
            threshold=args.threshold,
            norm=args.norm,
            compare_to=args.compare_to,
            drop_zero_rows=args.drop_zero_rows,
            zero_dim_specs=zero_dim_specs,
            zero_atol=args.zero_atol,
            drop_empty_episodes=args.drop_empty_episodes,
            overwrite=args.overwrite,
            in_place=args.in_place,
            backup_root=args.backup_path,
        )

        console.print(f"\n[green]✓ Dataset cleaned successfully: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def cmd_visualize(args: argparse.Namespace) -> None:
    """Visualize dataset."""
    console = Console()

    try:
        loader = LeRobotDatasetLoader(args.dataset_path)
        visualizer = DatasetVisualizer(loader)

        console.print(f"\n[cyan]Generating visualizations for {args.dataset_path}...[/cyan]\n")

        if args.report:
            # Generate full report
            output_dir = args.output_dir or Path("roboclean_report")
            report = visualizer.generate_report(output_dir)
            console.print(f"[green]✓ Report saved to: {output_dir}[/green]")
            console.print(f"  - Dataset stats: {report['dataset_stats']}")
        else:
            # Generate specific plots
            if args.episode_lengths:
                fig = visualizer.plot_episode_length_distribution()
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(output_dir / "episode_lengths.png", dpi=150, bbox_inches="tight")
                    console.print(f"[green]✓ Saved: {output_dir / 'episode_lengths.png'}[/green]")

            if args.episode_motion:
                fig = visualizer.plot_motion_over_episode(int(args.episode_motion))
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(output_dir / f"episode_{args.episode_motion}_motion.png", dpi=150, bbox_inches="tight")
                    console.print(f"[green]✓ Saved motion plot for episode {args.episode_motion}[/green]")

            if args.print_lengths:
                visualizer.print_episode_lengths()

        if args.show:
            visualizer.show()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def cmd_web(args: argparse.Namespace) -> None:
    """Start web interface."""
    if not HAS_WEB:
        console = Console()
        console.print("[red]Error: Flask not installed. Please install it with:[/red]")
        console.print("  pip install flask")
        return

    run_web_interface(
        dataset_path=str(args.dataset_path) if args.dataset_path else None,
        host=args.host,
        port=args.port
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="roboclean",
        description="Universal LeRobot dataset cleaning and visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.add_argument("dataset_path", type=Path, help="Path to LeRobot dataset")

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean dataset")
    clean_parser.add_argument("dataset_path", type=Path, help="Path to LeRobot dataset")
    clean_parser.add_argument("--output-path", type=Path, help="Output path for cleaned dataset")
    clean_parser.add_argument("--motion-columns", nargs="+", help="Columns for motion detection")
    clean_parser.add_argument("--threshold", type=float, default=1e-3, help="Motion threshold")
    clean_parser.add_argument("--norm", choices=["max", "l2"], default="max", help="Motion norm")
    clean_parser.add_argument("--compare-to", choices=["previous", "last-kept"], default="previous")
    clean_parser.add_argument("--drop-zero-rows", action="store_true", help="Drop all-zero rows")
    clean_parser.add_argument(
        "--drop-zero-dims", nargs="*", help="Drop rows with specified zero dims (e.g., action:14,15)"
    )
    clean_parser.add_argument("--zero-atol", type=float, default=1e-8)
    clean_parser.add_argument("--drop-empty-episodes", action="store_true")
    clean_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    clean_parser.add_argument("--in-place", action="store_true", help="Modify dataset in place")
    clean_parser.add_argument("--backup-path", type=Path, help="Backup path for in-place cleaning")

    # Visualize command
    viz_parser = subparsers.add_parser("viz", help="Visualize dataset")
    viz_parser.add_argument("dataset_path", type=Path, help="Path to LeRobot dataset")
    viz_parser.add_argument("--report", action="store_true", help="Generate full report")
    viz_parser.add_argument("--episode-lengths", action="store_true", help="Plot episode lengths")
    viz_parser.add_argument("--episode-motion", type=int, help="Plot motion for specific episode")
    viz_parser.add_argument("--print-lengths", action="store_true", help="Print each episode length")
    viz_parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    viz_parser.add_argument("--show", action="store_true", help="Show plots interactively")

    # Web interface command
    if HAS_WEB:
        web_parser = subparsers.add_parser("web", help="Start web interface for video viewing")
        web_parser.add_argument("dataset_path", type=Path, nargs="?", help="Path to LeRobot dataset (optional)")
        web_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
        web_parser.add_argument("--port", type=int, default=5000, help="Port to bind to (default: 5000)")

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "viz":
        cmd_visualize(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()