"""Flask web application for viewing LeRobot dataset videos."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, Response, render_template, send_file, jsonify, request

from ..core.loader import LeRobotDatasetLoader


def create_app(dataset_path: Optional[str] = None) -> Flask:
    """Create Flask application for video viewing.

    Args:
        dataset_path: Path to dataset to load at startup

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    app.config["DATASET_PATH"] = dataset_path

    loader: Optional[LeRobotDatasetLoader] = None

    def get_loader() -> LeRobotDatasetLoader:
        """Get or create dataset loader."""
        nonlocal loader
        if loader is None:
            dataset_path = app.config.get("DATASET_PATH")
            if dataset_path:
                loader = LeRobotDatasetLoader(dataset_path)
            else:
                raise ValueError("No dataset loaded. Please provide dataset_path.")
        return loader

    @app.route("/")
    def index():
        """Main page showing dataset overview."""
        dataset_path = app.config.get("DATASET_PATH")

        # Check if we should show input form (no dataset or explicit request)
        show_input = request.args.get("input", "false").lower() == "true"

        # If no dataset loaded or explicit request for input, show input form
        if not dataset_path or not loader or show_input:
            return render_template("input_path.html")

        try:
            loader_obj = get_loader()
            episode_lengths = {}
            for i, ep in enumerate(loader_obj.episodes):
                episode_lengths[i] = ep.get("length", 0)

            return render_template(
                "index.html",
                dataset_info=loader_obj.info,
                episode_lengths=episode_lengths,
                total_episodes=len(loader_obj.episodes),
                total_frames=sum(episode_lengths.values()),
                dataset_path=dataset_path
            )
        except Exception as e:
            return render_template("error.html", error=str(e))

    @app.route("/load", methods=["POST"])
    def load_dataset():
        """Load dataset from path."""
        dataset_path = request.form.get("dataset_path")
        if not dataset_path:
            return render_template("error.html", error="请输入数据集路径")

        try:
            # Validate path exists
            path = Path(dataset_path)
            if not path.exists():
                return render_template("error.html", error=f"路径不存在: {dataset_path}")

            # Load dataset
            nonlocal loader
            loader = LeRobotDatasetLoader(dataset_path)
            app.config["DATASET_PATH"] = dataset_path

            episode_lengths = {}
            for i, ep in enumerate(loader.episodes):
                episode_lengths[i] = ep.get("length", 0)

            return render_template(
                "index.html",
                dataset_info=loader.info,
                episode_lengths=episode_lengths,
                total_episodes=len(loader.episodes),
                total_frames=sum(episode_lengths.values()),
                dataset_path=dataset_path
            )
        except Exception as e:
            return render_template("error.html", error=f"加载数据集失败: {str(e)}")

    @app.route("/episode/<int:episode_idx>")
    def episode_detail(episode_idx: int):
        """Show episode detail with video player."""
        try:
            loader_obj = get_loader()
            if episode_idx >= len(loader_obj.episodes):
                return render_template("error.html", error=f"Episode {episode_idx} 不存在")

            episode_info = loader_obj.episodes[episode_idx]

            video_files = find_episode_videos(loader_obj, episode_idx)

            # Get frame timestamps for video synchronization
            frame_timestamps = loader_obj.get_episode_frame_timestamps(episode_idx)

            return render_template(
                "episode.html",
                episode_idx=episode_idx,
                episode_info=episode_info,
                video_files=video_files,
                frame_timestamps=frame_timestamps,
                total_episodes=len(loader_obj.episodes),
                dataset_path=app.config.get("DATASET_PATH", "")
            )
        except Exception as e:
            return render_template("error.html", error=str(e))

    @app.route("/video/<path:video_path>")
    def serve_video(video_path: str):
        """Serve video file."""
        try:
            loader_obj = get_loader()
            full_path = loader_obj.dataset_root / video_path

            if not full_path.exists():
                return "Video not found", 404

            return send_file(str(full_path))
        except Exception as e:
            return str(e), 500

    @app.route("/api/load_dataset")
    def api_load_dataset():
        """Load dataset from path (for dynamic loading)."""
        dataset_path = request.args.get("path")
        if not dataset_path:
            return jsonify({"error": "No path provided"}), 400

        try:
            nonlocal loader
            loader = LeRobotDatasetLoader(dataset_path)
            app.config["DATASET_PATH"] = dataset_path

            episode_lengths = {}
            for i, ep in enumerate(loader.episodes):
                episode_lengths[i] = ep.get("length", 0)

            return jsonify({
                "success": True,
                "info": loader.info,
                "episode_lengths": episode_lengths
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/episode_lengths")
    def api_episode_lengths():
        """Get episode lengths."""
        try:
            loader_obj = get_loader()
            lengths = {}
            for i, ep in enumerate(loader_obj.episodes):
                lengths[i] = ep.get("length", 0)
            return jsonify(lengths)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def find_episode_videos(loader: LeRobotDatasetLoader, episode_idx: int) -> Dict[str, Dict]:
    """Find video files for a specific episode.

    Args:
        loader: Dataset loader
        episode_idx: Episode index

    Returns:
        Dictionary mapping video key to video info dict with 'path' and optionally 'start_time'/'end_time'
    """
    videos = {}

    dataset_root = loader.dataset_root
    version = loader.version

    # For v2.1: Each episode has its own video file
    if version == "v2.1":
        for pattern in ["videos/**/*.mp4", "videos/*.mp4"]:
            video_files = list(dataset_root.glob(pattern))

            for video_file in video_files:
                file_name = video_file.name

                # Check if this video is for the requested episode
                if f"episode_{episode_idx:06d}" in file_name:
                    # Extract camera name from path
                    camera_name = video_file.parent.name

                    videos[camera_name] = {
                        "path": str(video_file.relative_to(dataset_root)),
                        "type": "file"
                    }

    # For v3.0: Use episodes table to get exact video file and timestamps
    elif version == "v3.0":
        video_info = loader.get_episode_video_info(episode_idx)

        for video_key, info in video_info.items():
            videos[video_key] = {
                "path": info["path"],
                "type": "chunk",
                "start_time": info["from_timestamp"],
                "end_time": info["to_timestamp"]
            }

    return videos


def run_web_interface(dataset_path: str = None, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run the web interface.

    Args:
        dataset_path: Path to LeRobot dataset (optional, can be input via web)
        host: Host to bind to
        port: Port to bind to
    """
    app = create_app(dataset_path)
    print(f"\n启动 RoboClean Web 界面...")
    if dataset_path:
        print(f"数据集路径: {dataset_path}")
    else:
        print("请在 Web 界面中输入数据集路径")
    print(f"请在浏览器中打开: http://{host}:{port}")
    print(f"按 Ctrl+C 停止服务器\n")
    app.run(host=host, port=port, debug=False)