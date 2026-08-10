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
            original_episode_lengths = {}
            for i, ep in enumerate(loader_obj.episodes):
                # Use cleaned frame count (actual rows in parquet)
                try:
                    episode_table = loader_obj.load_episode(i)
                    episode_lengths[i] = episode_table.num_rows
                except Exception:
                    # Fallback to metadata if loading fails
                    episode_lengths[i] = ep.get("length", 0)

                # Store original video frame count from metadata
                original_episode_lengths[i] = ep.get("length", 0)

            return render_template(
                "index.html",
                dataset_info=loader_obj.info,
                episode_lengths=episode_lengths,
                original_episode_lengths=original_episode_lengths,
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
            original_episode_lengths = {}
            for i, ep in enumerate(loader.episodes):
                # Use cleaned frame count (actual rows in parquet)
                try:
                    episode_table = loader.load_episode(i)
                    episode_lengths[i] = episode_table.num_rows
                except Exception:
                    # Fallback to metadata if loading fails
                    episode_lengths[i] = ep.get("length", 0)

                # Store original video frame count from metadata
                original_episode_lengths[i] = ep.get("length", 0)

            return render_template(
                "index.html",
                dataset_info=loader.info,
                episode_lengths=episode_lengths,
                original_episode_lengths=original_episode_lengths,
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

            # Get cleaned frame count (actual rows in parquet)
            try:
                episode_table = loader_obj.load_episode(episode_idx)
                cleaned_frame_count = episode_table.num_rows
            except Exception:
                cleaned_frame_count = len(frame_timestamps)

            # Get original frame count from metadata
            original_frame_count = episode_info.get("length", 0)

            return render_template(
                "episode.html",
                episode_idx=episode_idx,
                episode_info=episode_info,
                video_files=video_files,
                frame_timestamps=frame_timestamps,
                cleaned_frame_count=cleaned_frame_count,
                original_frame_count=original_frame_count,
                total_episodes=len(loader_obj.episodes),
                dataset_path=app.config.get("DATASET_PATH", "")
            )
        except Exception as e:
            return render_template("error.html", error=str(e))

    @app.route("/api/delete_frame/<int:episode_idx>/<int:frame_idx>", methods=["POST"])
    def delete_frame(episode_idx: int, frame_idx: int):
        """Delete a specific frame from an episode."""
        try:
            loader_obj = get_loader()
            
            if episode_idx >= len(loader_obj.episodes):
                return jsonify({"success": False, "error": f"Episode {episode_idx} 不存在"})

            import pyarrow as pa
            import pyarrow.parquet as pq
            import numpy as np
            
            # For v3.0, need to handle multiple episodes in one file
            if loader_obj.version == "v3.0":
                # Get the data file path
                episode_path = loader_obj.get_episode_path(episode_idx)
                
                # Load the full data file
                full_table = pq.read_table(episode_path)
                
                # Filter to get current episode
                mask = pa.compute.equal(full_table['episode_index'], pa.scalar(episode_idx))
                episode_table = full_table.filter(mask)
                
                if frame_idx >= episode_table.num_rows:
                    return jsonify({"success": False, "error": f"Frame {frame_idx} 不存在"})
                
                # Remove the frame
                all_indices = np.arange(episode_table.num_rows)
                keep_indices = np.concatenate([all_indices[:frame_idx], all_indices[frame_idx+1:]])
                
                if len(keep_indices) == 0:
                    return jsonify({"success": False, "error": "不能删除episode的最后一帧"})
                
                cleaned_episode = episode_table.take(pa.array(keep_indices, type=pa.int64()))

                # Keep original timestamp and frame_index unchanged for video sync
                # This ensures proper alignment without needing to update episode metadata

                # Combine cleaned episode with other episodes
                other_mask = pa.compute.not_equal(full_table['episode_index'], pa.scalar(episode_idx))
                other_episodes = full_table.filter(other_mask)

                if other_episodes.num_rows > 0:
                    combined_table = pa.concat_tables([other_episodes, cleaned_episode])
                else:
                    combined_table = cleaned_episode

                # Save the modified file
                pq.write_table(combined_table, episode_path)
                
            else:
                # For v2.x, process single episode file
                table = loader_obj.load_episode(episode_idx)
                
                if frame_idx >= table.num_rows:
                    return jsonify({"success": False, "error": f"Frame {frame_idx} 不存在"})
                
                # Remove the frame
                all_indices = np.arange(table.num_rows)
                keep_indices = np.concatenate([all_indices[:frame_idx], all_indices[frame_idx+1:]])
                
                if len(keep_indices) == 0:
                    return jsonify({"success": False, "error": "不能删除episode的最后一帧"})
                
                cleaned_table = table.take(pa.array(keep_indices, type=pa.int64()))

                # Keep original timestamp and frame_index unchanged for video sync
                # This ensures proper alignment without needing to update episode metadata
                
                # Save the modified episode
                episode_path = loader_obj.get_episode_path(episode_idx)
                pq.write_table(cleaned_table, episode_path)
            
            # Update episode metadata
            loader_obj.episodes[episode_idx]['length'] = len(keep_indices)
            
            # Save updated metadata
            import json
            episodes_jsonl = loader_obj.dataset_root / "meta" / "episodes.jsonl"
            with open(episodes_jsonl, 'w') as f:
                for ep in loader_obj.episodes:
                    f.write(json.dumps(ep) + '\n')
            
            # Update info.json total_frames
            info_path = loader_obj.dataset_root / "meta" / "info.json"
            with open(info_path, 'r') as f:
                info = json.load(f)
            info['total_frames'] = info.get('total_frames', 0) - 1
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            # Reload loader to reflect changes
            nonlocal loader
            loader = LeRobotDatasetLoader(loader.dataset_root)
            
            return jsonify({
                "success": True, 
                "remaining_frames": len(keep_indices),
                "message": f"已删除帧 {frame_idx}"
            })
            
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    @app.route("/api/delete_frame_range/<int:episode_idx>/<int:start_frame>/<int:end_frame>", methods=["POST"])
    def delete_frame_range(episode_idx: int, start_frame: int, end_frame: int):
        """Delete a range of frames from an episode."""
        try:
            loader_obj = get_loader()
            
            if episode_idx >= len(loader_obj.episodes):
                return jsonify({"success": False, "error": f"Episode {episode_idx} 不存在"})

            if start_frame > end_frame:
                return jsonify({"success": False, "error": "起始帧必须小于等于结束帧"})

            import pyarrow as pa
            import pyarrow.parquet as pq
            import numpy as np
            
            # For v3.0, need to handle multiple episodes in one file
            if loader_obj.version == "v3.0":
                # Get the data file path
                episode_path = loader_obj.get_episode_path(episode_idx)
                
                # Load the full data file
                full_table = pq.read_table(episode_path)
                
                # Filter to get current episode
                mask = pa.compute.equal(full_table['episode_index'], pa.scalar(episode_idx))
                episode_table = full_table.filter(mask)
                
                if end_frame >= episode_table.num_rows:
                    return jsonify({"success": False, "error": f"结束帧 {end_frame} 不存在"})
                
                # Remove the frame range
                all_indices = np.arange(episode_table.num_rows)
                keep_indices = np.concatenate([all_indices[:start_frame], all_indices[end_frame+1:]])
                
                if len(keep_indices) == 0:
                    return jsonify({"success": False, "error": "不能删除episode的所有帧"})
                
                cleaned_episode = episode_table.take(pa.array(keep_indices, type=pa.int64()))

                # Keep original timestamp and frame_index unchanged for video sync
                # This ensures proper alignment without needing to update episode metadata

                # Combine cleaned episode with other episodes
                other_mask = pa.compute.not_equal(full_table['episode_index'], pa.scalar(episode_idx))
                other_episodes = full_table.filter(other_mask)

                if other_episodes.num_rows > 0:
                    combined_table = pa.concat_tables([other_episodes, cleaned_episode])
                else:
                    combined_table = cleaned_episode

                # Save the modified file
                pq.write_table(combined_table, episode_path)
                
            else:
                # For v2.x, process single episode file
                table = loader_obj.load_episode(episode_idx)
                
                if end_frame >= table.num_rows:
                    return jsonify({"success": False, "error": f"结束帧 {end_frame} 不存在"})
                
                # Remove the frame range
                all_indices = np.arange(table.num_rows)
                keep_indices = np.concatenate([all_indices[:start_frame], all_indices[end_frame+1:]])
                
                if len(keep_indices) == 0:
                    return jsonify({"success": False, "error": "不能删除episode的所有帧"})
                
                cleaned_table = table.take(pa.array(keep_indices, type=pa.int64()))

                # Keep original timestamp and frame_index unchanged for video sync
                # This ensures proper alignment without needing to update episode metadata
                
                # Save the modified episode
                episode_path = loader_obj.get_episode_path(episode_idx)
                pq.write_table(cleaned_table, episode_path)
            
            # Update episode metadata
            deleted_count = end_frame - start_frame + 1
            loader_obj.episodes[episode_idx]['length'] = len(keep_indices)
            
            # Save updated metadata
            import json
            episodes_jsonl = loader_obj.dataset_root / "meta" / "episodes.jsonl"
            with open(episodes_jsonl, 'w') as f:
                for ep in loader_obj.episodes:
                    f.write(json.dumps(ep) + '\n')
            
            # Update info.json total_frames
            info_path = loader_obj.dataset_root / "meta" / "info.json"
            with open(info_path, 'r') as f:
                info = json.load(f)
            info['total_frames'] = info.get('total_frames', 0) - deleted_count
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            # Reload loader to reflect changes
            nonlocal loader
            loader = LeRobotDatasetLoader(loader.dataset_root)
            
            return jsonify({
                "success": True, 
                "remaining_frames": len(keep_indices),
                "deleted_frames": deleted_count,
                "message": f"已删除帧 {start_frame} 到 {end_frame}，共 {deleted_count} 帧"
            })
            
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    @app.route("/api/joint_angles/<int:episode_idx>")
    def get_joint_angles(episode_idx: int):
        """Get joint angles data for visualization."""
        try:
            loader_obj = get_loader()
            
            if episode_idx >= len(loader_obj.episodes):
                return jsonify({"success": False, "error": f"Episode {episode_idx} 不存在"})
            
            import numpy as np
            import pyarrow as pa
            
            table = loader_obj.load_episode(episode_idx)
            
            # Try to find joint angle columns
            joint_data = {}
            
            # Common column names for joint angles
            possible_columns = [
                'observation.state',
                'action',
                'joint_positions',
                'robot_state',
                'arm_joint'
            ]
            
            # Find columns containing joint data
            for col_name in table.column_names:
                # Check if column contains joint angles (usually float array)
                if any(key in col_name.lower() for key in ['joint', 'state', 'action', 'position']):
                    col_data = table[col_name].to_pylist()
                    
                    # Determine if it's left/right arm based on column name
                    arm_type = 'unknown'
                    if 'left' in col_name.lower():
                        arm_type = 'left'
                    elif 'right' in col_name.lower():
                        arm_type = 'right'
                    
                    joint_data[col_name] = {
                        'values': col_data,
                        'arm_type': arm_type
                    }
            
            # If no joint columns found, return empty
            if not joint_data:
                return jsonify({
                    "success": True, 
                    "joint_angles": {},
                    "message": "没有找到关节角度数据列"
                })
            
            return jsonify({
                "success": True,
                "joint_angles": joint_data,
                "total_frames": table.num_rows
            })
            
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

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