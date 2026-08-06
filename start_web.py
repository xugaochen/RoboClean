#!/usr/bin/env python3
"""Quick start script for RoboClean web interface."""

import argparse
from pathlib import Path
from roboclean.web import run_web_interface


def main():
    parser = argparse.ArgumentParser(
        description="Start RoboClean web interface for viewing LeRobot dataset videos"
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",
        help="Path to LeRobot dataset (optional, will be prompted in web interface if not provided)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for external access)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)"
    )

    args = parser.parse_args()

    if args.dataset_path and not args.dataset_path.exists():
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return 1

    print(f"""
╔════════════════════════════════════════════╗
║      RoboClean Web Interface               ║
╚════════════════════════════════════════════╝

Host: {args.host}
Port: {args.port}
""")

    if args.dataset_path:
        print(f"Dataset: {args.dataset_path}")
    else:
        print("将在 Web 界面中输入数据集路径")

    print(f"""
使用说明:
1. 打开浏览器
2. 访问: http://{args.host}:{args.port}
3. 输入数据集路径或直接点击 Episode 查看视频
4. 按 Ctrl+C 停止服务器

启动服务器...
""")

    try:
        run_web_interface(
            dataset_path=str(args.dataset_path) if args.dataset_path else None,
            host=args.host,
            port=args.port
        )
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"\n启动服务器失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())