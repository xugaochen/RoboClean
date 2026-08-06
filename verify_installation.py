#!/usr/bin/env python3
"""验证RoboClean安装和功能的完整性测试脚本"""

import sys
from pathlib import Path


def check_dependencies():
    """检查依赖包"""
    print("=" * 60)
    print("检查依赖包")
    print("=" * 60)

    dependencies = {
        "numpy": "numpy",
        "pyarrow": "pyarrow",
        "pandas": "pandas",
        "tqdm": "tqdm",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
        "PIL": "PIL",
        "rich": "rich",
        "cv2": "opencv-python",
        "av": "av",
    }

    all_ok = True
    for import_name, display_name in dependencies.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {display_name:15s}: {version}")
        except ImportError:
            print(f"❌ {display_name:15s}: NOT INSTALLED")
            if import_name in ["numpy", "pyarrow", "pandas", "tqdm", "matplotlib"]:
                all_ok = False

    return all_ok


def test_core_functionality():
    """测试核心功能"""
    print("\n" + "=" * 60)
    print("测试核心功能")
    print("=" * 60)

    try:
        from roboclean.core.loader import LeRobotDatasetLoader
        from roboclean.core.cleaner import DataCleaner

        print("✅ 核心模块导入成功")

        # 测试数据加载
        dataset_path = Path("demo_data/empty_table_reset/empty_table_reset_lerobot_v2.0")
        if not dataset_path.exists():
            print("❌ 测试数据集不存在")
            return False

        loader = LeRobotDatasetLoader(dataset_path)
        print(f"✅ 数据加载成功: {loader.version}, {loader.info.get('total_episodes')} episodes")

        # 测试清洗功能
        cleaner = DataCleaner(loader)
        table = loader.load_episode(0)
        motion_columns = loader.get_motion_columns()
        keep_mask, scores = cleaner.build_keep_mask(table, motion_columns=motion_columns)
        print(f"✅ 清洗功能正常: episode 0, {len(keep_mask)} frames")

        return True

    except Exception as e:
        print(f"❌ 核心功能测试失败: {e}")
        return False


def test_visualization():
    """测试可视化功能"""
    print("\n" + "=" * 60)
    print("测试可视化功能")
    print("=" * 60)

    try:
        from roboclean.viz.visualizer import DatasetVisualizer
        from roboclean.core.loader import LeRobotDatasetLoader

        print("✅ 可视化模块导入成功")

        dataset_path = Path("demo_data/empty_table_reset/empty_table_reset_lerobot_v2.0")
        loader = LeRobotDatasetLoader(dataset_path)
        viz = DatasetVisualizer(loader)

        # 测试图表生成
        fig = viz.plot_episode_length_distribution()
        print("✅ 基础可视化功能正常")

        return True

    except Exception as e:
        print(f"❌ 可视化功能测试失败: {e}")
        return False


def test_cli():
    """测试命令行工具"""
    print("\n" + "=" * 60)
    print("测试命令行工具")
    print("=" * 60)

    try:
        from roboclean.cli.main import main
        import subprocess

        # 测试help命令
        result = subprocess.run(
            ["python3", "-m", "roboclean.cli.main", "--help"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ CLI工具正常工作")
            return True
        else:
            print(f"❌ CLI工具异常: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ CLI测试失败: {e}")
        return False


def test_different_datasets():
    """测试不同数据集"""
    print("\n" + "=" * 60)
    print("测试数据集兼容性")
    print("=" * 60)

    datasets = [
        "demo_data/empty_table_reset/empty_table_reset_lerobot_v2.0",
        "demo_data/empty_table_reset/empty_table_reset_lerobot_v30",
        "demo_data/pick_from_shelf_0723",
    ]

    all_ok = True
    for dataset_path in datasets:
        try:
            from roboclean.core.loader import LeRobotDatasetLoader

            loader = LeRobotDatasetLoader(dataset_path)
            motion_columns = loader.get_motion_columns()
            print(f"✅ {Path(dataset_path).name:40s}: {loader.version}, {len(motion_columns)} motion cols")
        except Exception as e:
            print(f"❌ {Path(dataset_path).name:40s}: {e}")
            all_ok = False

    return all_ok


def main():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("RoboClean 完整性验证测试")
    print("🧪" * 30 + "\n")

    results = {
        "依赖检查": check_dependencies(),
        "核心功能": test_core_functionality(),
        "可视化": test_visualization(),
        "CLI工具": test_cli(),
        "数据兼容": test_different_datasets(),
    }

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！RoboClean已就绪，可以立即使用！")
        print("\n快速开始:")
        print("  python3 -m roboclean.cli.main info demo_data/pick_from_shelf_0723")
        print("  python3 -m roboclean.cli.main clean demo_data/pick_from_shelf_0723 --output-path cleaned")
        print("  python3 -m roboclean.cli.main viz demo_data/pick_from_shelf_0723 --report")
    else:
        print("⚠️  部分测试未通过，请检查错误信息")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()