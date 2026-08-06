#!/bin/bash
# RoboClean 环境安装脚本

set -e

echo "================================"
echo "RoboClean 环境安装脚本"
echo "================================"

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8.0"

echo "检测到Python版本: $PYTHON_VERSION"

# 比较版本
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "错误: Python版本需要 >= $REQUIRED_VERSION"
    echo "当前版本: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python版本符合要求"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo ""
    echo "创建虚拟环境..."
    python3 -m venv venv
    echo "✓ 虚拟环境创建成功"
else
    echo "✓ 虚拟环境已存在"
fi

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo ""
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "安装依赖包..."
pip install -r requirements.txt

echo "✓ 依赖安装完成"

# 安装项目包（开发模式）
echo ""
echo "安装RoboClean包（开发模式）..."
pip install -e .

echo "✓ RoboClean安装完成"

# 验证安装
echo ""
echo "验证安装..."
python3 -c "import roboclean; print('✓ RoboClean导入成功')"

# 显示信息
echo ""
echo "================================"
echo "安装完成！"
echo "================================"
echo ""
echo "使用方法："
echo "1. 激活环境: source venv/bin/activate"
echo "2. 查看帮助: roboclean --help"
echo "3. 快速开始: python quick_start.py"
echo ""
echo "或者使用Python模块方式："
echo "  python -m roboclean.cli.main --help"
echo ""