@echo off
REM RoboClean 环境安装脚本 (Windows)

echo ================================
echo RoboClean 环境安装脚本
echo ================================
echo.

REM 检查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    exit /b 1
)

echo ✓ Python已安装
echo.

REM 创建虚拟环境
if not exist "venv" (
    echo 创建虚拟环境...
    python -m venv venv
    echo ✓ 虚拟环境创建成功
) else (
    echo ✓ 虚拟环境已存在
)
echo.

REM 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM 升级pip
echo.
echo 升级pip...
python -m pip install --upgrade pip

REM 安装依赖
echo.
echo 安装依赖包...
pip install -r requirements.txt

echo ✓ 依赖安装完成
echo.

REM 安装项目包
echo 安装RoboClean包...
pip install -e .

echo ✓ RoboClean安装完成
echo.

REM 验证安装
echo 验证安装...
python -c "import roboclean; print('✓ RoboClean导入成功')"

echo.
echo ================================
echo 安装完成！
echo ================================
echo.
echo 使用方法：
echo 1. 激活环境: venv\Scripts\activate.bat
echo 2. 查看帮助: roboclean --help
echo 3. 快速开始: python quick_start.py
echo.
echo 或者使用Python模块方式：
echo   python -m roboclean.cli.main --help
echo.

pause