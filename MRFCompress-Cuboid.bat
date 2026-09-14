@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "SELF=%~nx0"

rem ===========================================================================
rem DrSR 单问题运行示例（Windows 批处理版，与 MRFCompress-Cuboid.sh 等价）：
rem 对 MRFCompress-Cuboid（压缩模式，8 个样本）执行一次方程发现。
rem
rem 用法：
rem   MRFCompress-Cuboid.bat                   直接双击，用默认档案与 Python
rem   set LLM_CONFIG=config/deepseek_deepseek-v4-flash.config 后再运行本脚本
rem   set PYTHON=C:\Python312\python.exe 后再运行本脚本
rem
rem 依赖：需存在 LLM 档案（默认 config/glm_glm-5.3-flash.config），且 api_key 已填或
rem 对应环境变量已设置（如 ZHIPU_API_KEY / DEEPSEEK_API_KEY）。
rem 首次使用：copy config\glm_glm-5.3-flash.config.example config\glm_glm-5.3-flash.config
rem 入口为 python -m drsr_420.cli.main（本脚本会先切到仓库根目录）。
rem 参数与入库的 IDE 运行配置 .idea/runConfigurations/ 下的同名 XML 一致。
rem ===========================================================================

rem ── Python 解释器：PYTHON 环境变量 优先，其次仓库内 .venv2，最后 PATH 上的 python
if not defined PYTHON if exist ".venv2\Scripts\python.exe" set "PYTHON=.venv2\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python"
set "PYTHON_OK="
where "%PYTHON%" >nul 2>&1 && set "PYTHON_OK=1"
if not defined PYTHON_OK if exist "%PYTHON%" set "PYTHON_OK=1"
if not defined PYTHON_OK (
  echo [ERROR] 找不到可执行的 Python：%PYTHON%（可用 PYTHON 环境变量指定）
  call :maybe_pause
  exit /b 1
)

rem ── LLM 档案：LLM_CONFIG 环境变量 优先，默认 glm-5.3-flash
if not defined LLM_CONFIG set "LLM_CONFIG=config/glm_glm-5.3-flash.config"
if not exist "%LLM_CONFIG%" (
  echo [ERROR] 未找到 LLM 档案：%LLM_CONFIG%
  echo         请从 config 目录下的 .example 模板复制一份并填入 api_key，或用 LLM_CONFIG 指定。
  call :maybe_pause
  exit /b 1
)

echo ================================================================
echo === MRFCompress-Cuboid
echo ================================================================
"%PYTHON%" -m drsr_420.cli.main ^
  --problem_name MRFCompress-Cuboid ^
  --data_csv data/MRFCompress-Cuboid/train.csv ^
  --num_samplers 4 ^
  --llm_config "%LLM_CONFIG%" ^
  --niterations 6 ^
  --samples_per_iteration 4 ^
  --background "Find the mathematical function skeleton that represents the magnetorheological effect of a magnetorheological fluid (MRF) in compress mode. An MRF is a suspension of micron-scale magnetizable particles in a non-magnetizable carrier fluid; under an applied magnetic field the particles magnetize and assemble into chain-like or columnar microstructures along the field direction, which markedly increases the stress the fluid can transmit, and this field-induced strengthening is the magnetorheological effect. The data describe how this effect depends on the shape of the cuboid particle through two axis-length ratios: L1, L2 and L3 denote the long-axis, medium-axis and short-axis lengths of the cuboid particle; lambda12 = L1/L2 is the ratio of the long-axis length to the medium-axis length; lambda23 = L2/L3 is the ratio of the medium-axis length to the short-axis length. Both ratios equal 1 for a shape-isotropic (cubic) particle, and their product lambda12 times lambda23 equals L1/L3, the overall long-to-short aspect ratio of the particle. Find how the measured stress sigma under the magnetorheological effect depends on these two axis-length ratios."
set "RC=%errorlevel%"
if not "%RC%"=="0" echo [FAIL] MRFCompress-Cuboid（退出码 %RC%）
if "%RC%"=="0" echo [OK] MRFCompress-Cuboid
call :maybe_pause
exit /b %RC%


rem ===========================================================================
rem 子过程
rem ===========================================================================

rem maybe_pause：双击运行时暂停，避免窗口一闪而过；从已有命令行调用则不暂停，
rem 便于把本脚本串进别的脚本里。
:maybe_pause
echo %cmdcmdline% | find /i "%SELF%" >nul
if not errorlevel 1 pause
exit /b 0
