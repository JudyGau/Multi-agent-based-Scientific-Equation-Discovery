@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "SELF=%~nx0"

rem ===========================================================================
rem DrSR 批量运行示例（Windows 批处理版，与 example.sh 等价）：
rem 对 data 目录下的基准数据集逐个执行方程发现。
rem
rem 用法：
rem   example.bat                              直接双击，用默认档案与 Python
rem   set LLM_CONFIG=config/deepseek_deepseek-v4-flash.config 后再运行 example.bat
rem   set PYTHON=C:\Python312\python.exe 后再运行 example.bat
rem
rem 依赖：需存在 LLM 档案（默认 config/glm_glm-5.3-flash.config），且 api_key 已填或
rem 对应环境变量已设置（如 ZHIPU_API_KEY / DEEPSEEK_API_KEY）。
rem 首次使用：copy config\glm_glm-5.3-flash.config.example config\glm_glm-5.3-flash.config
rem 角色到档案的绑定声明在 config\agents.config.json；自检：
rem   python -m drsr_420.llm.roles --check
rem
rem 入口为 python -m drsr_420.cli.main（本脚本会先切到仓库根目录）；等价于安装后的
rem drsr420 命令，以及 IDE 里 4 个 MRF* 运行配置（同样以模块方式启动）。
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

set /a TOTAL=0
set /a FAILED=0

rem ---------------------------------------------------------------------------
rem 每个问题两行：先 set BACKGROUND，再 call run_problem 问题名 train.csv 路径。
rem
rem background 之所以走变量、不作为 call 的参数：cmd 的 call 会对参数做第二轮
rem 解析，把参数里的脱字符（caret）翻倍——1 个变 2 个、2 个变 4 个；而 echo 又会
rem 把它显示回 1 个，所以错误只在真正传给 Python 时才暴露。下面 MRFCompress-3 的
rem background 含 LaTeX 的 ^{...}，作为 call 参数必然被写坏，走变量则逐字节无损。
rem ---------------------------------------------------------------------------

set "BACKGROUND=Find the mathematical function skeleton that represents Population growth rate, given data on Time, and Population at time t."
call :run_problem BPG0 "data/BPG0/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents Rate of change of concentration in chemistry reaction kinetics, given data on Time, and Concentration at time t."
call :run_problem CRK0 "data/CRK0/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents the intensity of the first wave source, given data on the resultant intensity of two wave sources, the intensity of the second wave source, and the phase difference between the two wave sources."
call :run_problem I.37.4_0_1 "data/I.37.4_0_1/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents the object's velocity, given data on the total energy of an object, the relativistic mass of an object, and the speed of light."
call :run_problem I.48.2_1_0 "data/I.48.2_1_0/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents the distance from the dipole to the point where the electric field is being measured, given data on the electric field, the electric constant or permittivity of the medium, the dipole moment, and the angle between the dipole axis and the position vector."
call :run_problem II.6.15b_3_0 "data/II.6.15b_3_0/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents the temperature of the system, given data on the energy of the nth mode of a quantum harmonic oscillator, the Planck constant, the angular frequency of the oscillator, and the Boltzmann constant."
call :run_problem III.4.33_3_0 "data/III.4.33_3_0/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents Stress, given data on Strain, and Temperature."
call :run_problem MatSci0 "data/MatSci0/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents Acceleration in Nonl-linear Harmonic Oscillator, given data on Position at time t, Time, and Velocity at time t."
call :run_problem PO0 "data/PO0/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents E. Coli bacterial growth rate, given data on population density, substrate concentration, temperature, and pH level."
call :run_problem bactgrow "data/bactgrow/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity."
call :run_problem oscillator1 "data/oscillator1/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on time, position, and velocity."
call :run_problem oscillator2 "data/oscillator2/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents stress, given data on strain and temperature in an Aluminium rod for both elastic and plastic regions."
call :run_problem stressstrain "data/stressstrain/train.csv"

rem ── 磁流变（MRF）：剪切 / 压缩 × 3 构型 ────────────────────────
set "BACKGROUND=Find the mathematical function skeleton that represents magnetorheological effect in shear mode, given data on lambda12(L1/L2), lambda23(L2/L3), and alpha(the parameter that controls the surface curvature of the particles). L1, L2, and L3 are the long axis, medium axis, and short axis of the cuboid bounding of superellipse particle."
call :run_problem MRFShear-3 "data/MRFShear-3/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents the magnetorheological effect of a magnetorheological fluid (MRF) in shear mode. An MRF is a suspension of micron-scale magnetizable particles in a non-magnetizable carrier fluid; under an applied magnetic field the particles magnetize and assemble into chain-like or columnar microstructures along the field direction, which markedly increases the stress the fluid can transmit, and this field-induced strengthening is the magnetorheological effect. The data describe how this effect depends on the shape of the cuboid particle through two axis-length ratios: L1, L2 and L3 denote the long-axis, medium-axis and short-axis lengths of the cuboid particle; lambda12 = L1/L2 is the ratio of the long-axis length to the medium-axis length; lambda23 = L2/L3 is the ratio of the medium-axis length to the short-axis length. Both ratios equal 1 for a shape-isotropic (cubic) particle, and their product lambda12 times lambda23 equals L1/L3, the overall long-to-short aspect ratio of the particle. Find how the measured shear response miu under the magnetorheological effect depends on these two axis-length ratios."
call :run_problem MRFShear-Cuboid "data/MRFShear-Cuboid/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents the magnetorheological effect of a magnetorheological fluid (MRF) in shear mode. An MRF is a suspension of micron-scale magnetizable particles in a non-magnetizable carrier fluid; under an applied magnetic field the particles magnetize and assemble into chain-like or columnar microstructures along the field direction, which markedly increases the stress the fluid can transmit, and this field-induced strengthening is the magnetorheological effect. The data describe how this effect depends on the shape of the ellipsoid particle through one axis-length ratio: L1 and L2 denote the long-axis and short-axis lengths of the ellipsoid particle, and lambda12 = L1/L2 is the ratio of the long-axis length to the short-axis length, that is, the elongation of the particle relative to a sphere, with lambda12 = 1 for a spherical particle. Find how the measured shear response miu under the magnetorheological effect depends on this axis-length ratio of the particle."
call :run_problem MRFShear-Ellipsoid "data/MRFShear-Ellipsoid/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents magnetorheological effect in compress mode, given data on lambda12(L1/L2), lambda23(L2/L3), and alpha(the parameter that controls the surface curvature of the particles). L1, L2, and L3 are the long axis, medium axis, and short axis of the cuboid bounding of superellipse particle. The superellipsoid equation was selected as the construction constraint equation: \left(\frac{x}{L_1}\right)^{\frac{2}{\alpha}} + \left(\frac{y}{L_2}\right)^{\frac{2}{\alpha}} + \left(\frac{z}{L_3}\right)^{\frac{2}{\alpha}} = 1"
call :run_problem MRFCompress-3 "data/MRFCompress-3/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents the magnetorheological effect of a magnetorheological fluid (MRF) in compress mode. An MRF is a suspension of micron-scale magnetizable particles in a non-magnetizable carrier fluid; under an applied magnetic field the particles magnetize and assemble into chain-like or columnar microstructures along the field direction, which markedly increases the stress the fluid can transmit, and this field-induced strengthening is the magnetorheological effect. The data describe how this effect depends on the shape of the cuboid particle through two axis-length ratios: L1, L2 and L3 denote the long-axis, medium-axis and short-axis lengths of the cuboid particle; lambda12 = L1/L2 is the ratio of the long-axis length to the medium-axis length; lambda23 = L2/L3 is the ratio of the medium-axis length to the short-axis length. Both ratios equal 1 for a shape-isotropic (cubic) particle, and their product lambda12 times lambda23 equals L1/L3, the overall long-to-short aspect ratio of the particle. Find how the measured stress sigma under the magnetorheological effect depends on these two axis-length ratios."
call :run_problem MRFCompress-Cuboid "data/MRFCompress-Cuboid/train.csv"

set "BACKGROUND=Find the mathematical function skeleton that represents the magnetorheological effect of a magnetorheological fluid (MRF) in compress mode. An MRF is a suspension of micron-scale magnetizable particles in a non-magnetizable carrier fluid; under an applied magnetic field the particles magnetize and assemble into chain-like or columnar microstructures along the field direction, which markedly increases the stress the fluid can transmit, and this field-induced strengthening is the magnetorheological effect. The data describe how this effect depends on the shape of the ellipsoid particle through one axis-length ratio: L1 and L2 denote the long-axis and short-axis lengths of the ellipsoid particle, and lambda12 = L1/L2 is the ratio of the long-axis length to the short-axis length, that is, the elongation of the particle relative to a sphere, with lambda12 = 1 for a spherical particle. Find how the measured stress sigma under the magnetorheological effect depends on this axis-length ratio of the particle."
call :run_problem MRFCompress-Ellipsoid "data/MRFCompress-Ellipsoid/train.csv"

rem ── 汇总 ──────────────────────────────────────────────────────
echo ================================================================
echo 完成：共 %TOTAL% 个问题，失败/跳过 %FAILED% 个。
echo ================================================================
set "RC=0"
if %FAILED% GTR 0 set "RC=1"
call :maybe_pause
exit /b %RC%


rem ===========================================================================
rem 子过程
rem ===========================================================================

rem run_problem：参数1=问题名，参数2=train.csv 路径；background 读全局变量 BACKGROUND
:run_problem
set "NAME=%~1"
set "CSV=%~2"
set /a TOTAL+=1
if not exist "%CSV%" (
  echo [SKIP] %NAME%：数据文件不存在 - "%CSV%"
  set /a FAILED+=1
  exit /b 0
)
echo ================================================================
echo === [%TOTAL%] %NAME%
echo ================================================================
"%PYTHON%" -m drsr_420.cli.main --problem_name "%NAME%" --data_csv "%CSV%" --llm_config "%LLM_CONFIG%" --background "%BACKGROUND%"
if errorlevel 1 goto :run_problem_failed
echo [OK] %NAME%
exit /b 0

:run_problem_failed
echo [FAIL] %NAME%（退出码 %errorlevel%）
set /a FAILED+=1
exit /b 0


rem maybe_pause：双击运行时暂停，避免窗口一闪而过；从已有命令行调用则不暂停，
rem 便于把本脚本串进别的脚本里。
:maybe_pause
echo %cmdcmdline% | find /i "%SELF%" >nul
if not errorlevel 1 pause
exit /b 0
