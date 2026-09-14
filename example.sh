#!/usr/bin/env bash
# DrSR 批量运行示例：对 data/ 下的基准数据集逐个执行方程发现。
#
# 用法：
#   bash example.sh                      # 使用默认配置与 python
#   LLM_CONFIG=config/deepseek_deepseek-v4-flash.config bash example.sh
#   PYTHON=.venv2/Scripts/python.exe bash example.sh   # Windows venv
#
# 依赖：需存在 LLM 档案（默认 config/glm_glm-5.3-flash.config），且 api_key 已填或
# 对应环境变量已设置（如 ZHIPU_API_KEY / DEEPSEEK_API_KEY）。
# 首次使用：cp config/glm_glm-5.3-flash.config.example config/glm_glm-5.3-flash.config
# 角色 → 档案 的绑定声明在 config/agents.config.json；自检：
#   python -m drsr_420.llm.roles --check
#
# 入口为 `python -m drsr_420.cli.main`（需在仓库根目录执行）；等价于安装后的
# `drsr420` 命令，以及 IDE 里 4 个 MRF* 运行配置（同样以模块方式启动）。
set -u

PYTHON="${PYTHON:-python3}"
LLM_CONFIG="${LLM_CONFIG:-config/glm_glm-5.3-flash.config}"

if ! command -v "$PYTHON" >/dev/null 2>&1 && [ ! -x "$PYTHON" ]; then
  echo "[ERROR] 找不到可执行的 Python：$PYTHON（可用 PYTHON 环境变量指定）" >&2
  exit 1
fi

if [ ! -f "$LLM_CONFIG" ]; then
  echo "[ERROR] 未找到 LLM 档案：$LLM_CONFIG" >&2
  echo "        请从 config/ 下的 .example 模板复制一份并填入 api_key，或用 LLM_CONFIG 指定。" >&2
  exit 1
fi

TOTAL=0
FAILED=0

# run_problem <problem_name> <train.csv 路径> <background>
run_problem() {
  local name="$1" csv="$2" background="$3"
  TOTAL=$((TOTAL + 1))
  if [ ! -f "$csv" ]; then
    echo "[SKIP] $name：数据文件不存在 -> $csv" >&2
    FAILED=$((FAILED + 1))
    return
  fi
  echo "================================================================"
  echo "=== [$TOTAL] $name"
  echo "================================================================"
  if "$PYTHON" -m drsr_420.cli.main \
      --problem_name "$name" \
      --data_csv "$csv" \
      --llm_config "$LLM_CONFIG" \
      --background "$background"; then
    echo "[OK] $name"
  else
    echo "[FAIL] $name（退出码 $?）" >&2
    FAILED=$((FAILED + 1))
  fi
}

# ── 物理 / 力学 ────────────────────────────────────────────────
run_problem BPG0 ./data/BPG0/train.csv \
  'Find the mathematical function skeleton that represents Population growth rate, given data on Time, and Population at time t.'

run_problem CRK0 ./data/CRK0/train.csv \
  'Find the mathematical function skeleton that represents Rate of change of concentration in chemistry reaction kinetics, given data on Time, and Concentration at time t.'

run_problem I.37.4_0_1 ./data/I.37.4_0_1/train.csv \
  'Find the mathematical function skeleton that represents the intensity of the first wave source, given data on the resultant intensity of two wave sources, the intensity of the second wave source, and the phase difference between the two wave sources.'

run_problem I.48.2_1_0 ./data/I.48.2_1_0/train.csv \
  "Find the mathematical function skeleton that represents the object's velocity, given data on the total energy of an object, the relativistic mass of an object, and the speed of light."

run_problem II.6.15b_3_0 ./data/II.6.15b_3_0/train.csv \
  'Find the mathematical function skeleton that represents the distance from the dipole to the point where the electric field is being measured, given data on the electric field, the electric constant or permittivity of the medium, the dipole moment, and the angle between the dipole axis and the position vector.'

run_problem III.4.33_3_0 ./data/III.4.33_3_0/train.csv \
  'Find the mathematical function skeleton that represents the temperature of the system, given data on the energy of the nth mode of a quantum harmonic oscillator, the Planck constant, the angular frequency of the oscillator, and the Boltzmann constant.'

run_problem MatSci0 ./data/MatSci0/train.csv \
  'Find the mathematical function skeleton that represents Stress, given data on Strain, and Temperature.'

run_problem PO0 ./data/PO0/train.csv \
  'Find the mathematical function skeleton that represents Acceleration in Nonl-linear Harmonic Oscillator, given data on Position at time t, Time, and Velocity at time t.'

run_problem bactgrow ./data/bactgrow/train.csv \
  'Find the mathematical function skeleton that represents E. Coli bacterial growth rate, given data on population density, substrate concentration, temperature, and pH level.'

run_problem oscillator1 ./data/oscillator1/train.csv \
  'Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity.'

run_problem oscillator2 ./data/oscillator2/train.csv \
  'Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on time, position, and velocity.'

run_problem stressstrain ./data/stressstrain/train.csv \
  'Find the mathematical function skeleton that represents stress, given data on strain and temperature in an Aluminium rod for both elastic and plastic regions.'

# ── 磁流变（MRF）：剪切 / 压缩 × 3 构型 ────────────────────────
run_problem MRFShear-3 ./data/MRFShear-3/train.csv \
  'Find the mathematical function skeleton that represents magnetorheological effect in shear mode, given data on lambda12(L1/L2), lambda23(L2/L3), and alpha(the parameter that controls the surface curvature of the particles). L1, L2, and L3 are the long axis, medium axis, and short axis of the cuboid bounding of superellipse particle.'

run_problem MRFShear-Cuboid ./data/MRFShear-Cuboid/train.csv \
  'Find the mathematical function skeleton that represents the magnetorheological effect of a magnetorheological fluid (MRF) in shear mode. An MRF is a suspension of micron-scale magnetizable particles in a non-magnetizable carrier fluid; under an applied magnetic field the particles magnetize and assemble into chain-like or columnar microstructures along the field direction, which markedly increases the stress the fluid can transmit, and this field-induced strengthening is the magnetorheological effect. The data describe how this effect depends on the shape of the cuboid particle through two axis-length ratios: L1, L2 and L3 denote the long-axis, medium-axis and short-axis lengths of the cuboid particle; lambda12 = L1/L2 is the ratio of the long-axis length to the medium-axis length; lambda23 = L2/L3 is the ratio of the medium-axis length to the short-axis length. Both ratios equal 1 for a shape-isotropic (cubic) particle, and their product lambda12 times lambda23 equals L1/L3, the overall long-to-short aspect ratio of the particle. Find how the measured shear response miu under the magnetorheological effect depends on these two axis-length ratios.'

run_problem MRFShear-Ellipsoid ./data/MRFShear-Ellipsoid/train.csv \
  'Find the mathematical function skeleton that represents the magnetorheological effect of a magnetorheological fluid (MRF) in shear mode. An MRF is a suspension of micron-scale magnetizable particles in a non-magnetizable carrier fluid; under an applied magnetic field the particles magnetize and assemble into chain-like or columnar microstructures along the field direction, which markedly increases the stress the fluid can transmit, and this field-induced strengthening is the magnetorheological effect. The data describe how this effect depends on the shape of the ellipsoid particle through one axis-length ratio: L1 and L2 denote the long-axis and short-axis lengths of the ellipsoid particle, and lambda12 = L1/L2 is the ratio of the long-axis length to the short-axis length, that is, the elongation of the particle relative to a sphere, with lambda12 = 1 for a spherical particle. Find how the measured shear response miu under the magnetorheological effect depends on this axis-length ratio of the particle.'

run_problem MRFCompress-3 ./data/MRFCompress-3/train.csv \
  "Find the mathematical function skeleton that represents magnetorheological effect in compress mode, given data on lambda12(L1/L2), lambda23(L2/L3), and alpha(the parameter that controls the surface curvature of the particles). L1, L2, and L3 are the long axis, medium axis, and short axis of the cuboid bounding of superellipse particle. The superellipsoid equation was selected as the construction constraint equation: \left(\frac{x}{L_1}\right)^{\frac{2}{\alpha}} + \left(\frac{y}{L_2}\right)^{\frac{2}{\alpha}} + \left(\frac{z}{L_3}\right)^{\frac{2}{\alpha}} = 1"

run_problem MRFCompress-Cuboid ./data/MRFCompress-Cuboid/train.csv \
  'Find the mathematical function skeleton that represents the magnetorheological effect of a magnetorheological fluid (MRF) in compress mode. An MRF is a suspension of micron-scale magnetizable particles in a non-magnetizable carrier fluid; under an applied magnetic field the particles magnetize and assemble into chain-like or columnar microstructures along the field direction, which markedly increases the stress the fluid can transmit, and this field-induced strengthening is the magnetorheological effect. The data describe how this effect depends on the shape of the cuboid particle through two axis-length ratios: L1, L2 and L3 denote the long-axis, medium-axis and short-axis lengths of the cuboid particle; lambda12 = L1/L2 is the ratio of the long-axis length to the medium-axis length; lambda23 = L2/L3 is the ratio of the medium-axis length to the short-axis length. Both ratios equal 1 for a shape-isotropic (cubic) particle, and their product lambda12 times lambda23 equals L1/L3, the overall long-to-short aspect ratio of the particle. Find how the measured stress sigma under the magnetorheological effect depends on these two axis-length ratios.'

run_problem MRFCompress-Ellipsoid ./data/MRFCompress-Ellipsoid/train.csv \
  'Find the mathematical function skeleton that represents the magnetorheological effect of a magnetorheological fluid (MRF) in compress mode. An MRF is a suspension of micron-scale magnetizable particles in a non-magnetizable carrier fluid; under an applied magnetic field the particles magnetize and assemble into chain-like or columnar microstructures along the field direction, which markedly increases the stress the fluid can transmit, and this field-induced strengthening is the magnetorheological effect. The data describe how this effect depends on the shape of the ellipsoid particle through one axis-length ratio: L1 and L2 denote the long-axis and short-axis lengths of the ellipsoid particle, and lambda12 = L1/L2 is the ratio of the long-axis length to the short-axis length, that is, the elongation of the particle relative to a sphere, with lambda12 = 1 for a spherical particle. Find how the measured stress sigma under the magnetorheological effect depends on this axis-length ratio of the particle.'

# ── 汇总 ──────────────────────────────────────────────────────
echo "================================================================"
echo "完成：共 $TOTAL 个问题，失败/跳过 $FAILED 个。"
echo "================================================================"
if [ "$FAILED" -gt 0 ]; then
  exit 1
fi
