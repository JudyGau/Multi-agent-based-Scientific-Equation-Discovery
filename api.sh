# ⚠️ 遗留脚本（已失效，请勿使用）
# ---------------------------------------------------------------
# 这是早期服务器上的并发启动脚本，与当前 CLI 已不兼容：
#   - 使用 `--spec_path` 参数，而 main.py 已改为 `--data_csv`（必填）；
#   - 硬编码个人 conda 环境（mastsr）与绝对路径（/data1/wangboxiao/DrSR-api）；
#   - 15 个相同的后台任务，无并发/资源控制。
# 请改用根目录 `example.sh`（可用 LLM_CONFIG / PYTHON 环境变量配置）。
# ---------------------------------------------------------------
conda activate mastsr
cd /data1/wangboxiao/DrSR-api

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &

nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &
nohup python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt >/dev/null 2>&1 &
