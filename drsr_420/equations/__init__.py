"""公式领域模型层：把「方程」本身当作一等公民（本项目的通用语言）。

存放**只跟公式 / 样本字符串与数值打交道**的纯模块：解析、求值、文本代数、
样本记录读取、数值病理内核。它们不读实验目录结构、不调 LLM、不做编排，
因此可以被 agents / evidence / reporting 任意共享，也是「自变量 / 因变量 /
分数分解」这套词汇的唯一来源。

放在 equations 而不是 core：这些是**领域概念**而非领域无关基础设施；
放在这里而不是 reporting：采样闭环（agents）与收尾（reporting）都要用它。

依赖：仅 ``core``。

模块：
    header        样本头部（Dependent / Independents）解析
    records       本次实验样本记录的读取与分数分解
    pathology     拟合后动态范围体检（数值病理三判据）——评分准则的数值内核
    text_algebra  样本方程的文本代数（记号化 / 架构指纹 / 代表元编译）
    parse         表达式解析（LLM 输出 → sympy）
    numeric       表达式数值化工具
    evaluator     表达式求值内核
"""