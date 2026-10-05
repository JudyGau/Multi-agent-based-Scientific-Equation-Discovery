"""收尾分析与报告装配层：把一次实验的产物整理成可读、可复现的报告。

依赖：``core`` / ``equations`` / ``llm`` / ``knowledge``。不依赖 ``agents`` /
``runtime``（编排在其上）。

子包：
    pruning        敏感度剪枝与剪枝判定
    generalization 泛化口径（held-out / 留一交叉验证 LOO）

模块：
    find_best_eq   收尾编排（参数拟合 → 剪枝 → 解释 → 可视化）
    explain        物理解释（LLM）与报告装配
    report_sections report.md 各机器小节
    references     参考文献检索 / 去重 / 渲染
    md_sections    markdown 小节的通用读写
    data_io        结果目录的 CSV / JSON 读取
    progress_curve 训练进度曲线
    curves         数据 / 拟合曲线
    viz            表达式树可视化
"""