"""论文实验设施层：把「多次 run」变成「一张表」的确定性计算。

与 ``reporting`` 的分工：``reporting`` 处理**单次** run 的收尾报告（选解 / 剪枝 /
物理解释）；``harness`` 面向**跨 run** 的实验汇总（论文 E1 主表的
SA / Acc@0.1 / NMSE 等列）。本层只做指标定义与聚合，纯计算、不重新拟合、不调用
LLM、不引入新依赖。

依赖：可用下层任意层。

模块：
    metrics    指标定义（MSE / NMSE / Acc@阈值 / 符号等价 SA）
    aggregate  跨 run 汇总（读 ``experiments/`` 下各 run 的样本与快照 → 表格）
"""