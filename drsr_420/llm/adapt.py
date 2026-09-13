"""请求体方言适配：把跨提供商的语义参数翻译成各家合法的请求字段。

为什么单独成文件
================
``client.py`` 回答"**怎么发**"（重试、流式、记账），本模块回答"**发给谁时字段长什么样**"。
方言是用户可声明的东西（档案里的 ``dialect`` 字段），两者的变化原因彻底分开：
加一个方言分支不该牵动传输层代码。

方言（dialect）
===============
:data:`DIALECTS` 是档案 ``dialect`` 字段的合法取值，按"**思考强度怎么表达**"划分：

==========  ==================================================================
``glm``     智谱 v4：``thinking={'type':'enabled'}`` 时 ``reasoning_effort`` 生效；
            输出上限字段名是 ``max_tokens``（``max_completion_tokens`` 会被改名）
``deepseek`` ``reasoning_effort`` 直通（low/medium/high）
``ollama``  用 ``think`` 布尔控制思考，不认 ``reasoning_effort``
``openai``  纯 OpenAI 兼容：不认识的私有参数一律不发明（**默认方言**）
==========  ==================================================================

方言的**唯一来源是声明**，不按 provider 名推断
==============================================
``resolve_dialect`` 只回答"声明了什么、没声明就用谁给的默认值"。曾经的规则是
"provider 名恰好等于某个方言名就用它"，那让方言变成**接入方的名字副作用**：

* 别名用户（``zhipu``/``bigmodel``/``glm4``）一度因此**静默丢失**按角色注入的
  ``reasoning_effort``——名字没撞上方言名，分支就跳过了，且毫无信号；
* 自定义提供商的 provider 段（``ustc``）永远撞不上，只能靠显式声明。

现在两层都显式：内置提供商的默认方言写在 ``factory._PROVIDER_SPECS`` 的规格表里
（一行一家，看得见），档案的 ``dialect`` 字段覆盖它；自定义提供商不声明即
``openai``。加一个已知家族的端点，写 ``"dialect": "deepseek"`` 即可复用该分支。
"""
from __future__ import annotations

#: 档案 ``dialect`` 字段的合法取值（顺序即文档顺序）。
DIALECTS: tuple[str, ...] = ("openai", "glm", "deepseek", "ollama")

#: 方言 → 该方言的说明，供错误信息与文档复用（新增方言时只改这一处）。
DIALECT_NOTES: dict[str, str] = {
    "openai": "纯 OpenAI 兼容：不认识的私有参数一律不发（默认）",
    "glm": "智谱：thinking 开启时 reasoning_effort 生效，输出上限改名为 max_tokens",
    "deepseek": "DeepSeek：reasoning_effort 直通",
    "ollama": "Ollama：reasoning_effort 翻译为 think 布尔",
}


def resolve_dialect(declared: str | None = None, *, default: str = "openai") -> str:
    """确定实际生效的方言。

    Args:
        declared: 档案显式声明的 ``dialect``；``None`` / 空串表示未声明。
        default: 未声明时的默认方言。调用方（``ClientFactory``）传**该提供商规格行
            里的默认方言**；不传即 ``openai``。

    Raises:
        ValueError: 生效的方言不在 :data:`DIALECTS` 里——拼错必须尽早暴露，
            静默降级为 ``openai`` 会让"配置了却没生效"极难排查。默认值一并校验：
            规格表里写错一个字母同样要在这里响，而不是让所有档案悄悄退化。
    """
    effective = default if (declared is None or declared == "") else declared
    if effective not in DIALECTS:
        known = "、".join(f"'{name}'" for name in DIALECTS)
        raise ValueError(
            f"不支持的 dialect: {effective!r}（可用：{known}）。"
            f"说明：" + "；".join(f"{k}={v}" for k, v in DIALECT_NOTES.items()))
    return effective


def adapt_payload(payload: dict, dialect: str) -> None:
    """就地按方言修正请求体（``payload`` 已含白名单内的生成参数）。

    ``extra_body`` 不在这里处理：它的语义是**调用方对具体端点能力的显式声明**
    （见 ``client.LLMClient._build_payload``），因此由调用方在方言适配**之后**并入，
    保证方言规则不会把它删掉。
    """
    effort = payload.get('reasoning_effort')
    payload.pop('extra_body', None)          # 兜底：正常路径下已被合并

    if dialect == 'glm':
        if 'max_completion_tokens' in payload:
            mc = payload.pop('max_completion_tokens')
            if mc is not None:   # None 表示调用方未设置：不得覆盖已配置的 max_tokens
                payload['max_tokens'] = mc
        if effort:
            payload['thinking'] = {'type': 'enabled'}
        return

    # 非智谱方言不支持 'thinking' 字段，一律移除
    payload.pop('thinking', None)

    if dialect == 'deepseek':
        if not effort:
            payload.pop('reasoning_effort', None)
        return

    if dialect == 'ollama':
        payload.pop('reasoning_effort', None)
        if effort:
            payload['think'] = True
        return

    # openai：不认识的一律不发（siliconflow/deepinfra/blt/cstcloud 与自定义提供商）
    payload.pop('reasoning_effort', None)
