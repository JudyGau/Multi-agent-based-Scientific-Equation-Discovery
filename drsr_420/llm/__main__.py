"""``python -m drsr_420.llm`` —— 命令行冒烟测试入口。

需要真实 API key，请通过环境变量提供，例如：
    DEEPSEEK_API_KEY=sk-... python -m drsr_420.llm
切勿硬编码密钥。
"""
import os
import sys

from drsr_420.llm.factory import ClientFactory


def main(argv=None) -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("未设置 DEEPSEEK_API_KEY 环境变量，跳过冒烟测试。")
        print("用法：DEEPSEEK_API_KEY=sk-... python -m drsr_420.llm")
    else:
        # 走工厂而不是直接构造客户端：端点与方言由提供商规格表给出，
        # 这里也顺带验证"内置提供商只写 model + api_key 就能用"这条路径。
        client = ClientFactory.from_config({
            "model": "deepseek/deepseek-reasoner",
            "api_key": api_key,
        })
        messages = [{"role": "user", "content": "你好，请介绍一下你自己，并说明你的思考过程。"}]
        try:
            print(client.chat(messages))
        except Exception as e:
            print(f"调用模型时出错: {e}")
    print('=' * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
