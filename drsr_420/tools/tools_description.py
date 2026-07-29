# ── Tool Schema（DeepSeek 兼容 OpenAI tools 协议）───────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_paper",
            "description": "根据关键词 搜索中/英文论文，返回DOI、标题、期刊/会议名称、作者、年份、被引量等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "学术搜索关键词，如 'Machine Learning'",
                    },
                    "num": {
                        "type": "integer",
                        "description": "返回条数，默认 10",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_paper",
            "description": "根据论文的doi链接来下载论文并获取论文内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "title_url": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": [
                                {"type": "string"},
                                {"type": "string"}
                            ]
                        },
                        "description": "A list of (paper_title, url) string pairs."
                    }
                },
                "required": ["title_url"],
            },
        },
    },
]
