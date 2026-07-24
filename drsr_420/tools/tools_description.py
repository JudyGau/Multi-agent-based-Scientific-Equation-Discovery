# ── Tool Schema（DeepSeek 兼容 OpenAI tools 协议）───────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_google_scholar",
            "description": "用 Google Scholar 搜索中文学术 / 英文论文，返回标题、作者、年份、被引量等。当用户问到论文、研究、文献、某作者工作时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "学术搜索关键词，如 'Apple Inc international marketing 2024'",
                    },
                    "num": {
                        "type": "integer",
                        "description": "返回条数，默认 10，最大 20",
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
