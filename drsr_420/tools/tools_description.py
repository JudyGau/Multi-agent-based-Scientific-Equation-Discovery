# ── Tool Schema（DeepSeek 兼容 OpenAI tools 协议）───────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_paper",
            "description": "根据关键词 搜索中/英文论文，返回文献的元数据(DOI、标题、期刊/会议名称、作者、年份、被引量等)，但无法直接获取文献内容",
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
                    "title_doi": {
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
                        "description": "A list of (paper_title, doi) string pairs."
                    }
                },
                "required": ["title_doi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_paper",
            "description": (
                "将本地已有的 PDF 文献文件嵌入到 RAG 文献知识库。"
                "入参 pdf_path 为 PDF 文件路径（可相对项目根目录，如 'pdf_downloads/xxx.pdf'），"
                "doi 与 title 可选。返回入库的片段数量。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "PDF 文件路径，可相对项目根目录。",
                    },
                    "doi": {
                        "type": "string",
                        "description": "文献 DOI（可选，用于去重与溯源）。",
                    },
                    "title": {
                        "type": "string",
                        "description": "文献标题（可选）。",
                    },
                },
                "required": ["pdf_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": (
                "在 RAG 文献知识库中检索与 query 语义相关的文献片段，返回 top-k 条命中"
                "（标题、DOI、来源文件、正文、相似度距离）。知识库需已用 ingest_paper 或 CLI 入库。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词或问题描述。",
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回条数，默认 5。",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]
