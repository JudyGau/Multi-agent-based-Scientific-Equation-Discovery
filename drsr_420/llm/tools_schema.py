# ── Tool Schema（DeepSeek 兼容 OpenAI tools 协议）───────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_paper",
            "description": "Search academic papers by keywords (Chinese/English). Returns paper metadata (DOI, title, journal/conference, authors, year, citation count, etc.), but not the paper content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Academic search keyword, e.g. 'Machine Learning'",
                    },
                    "num": {
                        "type": "integer",
                        "description": "Number of results to return, default 10",
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
            "description": "Download the paper by its DOI link and extract its content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title_doi": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "[title, DOI] pair",
                        },
                        "description": "List of (title, DOI) pairs; each item is a [title, DOI] pair.",
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
                "Embed an existing local PDF literature file into the RAG knowledge base. "
                "pdf_path is the PDF file path (relative to the project root is OK, e.g. 'pdf_downloads/xxx.pdf'). "
                "doi and title are optional. Returns the number of chunks ingested."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "PDF file path, relative to the project root is allowed.",
                    },
                    "doi": {
                        "type": "string",
                        "description": "Paper DOI (optional, used for deduplication and provenance).",
                    },
                    "title": {
                        "type": "string",
                        "description": "Paper title (optional).",
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
                "Search the RAG knowledge base for literature chunks semantically related to query; "
                "returns the top-k hits (title, DOI, source file, text, similarity distance). "
                "The knowledge base must be populated first via ingest_paper or the CLI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or question description.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return, default 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]
