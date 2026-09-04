"""RAG 文献知识库命令行工具。

用法：
    python -m drsr_420.rag_build --ingest [--dir pdf_downloads] [--limit N] [--rebuild]
    python -m drsr_420.rag_build --query "磁流变 屈服应力 压缩" [--k 5]
"""
import argparse
import json

from drsr_420.rag_kb import RagKB, load_config


def main():
    parser = argparse.ArgumentParser(description="RAG 文献知识库工具")
    parser.add_argument("--ingest", action="store_false", help="批量入库 PDF 到知识库")
    parser.add_argument("--dir", default="../pdf_downloads", help="PDF 目录（默认 pdf_downloads）")
    parser.add_argument("--limit", type=int, default=None, help="最多入库的文件数")
    parser.add_argument("--rebuild", action="store_false", help="重建 collection（删除后重新入库）")
    parser.add_argument("--query", default=None, help="检索关键词")
    parser.add_argument("--k", type=int, default=5, help="检索返回条数")
    parser.add_argument("--config", default="rag.config", help="配置文件路径（默认 rag.config）")
    args = parser.parse_args()

    kb = RagKB(load_config(args.config))

    if args.ingest:
        if args.rebuild:
            print("[RAG] 重建 collection ...")
            kb.reset_collection()
        results = kb.ingest_dir(args.dir, limit=args.limit)
        print(json.dumps(results, ensure_ascii=False))
        print(f"[RAG] 库内总数: {kb.count()}")

    if args.query:
        hits = kb.search(args.query, k=args.k)
        print(f"\n[RAG] 共 {len(hits)} 条命中（distance 越小越相关）:\n")
        for i, h in enumerate(hits, 1):
            print(f"--- [{i}] {h['title']} | doi={h['doi']} | source={h['source_file']} | distance={h['distance']:.4f} ---")
            print(h["text"])
            print()

    if not args.ingest and not args.query:
        parser.print_help()


if __name__ == "__main__":
    main()
