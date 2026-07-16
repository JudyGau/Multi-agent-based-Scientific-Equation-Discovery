import sqlite3
import pandas as pd

DB_PATH = "../knowledge_base/L3_data/experiment.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS samples (
        sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
        shape TEXT NOT NULL,
        AR1 REAL, AR2 REAL,
        phi REAL, B_field REAL,
        tau_y REAL,
        A_hys REAL DEFAULT NULL,
        notes TEXT
    )''')
    conn.commit()
    conn.close()

def insert_data(data_list):
    """data_list: list of dicts with keys shape, AR1, AR2, phi, B_field, tau_y"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.DataFrame(data_list)
    df.to_sql('samples', conn, if_exists='append', index=False)
    conn.close()

# if __name__ == "__main__":
#     init_db()
#     # 示例：插入你的19个点
#     your_data = [
#         {"shape":"ellip", "AR1":1.0, "AR2":1.0, "phi":0.3, "B_field":0.5, "tau_y":1200},
#         # ... 全部19个
#     ]
#     insert_data(your_data)
#     print("数据已插入")

import openai
import json
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# 初始化 Chroma 客户端
client = PersistentClient(path="./knowledge_base/chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction("BAAI/bge-small-zh-v1.5")
col_l0 = client.get_collection("L0_literature", embedding_function=ef)
# context_l0 = get_context(f"{shape_type} 颗粒 磁流变 本构 屈服应力", col_l0)
# col_l2 = client.get_collection("L2_methodology", embedding_function=ef)


def get_context(query, collection, n=3):
    results = collection.query(query_texts=[query], n_results=n)
    return "\n".join(results["documents"][0])


def generate_candidates(shape_type, AR1_range, AR2_range):
    """shape_type: 'ellip' or 'cube'"""
    # 从 L0 拉经典骨架
    context_l0 = get_context(f"{shape_type} 颗粒 磁流变 本构 屈服应力", col_l0)
    # 从 L2 拉 SR 约束
    # context_l2 = get_context("小样本 SR 骨架 候选 生成", col_l2)

    prompt = f"""你是一位物理学家，正在研究{shape_type}颗粒的几何参数(AR1, AR2)与屈服应力τ_y的关系。
已知经典 MRF 本构骨架如下：
{context_l0}
请基于以上知识，提出 3 个候选表达式（数学形式），用于描述 τ_y 随 AR1, AR2 的变化。
每个表达式应包含不超过 4 个自由参数（用字母 k, α, β, γ 表示）。
只输出 JSON 格式，不要多余文字：
[
  {{"name": "候选1", "expression": "τ_y = k * AR1^α * AR2^β"}},
  ...
]
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",  # 或你用的模型
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    candidates = json.loads(response.choices[0].message.content)
    return candidates


# 示例：为椭球生成候选
cands_ellip = generate_candidates("ellip", [1, 5], [1, 2])
print(cands_ellip)
