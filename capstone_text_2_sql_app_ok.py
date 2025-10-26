import os
import json
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# ---- Load environment variables ----
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "imdb_movies")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/imdb.db")
CSV_PATH = os.getenv("CSV_PATH", "./data/imdb_top_1000.csv")

# ---- Optional imports ----
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except Exception:
    QdrantClient = None
    rest = None

# Prefer modern OpenAI client if available, fallback to legacy openai
_openai_modern = None
_openai_legacy = None
try:
    from openai import OpenAI as _OpenAIModern
    _openai_modern = _OpenAIModern(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    try:
        import openai as _openai_legacy_module
        _openai_legacy_module.api_key = OPENAI_API_KEY
        _openai_legacy = _openai_legacy_module
    except Exception:
        _openai_legacy = None

# ---- Helpers ----

def get_embeddings(texts):
    """Return list of vectors for given texts using OpenAIEmbeddings if available."""
    if OpenAIEmbeddings is not None:
        emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        return emb.embed_documents(texts)
    # fallback: attempt to use OpenAI embeddings directly
    if _openai_modern is not None:
        resp = _openai_modern.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]
    if _openai_legacy is not None:
        resp = _openai_legacy.Embedding.create(model="text-embedding-3-small", input=texts)
        return [d["embedding"] for d in resp["data"]]
    raise RuntimeError("No embedding provider available. Install langchain_openai or openai and set OPENAI_API_KEY.")


def normalize_columns(df: pd.DataFrame):
    """Normalize DataFrame column names to lowercase snake_case and return mapping."""
    col_map = {}
    new_cols = []
    for c in df.columns:
        nc = str(c).strip()
        nc = nc.replace(" ", "_")
        nc = nc.replace("-", "_")
        nc = nc.lower()
        # ensure valid sqlite column name (basic)
        nc = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in nc)
        col_map[c] = nc
        new_cols.append(nc)
    df.columns = new_cols
    return df, col_map


def create_sqlite_db_from_csv(csv_path, db_path):
    """
    Create SQLite DB from CSV and normalize column names (lowercase, underscores).
    Saves a column mapping file `${db_dir}/col_map.json` for debugging.
    """
    df = pd.read_csv(csv_path)
    df, col_map = normalize_columns(df)
    # ensure folder exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("movies", conn, if_exists="replace", index=False)
    conn.close()
    # save mapping
    try:
        with open(Path(db_path).parent / "col_map.json", "w", encoding="utf-8") as f:
            json.dump(col_map, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return df


def get_table_schema_sqlite(db_path, table="movies"):
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(f"PRAGMA table_info({table});")
        rows = cur.fetchall()
        cols = [r[1] for r in rows]
    finally:
        conn.close()
    return cols


def ingest_to_qdrant(df: pd.DataFrame, collection_name: str = None):
    collection_name = collection_name or QDRANT_COLLECTION
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set to ingest.")
    if QdrantClient is None:
        raise RuntimeError("qdrant-client not available. Install qdrant-client.")

    texts = []
    payloads = []
    # use normalized keys (lowercase)
    for _, row in df.iterrows():
        title = row.get("title") or row.get("name") or ""
        year = row.get("year") or ""
        genre = row.get("genre") or ""
        desc = row.get("description") or row.get("overview") or ""
        texts.append(f"Title: {title}\nYear: {year}\nGenre: {genre}\nDescription: {desc}")
        payloads.append({"title": title, "year": year, "genre": genre, "description": desc})

    vectors = get_embeddings(texts)

    qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_size = len(vectors[0]) if vectors else 1536
    try:
        qclient.recreate_collection(collection_name=collection_name,
                                    vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE))
    except Exception:
        qclient.create_collection(collection_name=collection_name,
                                  vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE))

    batch = 64
    points = []
    for i, vec in enumerate(vectors):
        points.append(rest.PointStruct(id=i, vector=vec, payload=payloads[i]))
        if len(points) >= batch:
            qclient.upsert(collection_name=collection_name, points=points)
            points = []
    if points:
        qclient.upsert(collection_name=collection_name, points=points)
    return len(vectors)


# ---- Simple NL->SQL generator using OpenAI (modern or legacy) ----
PROMPT_TEMPLATE = """
You are an assistant that converts a natural language question about the SQL table `movies` into a valid SQLite SQL SELECT query.
Use only the columns present in the table. Return only the SQL string on a single line (no explanation).
Columns: {schema}
Question: {question}
"""

def generate_sql_with_openai(question: str):
    if not OPENAI_API_KEY:
        return "", "OPENAI_API_KEY not set"
    # provide the current schema to the model
    cols = get_table_schema_sqlite(SQLITE_DB_PATH, "movies")
    schema_text = ", ".join(cols)
    prompt = PROMPT_TEMPLATE.format(schema=schema_text, question=question)
    try:
        if _openai_modern is not None:
            resp = _openai_modern.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt}], temperature=0)
            txt = resp.choices[0].message.content
        elif _openai_legacy is not None:
            resp = _openai_legacy.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt}], temperature=0)
            txt = resp["choices"][0]["message"]["content"]
        else:
            return "", "No OpenAI client available"
        # try to extract SQL (first occurrence starting with SELECT)
        import re
        m = re.search(r"(SELECT[\s\S]+?);?$", txt.strip(), flags=re.I)
        if m:
            return m.group(1).strip(), None
        # fallback: return whole text
        return txt.strip(), None
    except Exception as e:
        return "", str(e)


# ---- Execute SQL against local sqlite DB ----
def execute_sql(db_path: str, sql: str):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()


# ---- Streamlit UI ----
st.set_page_config(page_title="Text-to-SQL Capstone", layout="wide")
st.title("🧠 Capstone Project — Text to SQL with LangChain + Qdrant + Streamlit")

with st.sidebar:
    st.header("Project Configuration")
    st.write(f"SQLite DB: `{SQLITE_DB_PATH}`")
    st.write(f"Qdrant URL: `{QDRANT_URL}`")
    st.write(f"Collection: `{QDRANT_COLLECTION}`")
    if st.button("Create DB from CSV"):
        try:
            df = create_sqlite_db_from_csv(CSV_PATH, SQLITE_DB_PATH)
            st.success(f"Created DB with {len(df)} rows")
        except Exception as e:
            st.error(f"Create DB failed: {e}")
    if st.button("Ingest to Qdrant"):
        try:
            df = pd.read_csv(CSV_PATH)
            # normalize if needed
            df, _ = normalize_columns(df)
            cnt = ingest_to_qdrant(df)
            st.success(f"Ingested {cnt} vectors to {QDRANT_COLLECTION}")
        except Exception as e:
            st.error(f"Ingest failed: {e}")

st.header("Ask a natural language question")
question = st.text_area("Question", height=140)
col1, col2 = st.columns([1, 1])
with col1:
    k = st.number_input("RAG top-k (not used here)", min_value=0, max_value=10, value=4)
    show_sql = st.checkbox("Show generated SQL", value=True)
with col2:
    show_raw = st.checkbox("Show raw model output", value=False)

if st.button("Run Query"):
    if not question.strip():
        st.warning("Please write a question first.")
    else:
        st.info("Generating SQL...")
        sql, err = generate_sql_with_openai(question)
        if err:
            st.error(f"SQL generation failed: {err}")
        elif not sql:
            st.error("No SQL generated.")
        else:
            if show_sql:
                st.subheader("Generated SQL")
                st.code(sql)
            try:
                df_res = execute_sql(SQLITE_DB_PATH, sql)
                st.subheader("Result")
                st.dataframe(df_res)
                st.download_button("Download Results", data=df_res.to_csv(index=False).encode('utf-8'), file_name='results.csv', mime='text/csv')
            except Exception as e:
                st.error(f"SQL execution failed: {e}")

st.caption("NL->SQL uses OpenAI to generate queries. Make sure OPENAI_API_KEY is set in your .env.")
