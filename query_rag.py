import os
import pickle
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_KEY)

# -----------------------------
# Load BM25 + reviews list
# -----------------------------
with open("reviews.pkl", "rb") as f:
    reviews = pickle.load(f)

with open("bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

# -----------------------------
# Load ChromaDB (semantic vectors)
# -----------------------------
chroma_client = chromadb.PersistentClient(path="chroma_db")

openai_embed = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY,
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_or_create_collection(
    name="skincare_reviews",
    embedding_function=openai_embed
)

# -----------------------------
# HYBRID RETRIEVAL FUNCTION
# -----------------------------
def retrieve_hybrid(query: str, k: int = 6):
    # ✅ 1. BM25 keyword retrieval
    bm25_docs = bm25.get_top_n(query.split(), reviews, n=k)

    # ✅ 2. Semantic vector retrieval via Chroma
    results = collection.query(query_texts=[query], n_results=k)
    chroma_docs = results["documents"][0]

    # ✅ 3. Merge + remove duplicates (preserve order)
    merged = list(dict.fromkeys(bm25_docs + chroma_docs))
    return merged[:k]

# -----------------------------
# GPT-4 (or GPT-4o) Summary Function
# -----------------------------
def summarize_with_gpt(query: str, k: int = 6) -> str:
    retrieved = retrieve_hybrid(query, k=k)
    context = "\n".join(retrieved)

    prompt = f"""
You are an expert skincare product-review summarizer.
Based ONLY on the customer reviews below, write a clear factual summary including:

✅ Key Benefits (bullets)
✅  Common Complaints (bullets)
     pro and cons (bullets) 
✅ Ideal skin types / usage (1 line)
✅ Overall sentiment (3 line)

Customer Reviews:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # ✅ Use 4o-mini or gpt-4o
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

# -----------------------------
# Streamlit wrapper
# -----------------------------
def rag_summary(query: str, k: int = 6) -> str:
    return summarize_with_gpt(query, k=k)

# -----------------------------
# Optional CLI test
# -----------------------------
if __name__ == "__main__":
    q = input("Enter product or concern: ").strip()
    print("\n--- SUMMARY ---\n")
    print(rag_summary(q))
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt')

def evaluate_summary(generated: str, reference: str):
    """
    Compute ROUGE scores comparing generated summary with a reference summary.
    """

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"], use_stemmer=True
    )

    scores = scorer.score(reference, generated)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }
