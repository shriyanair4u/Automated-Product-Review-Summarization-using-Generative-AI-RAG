iimport pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import os
import pickle
import sys

# Load API Key
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    print("❌ ERROR: OPENAI_API_KEY missing in .env!")
    sys.exit()

CSV_FILE = "skincare.csv"

if not os.path.exists(CSV_FILE):
    print(f"❌ CSV not found: {CSV_FILE}")
    sys.exit()

# Load CSV
df = pd.read_csv(CSV_FILE)

TEXT_COLUMN = "review"
if TEXT_COLUMN not in df.columns:
    print(f"❌ ERROR: CSV must contain a '{TEXT_COLUMN}' column.")
    sys.exit()

reviews = df[TEXT_COLUMN].dropna().tolist()
print(f"✅ Loaded {len(reviews)} reviews.")

# Save reviews.pkl
with open("reviews.pkl", "wb") as f:
    pickle.dump(reviews, f)
print("✅ Saved reviews.pkl")

# Build BM25
tokenized = [r.split() for r in reviews]
bm25 = BM25Okapi(tokenized)

with open("bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)
print("✅ Saved bm25.pkl")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")

openai_embed = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY,
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_or_create_collection(
    name="skincare_reviews",
    embedding_function=openai_embed
)

collection.delete(where={})

print("⏳ Indexing started...")

BATCH = 200
for i in range(0, len(reviews), BATCH):
    batch = reviews[i:i + BATCH]
    ids = [f"doc_{j}" for j in range(i, i + len(batch))]
    collection.add(documents=batch, ids=ids)
    print(f"✅ Indexed {i + len(batch)}/{len(reviews)}")

print("\n✅ DONE! Created files: reviews.pkl, bm25.pkl, chroma_db/\n")
