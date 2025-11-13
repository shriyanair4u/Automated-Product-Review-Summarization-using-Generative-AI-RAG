Automated Product Review Summarization using Generative AI + RAG

A fully automated Retrieval-Augmented Generation (RAG) pipeline designed to summarize large volumes of customer product reviews using GPT-based LLMs, OpenAI embeddings, and ChromaDB.
This system extracts key customer sentiments, product issues, and feature insights—reducing manual review time and improving decision-making for product teams.

📌 Project Overview

Retail businesses receive thousands of customer reviews that are time-consuming to read and analyze.
This project solves that by building a context-aware summarization engine that combines:

Semantic Retrieval (ChromaDB + OpenAI Embeddings)

LLM Summarization (GPT-4/GPT-3.5)

BM25 Hybrid Search (optional)

ROUGE/BLEU evaluation metrics

The RAG model retrieves only the most relevant reviews and feeds them into GPT to generate accurate, concise, sentiment-aligned summaries.

🧠 Architecture
Raw Reviews → Preprocessing → Embeddings (OpenAI) → Vector Store (ChromaDB)
                     ↓
          Semantic Retrieval (Top-k Reviews)
                     ↓
        RAG Pipeline (LangChain + GPT Models)
                     ↓
       Final Summary (Sentiment-Aware, Concise)
                     ↓
       Evaluation (ROUGE/BLEU Metrics)

🔥 Key Features

✔ Automatic summarization of large review datasets
✔ RAG-enabled hybrid search (ChromaDB + BM25)
✔ GPT-based sentiment-aware summarization
✔ Indexed vector database for fast retrieval
✔ Streamlit UI for live summarization
✔ FastAPI endpoint for integration
✔ ROUGE/BLEU metric evaluation

🛠 Tech Stack
Component	Tools
Language	Python
LLM	GPT-4 / GPT-3.5
Embeddings	OpenAI Embeddings 3 Small
Vector DB	ChromaDB / FAISS
Framework	LangChain
Retrieval	Semantic + BM25
Evaluation	ROUGE, BLEU
Deployment	Docker, FastAPI, Streamlit
📁 Project Structure
src/
│── rag/
│   ├── build_index.py        # Create embeddings + ChromaDB vectorstore
│   ├── query_rag.py          # Retrieval + GPT summarization logic
│   ├── eval.py               # ROUGE/BLEU evaluation script
│── api/
│   └── api.py                # FastAPI endpoint for API use
│── ui/
│   └── app_streamlit.py      # Streamlit UI for demo
data/
│── raw/
│   └── reviews.csv           # Source reviews
│── processed/
│   └── reviews.pkl           # Cleaned/processed reviews
models/
│── bm25.pkl                  # BM25 index
vectorstore/
│── chroma_db/                # Chroma vector database
requirements.txt
Dockerfile
README.md

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Add API Key (.env file)
OPENAI_API_KEY=your_key_here

3️⃣ Build the Index (Embeddings + Vectorstore)
python src/rag/build_index.py

4️⃣ Run Streamlit App
streamlit run src/ui/app_streamlit.py

5️⃣ Run FastAPI
uvicorn src.api.api:app --reload

🧪 Evaluation (ROUGE/BLEU)

Run:

python src/rag/eval.py


The script outputs:

ROUGE-1

ROUGE-L

BLEU score

Summary vs. Reference comparison

📊 Business Impact

✔ Reduced manual review analysis time by 50%
✔ Enabled faster product insight generation
✔ Improved accuracy of customer sentiment interpretation
✔ Helped product teams identify top issues and feature requests quickly

🙋‍♀️ Author

Shriya Nair
Data Scientist | GenAI | RAG | NLP
