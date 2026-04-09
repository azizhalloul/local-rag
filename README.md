#  Local RAG Engine — TP n°1

> **Retrieval-Augmented Generation** from scratch using **FAISS**, **Gemma 3** and **Ollama** — fully local, no cloud calls.

---

## Overview

Large Language Models (LLMs) have a fixed knowledge cutoff: they cannot answer questions about private, proprietary, or recent data unless that information is explicitly provided to them.

**RAG (Retrieval-Augmented Generation)** solves this by combining two components:
1. A **semantic search engine** that retrieves the most relevant documents from a private knowledge base.
2. A **language model** that generates a grounded, contextual answer using only those retrieved documents.

This project implements a complete local RAG pipeline — from raw text to an interactive Q&A system — using only open-source tools running on your own machine.

---

##  Architecture

```
┌─────────────────────────────────────────────────┐
│                  YOUR MACHINE                   │
│                                                 │
│  ┌──────────────┐      HTTP      ┌───────────┐  │
│  │ Python Client│ ◄──────────── │  Ollama   │  │
│  │  (tp_rag.py) │  localhost:   │  Server   │  │
│  └──────────────┘    11434      └───────────┘  │
│         │                            │         │
│         │                    ┌───────┴───────┐  │
│         │                    │    Models     │  │
│         │                    │ nomic-embed   │  │
│         │                    │ gemma3:4b     │  │
│         │                    └───────────────┘  │
│         │                                       │
│  ┌──────┴──────────────────────────────────┐    │
│  │              RAG Pipeline               │    │
│  │  Documents → Embeddings → FAISS Index   │    │
│  │  Query → Retrieve → Prompt → Answer     │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| **Text embeddings** | 768-dimensional semantic vectors via `nomic-embed-text` |
| **Naïve search** | Brute-force cosine similarity (correctness baseline) |
| **FAISS search** | Optimized L2 nearest-neighbor index (scales to millions of docs) |
| **RAG pipeline** | Context-grounded Q&A with Gemma 3 (no hallucination from prior knowledge) |
| **Multimodal RAG** *(bonus)* | Image captioning bridge — index and retrieve images via text descriptions |

---

##  Key Concepts

### Embedding Space
Each document is transformed into a vector in a 768-dimensional semantic space. Two documents that are semantically similar will have vectors that point in the same direction, regardless of exact wording.

### Why FAISS instead of a plain loop?
A Python loop over n documents computes n dot products sequentially. FAISS uses optimized C++ routines with SIMD/AVX instructions and can index millions of vectors in a fraction of the time.

### L2 distance ≡ Cosine similarity (after normalization)
FAISS uses Euclidean (L2) distance by default. For **unit-normalized** vectors:

```
‖A − B‖² = 2 − 2·cos(θ)
```

Because `f(x) = 2 − 2x` is strictly decreasing:

> **Maximizing cosine similarity ↔ Minimizing L2 distance**

This is why every vector is normalized before being added to the index.

### The Semantic Bridge (Multimodal)
FAISS and nomic-embed-text only understand text. To index images, Gemma 3 acts as a **captioning bridge**:

```
Image → Gemma 3 (vision) → Text description → nomic-embed → FAISS vector
```

At query time, the original image is re-injected into the final prompt for fine-grained visual analysis.

---

##  Project Structure

```
tp1-local-rag/
├── tp_rag.py               # Main RAG pipeline
├── images/
│   └── turbo-encabulator_3000.png   # Sample image for multimodal RAG
└── README.md
```

---

##  Setup

### 1. Install Ollama

**Linux:**
```bash
mkdir ~/tp-ollama && cd ~/tp-ollama
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz
tar -xzf ollama.tgz
./bin/ollama serve          # Terminal 1 — keep this open
```

**Windows:** Download [OllamaSetup.exe](https://ollama.com/download/windows) and run it.

### 2. Pull the models

```bash
ollama pull nomic-embed-text
ollama pull gemma3:4b
```

> `gemma3:4b` is ~3 GB. If you are on a restricted network, download the `.gguf` files manually and import them with a `Modelfile`.

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install ollama faiss-cpu numpy
```

### 4. Run

```bash
python tp_rag.py
```

---

## 🔬 Implementation Details

### `get_embedding(text)` → `list[float]`
Calls the Ollama embedding endpoint and returns a 768-dim vector for any input string.

### `generate_embeddings(docs)` → `list[np.ndarray]`
Batch-encodes the entire knowledge base.

### `cosine_similarity(v1, v2)` → `float`
Manual implementation: `dot(v1, v2) / (‖v1‖ · ‖v2‖)`.

### `naive_search(query, embeddings, docs, k)` → `[(score, doc)]`
O(n) brute-force scan — useful as a ground-truth reference.

### `faiss_index(embeddings_np)` → `IndexFlatL2`
Normalizes vectors to unit length and adds them to a flat L2 index.

### `faiss_search(query, index, docs, k)` → `[(distance, doc)]`
Encodes and normalizes the query, then calls `index.search()`. Returns `(l2_distance, doc)` pairs — smaller distance = better match.

### `rag_query(user_query, index, docs)`
Full pipeline: retrieve top-3 → build grounded prompt → query Gemma 3 → print answer.

### `generate_description_for_image(path)` → `str`
Sends an image to Gemma 3 (vision mode) and returns a ≤200-word text description.

### `generate_multimodal_embeddings(docs)` → `list[np.ndarray]`
Like `generate_embeddings` but transparently captions image paths before encoding.

### `multimodal_rag_query(user_query, index, docs)`
Like `rag_query` but re-injects image files into the final prompt when retrieved.

---

##  Example Output

```
--- Generating embeddings for the knowledge base ---
    -> Embedding vector dimension: 768
    -> Embeddings generated for 8 documents.

--- Naïve cosine search for: 'How do I restart the machine in an emergency?' ---
    -> Top-3 results (naïve):
       [Score: 0.8742] To restart the emergency system, hold the red button for 5 seconds...
       [Score: 0.6123] Error 404 on this machine indicates overheating...
       [Score: 0.5891] The touchscreen interface may freeze...

--- FAISS search ---
    -> Top-3 results (FAISS):
       [L2 distance: 0.2516] To restart the emergency system, hold the red button for 5 seconds...

*** GEMMA 3 RESPONSE ***
----------------------------------------
To restart the machine in an emergency, hold the red button for 5 seconds, then turn the blue key.
----------------------------------------
```

---

##  Production Comparison

| Feature | FAISS (this project) | Vector DB (Chroma, Qdrant…) |
|---|---|---|
| Storage | RAM only | Persistent on disk |
| Stored data | Vectors only | Vectors + text + metadata |
| Filtering | Not supported | `"find 'PDF' docs near X"` |
| Architecture | Single process | Client-server (multi-user) |
| Use case | Learning & prototyping | Production workloads |

> FAISS is often the underlying engine inside production vector databases — understanding it gives you the foundation to evaluate any RAG infrastructure.

---

##  References

- [FAISS — Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Ollama — Local LLM runtime](https://ollama.com)
- [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- [Gemma 3 — Google DeepMind](https://ai.google.dev/gemma)
- [RAG — Lewis et al., 2020](https://arxiv.org/abs/2005.11401)

---

##  License

Academic project — Polytech Nantes, IDIA.
