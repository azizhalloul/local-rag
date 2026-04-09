"""
Local RAG Engine  
Retrieval-Augmented Generation with Faiss, Gemma 3 & Ollama

Architecture:
  - Embedding model : nomic-embed-text  (768-dim vectors)
  - Chat model      : gemma3:4b         (multimodal)
  - Vector store    : FAISS IndexFlatL2 (with L2-normalization ≡ cosine similarity)
  - Runtime         : fully local via Ollama (no cloud calls)

Phases:
  1. Environment setup & Ollama client init
  2. Document vectorization (naive cosine + FAISS)
  3. Augmented generation with Gemma 3
  4. Bonus — Multimodal RAG (image captioning bridge)
"""

import sys
import os
import time

import numpy as np
import faiss
import ollama


# CONFIGURATION

# local Ollama server
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

# Ollama client pointing at the local server (default port 11434)
client = ollama.Client(host="http://127.0.0.1:11434")

# Model used to encode text into semantic vectors
EMBED_MODEL = "nomic-embed-text"

# Model used to generate natural-language answers (supports multimodal input)
CHAT_MODEL = "gemma3:4b"


# PHASE 2 — DOCUMENT VECTORIZATION

# Step 3 : Embedding generation 

def get_embedding(text: str) -> list[float]:
    """
    Generate a semantic embedding vector for a given text using nomic-embed-text.

    The returned vector lives in a 768-dimensional space where geometric
    proximity reflects semantic similarity between texts.

    Args:
        text: Any UTF-8 string to encode.

    Returns:
        A list of 768 floats representing the text's position in embedding space.

    Raises:
        SystemExit if Ollama is unreachable or the model is not installed.
    """
    try:
        response = client.embeddings(model=EMBED_MODEL, prompt=text)
        return response["embedding"]
    except Exception as exc:
        print(
            f"[ERROR] Ollama call failed — make sure 'ollama serve' is running "
            f"and model '{EMBED_MODEL}' is installed.\nDetails: {exc}"
        )
        sys.exit(1)


def generate_embeddings(docs: list[str]) -> list[np.ndarray]:
    """
    Encode every document in the knowledge base into a float32 numpy vector.

    Args:
        docs: List of text strings to encode.

    Returns:
        List of numpy arrays, one per document, dtype float32.
    """
    embeddings = []
    for doc in docs:
        vec = get_embedding(doc)
        embeddings.append(np.array(vec, dtype="float32"))
    return embeddings


# ── Step 4 : Naïve cosine-similarity search ──────────────────────────────────

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Formula:
        cos(θ) = (v1 · v2) / (‖v1‖ · ‖v2‖)

    A value of 1.0 means the vectors are identical in direction (maximum
    semantic match); 0.0 means orthogonal (no semantic overlap).

    Args:
        v1, v2: 1-D numpy arrays of the same length.

    Returns:
        Cosine similarity score in [-1, 1].
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def naive_search(
    query: str,
    embeddings: np.ndarray,
    docs: list[str],
    nb_results: int = 3,
) -> list[tuple[float, str]]:
    """
    Retrieve the top-k most relevant documents via brute-force cosine similarity.

    This O(n) scan is simple and correct but does not scale to large corpora
    (millions of documents).  It serves as a correctness baseline for FAISS.

    Args:
        query      : User question in natural language.
        embeddings : (n, d) numpy array of pre-computed document embeddings.
        docs       : List of n document strings (parallel to embeddings).
        nb_results : Number of top matches to return.

    Returns:
        List of (score, document) tuples sorted by descending similarity.
    """
    query_vec = np.array(get_embedding(query), dtype="float32")
    scored = [
        (cosine_similarity(query_vec, embeddings[i]), docs[i])
        for i in range(len(docs))
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:nb_results]


# ── Step 5 : Industrial-scale search with FAISS ──────────────────────────────

def faiss_index(embeddings_np: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS flat-L2 index from a set of embedding vectors.

    Mathematical equivalence — why L2 ≡ cosine after normalization:
        ‖A − B‖² = ‖A‖² + ‖B‖² − 2(A · B)
                 = 1 + 1 − 2·cos(θ)      (when ‖A‖ = ‖B‖ = 1)
                 = 2 − 2·cos(θ)

    Because f(x) = 2 − 2x is strictly decreasing, maximizing cosine is
    equivalent to minimizing L2 distance.  FAISS uses L2 because modern CPUs
    compute Euclidean distances faster than normalized dot products.

    Args:
        embeddings_np: (n, d) array of document embeddings (any dtype).

    Returns:
        A FAISS IndexFlatL2 object ready for search queries.
    """
    # FAISS requires float32
    vectors = embeddings_np.astype("float32")

    # Normalize to unit length → L2 distance becomes equivalent to cosine
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    print(f"\t-> FAISS index built with {index.ntotal} documents.")
    return index


def faiss_search(
    query: str,
    index: faiss.IndexFlatL2,
    docs: list[str],
    nb_results: int = 3,
) -> list[tuple[float, str]]:
    """
    Retrieve the top-k most relevant documents using an optimized FAISS index.

    Compared to naive_search, FAISS scales to millions of vectors by using
    optimized C++ routines and can leverage SIMD/GPU acceleration.

    Note: The returned distance is a *squared* L2 distance after normalization,
    which is equivalent to (2 − 2·cosine_similarity). Smaller = more similar.

    Args:
        query      : User question in natural language.
        index      : Pre-built FAISS index (see faiss_index).
        docs       : List of document strings in the same order as the index.
        nb_results : Number of nearest neighbors to retrieve.

    Returns:
        List of (l2_distance, document) tuples sorted by ascending distance.
    """
    query_vec = np.array([get_embedding(query)], dtype="float32")
    faiss.normalize_L2(query_vec)  # Must normalize the query as well

    distances, indices = index.search(query_vec, nb_results)

    return [
        (distances[0][i], docs[indices[0][i]])
        for i in range(nb_results)
    ]


# PHASE 3 — AUGMENTED GENERATION WITH GEMMA 3

# Step 6 : RAG query pipeline

def rag_query(user_query: str, index: faiss.IndexFlatL2, docs: list[str]) -> None:
    """
    Full RAG pipeline: retrieve → augment prompt → generate answer.

    Steps:
      1. Embed the user question and retrieve the 3 most relevant documents.
      2. Build a grounded prompt that instructs the model to answer *only*
         from the retrieved context (preventing hallucination).
      3. Send the prompt to Gemma 3 and print the response.

    The system prompt deliberately constrains Gemma 3 to the provided context.
    If the answer is not present, the model should reply "I don't know" rather
    than fabricating an answer from its pre-training knowledge.

    Args:
        user_query : User question in natural language.
        index      : Pre-built FAISS index over the knowledge base.
        docs       : Original document strings (parallel to the index).
    """
    # ── 1. Retrieve the most relevant context 
    retrieved = faiss_search(user_query, index, docs, nb_results=3)
    context = "\n".join([f"- {doc}" for _, doc in retrieved])

    # ── 2. Build the grounded prompt
    system_prompt = f"""
You are a technical expert assistant.
Answer ONLY using the context provided below.
If the answer is not present in the context, say "I don't know".

CONTEXT:
{context}

QUESTION:
{user_query}
"""

    # ── 3. Generate and display the answer
    try:
        response = client.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": system_prompt}],
        )
        print("*** GEMMA 3 RESPONSE ***")
        print("-" * 40)
        print(response["message"]["content"])
        print("-" * 40)
    except Exception as exc:
        print(f"[ERROR] Gemma 3 call failed: {exc}")


# PHASE 4 — BONUS : MULTIMODAL RAG WITH GEMMA 3

#  Step 7 : The "Semantic Bridge" concept
#
# FAISS and nomic-embed-text only understand text.  To index images we use a
# "captioning-based retrieval" strategy:
#
#   1. Ingestion  — Gemma 3 (vision mode) describes the image in natural language.
#   2. Indexing   — The description is vectorized and stored exactly like any text.
#   3. Retrieval  — A text query retrieves the closest description in embedding space.
#   4. Generation — The *original image* (via its path) is re-injected into the
#                   final prompt so Gemma 3 can do fine-grained visual analysis.
#
# Why not embed the image pixels directly?
#   Gemma 3's internal visual vectors are *not mathematically aligned* with
#   nomic-embed-text's textual vectors.  Computing distances across both spaces
#   would be meaningless.  The captioning bridge keeps everything in the same
#   768-dimensional text space.


def generate_description_for_image(image_path: str) -> str:
    """
    Use Gemma 3 (vision mode) to generate a concise textual description of an image.

    The description is deliberately capped at ~200 words to stay within the
    token budget when the embedding model processes it downstream.

    Args:
        image_path: Absolute or relative path to the image file (PNG, JPEG…).

    Returns:
        A plain-text description of the image content, or an empty string on error.
    """
    try:
        response = client.chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Describe this image in detail in 200 words maximum. "
                        "Focus on visible technical elements, diagrams, labels, "
                        "and any text present."
                    ),
                    "images": [image_path],
                }
            ],
        )
        return response["message"]["content"]
    except Exception as exc:
        print(f"[ERROR] Could not describe image '{image_path}': {exc}")
        return ""


def generate_multimodal_embeddings(docs: list[str]) -> list[np.ndarray]:
    """
    Generate embeddings for a mixed knowledge base (text documents + image paths).

    For each entry:
      - If the entry looks like an image path (.png / .jpg / .jpeg / .webp / .gif),
        Gemma 3 first generates a textual caption, which is then embedded.
      - Otherwise, the text is embedded directly.

    This keeps all vectors in the same semantic space regardless of media type.

    Args:
        docs: List of strings — either raw text or file paths to images.

    Returns:
        List of float32 numpy arrays, one per document.
    """
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    embeddings = []

    for doc in docs:
        if doc.lower().endswith(IMAGE_EXTENSIONS):
            print(f"\t   [IMAGE] Generating caption for: {doc}")
            text_to_embed = generate_description_for_image(doc)
            if not text_to_embed:
                text_to_embed = f"[Image at path: {doc}]"  # fallback
        else:
            text_to_embed = doc

        vec = get_embedding(text_to_embed)
        embeddings.append(np.array(vec, dtype="float32"))

    return embeddings


def multimodal_rag_query(
    user_query: str,
    index: faiss.IndexFlatL2,
    docs: list[str],
) -> None:
    """
    RAG pipeline extended to handle image documents.

    If any of the top-k retrieved documents is an image path, the image is
    included directly in the prompt so Gemma 3 can perform visual analysis
    in addition to using the text context.

    Args:
        user_query : User question in natural language.
        index      : Pre-built FAISS multimodal index.
        docs       : Knowledge base entries (text strings or image paths).
    """
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

    # 1. Retrieve top-3 documents 
    retrieved = faiss_search(user_query, index, docs, nb_results=3)

    # ── 2. Separate text context from image paths
    text_parts: list[str] = []
    image_paths: list[str] = []

    for _, doc in retrieved:
        if doc.lower().endswith(IMAGE_EXTENSIONS):
            image_paths.append(doc)
            text_parts.append(f"[Attached image: {doc}]")
        else:
            text_parts.append(f"- {doc}")

    context = "\n".join(text_parts)

    # 3. Build the multimodal prompt
    prompt_text = f"""
You are a technical expert assistant.
Answer ONLY using the context and images provided below.
If the answer cannot be found in the context, say "I don't know".

CONTEXT:
{context}

QUESTION:
{user_query}
"""

    message: dict = {"role": "user", "content": prompt_text}
    if image_paths:
        message["images"] = image_paths

    # ── 4. Generate and display the answer
    try:
        response = client.chat(
            model=CHAT_MODEL,
            messages=[message],
        )
        print("*** GEMMA 3 MULTIMODAL RESPONSE ***")
        print("-" * 40)
        print(response["message"]["content"])
        print("-" * 40)
    except Exception as exc:
        print(f"[ERROR] Multimodal Gemma 3 call failed: {exc}")


# MAIN

if __name__ == "__main__":

    #  Simulated private knowledge base (Gemma 3 has no prior knowledge of this)
    knowledge_base: list[str] = [
        "The Turbo-Encabulator 3000 uses a pre-activated logarithmic stator.",
        "To restart the emergency system, hold the red button for 5 seconds, then turn the blue key.",
        "Error 404 on this machine indicates overheating of the main fluxional capacitor.",
        "Ball bearing maintenance must be performed every 150 cycles by a level-2 certified technician.",
        "The nominal input voltage is 220V, but the device tolerates fluctuations between 210V and 240V.",
        "In case of coolant leak (green color), evacuate the area immediately.",
        "The Wi-Fi module connects only on the 2.4GHz band using the WPA2 protocol.",
        "The touchscreen interface may freeze if the operator wears non-conductive latex gloves.",
    ]

    user_query = "How do I restart the machine in an emergency?"

    # STEP 3 : Generate embeddings
    print("\n--- Generating embeddings for the knowledge base ---")
    print(f"\t-> Embedding vector dimension: {len(get_embedding('Test'))}")

    t0 = time.time()
    embeddings = generate_embeddings(knowledge_base)
    embeddings_np = np.array(embeddings)
    t_vectorize = time.time() - t0

    print(f"\t-> Embeddings generated for {len(embeddings)} documents.")
    print(f"\t-> NumPy array shape: {embeddings_np.shape}")

    # STEP 4 : Naïve cosine search
    print(f"\n--- Naïve cosine search for: '{user_query}' ---")
    t0 = time.time()
    naive_results = naive_search(user_query, embeddings_np, knowledge_base, nb_results=3)
    t_naive = time.time() - t0

    print("\t-> Top-3 results (naïve):")
    for score, doc in naive_results:
        print(f"\t   [Score: {score:.4f}] {doc}")

    # ── STEP 5 : FAISS index + search
    print("\n--- Building FAISS index ---")
    t0 = time.time()
    index = faiss_index(embeddings_np)
    t_faiss_build = time.time() - t0

    print(f"\n--- FAISS search for: '{user_query}' ---")
    t0 = time.time()
    faiss_results = faiss_search(user_query, index, knowledge_base, nb_results=3)
    t_faiss_search = time.time() - t0

    print("\t-> Top-3 results (FAISS):")
    for dist, doc in faiss_results:
        print(f"\t   [L2 distance: {dist:.4f}] {doc}")

    # ── STEP 6 : Full RAG with Gemma 3
    print(f"\n--- Full RAG pipeline for: '{user_query}' ---")
    t0 = time.time()
    rag_query(user_query, index, knowledge_base)
    t_rag = time.time() - t0

    # ── STEP 7 (BONUS) : Multimodal RAG
    print("\n--- Multimodal RAG (Bonus) ---")
    knowledge_base.append("images/turbo-encabulator_3000.png")

    print("-> Generating multimodal embeddings...")
    t0 = time.time()
    mm_embeddings = generate_multimodal_embeddings(knowledge_base)
    t_mm_vectorize = time.time() - t0

    print("-> Building multimodal FAISS index...")
    t0 = time.time()
    mm_index = faiss_index(np.array(mm_embeddings))
    t_mm_build = time.time() - t0

    print("-> Running multimodal RAG query...")
    t0 = time.time()
    multimodal_rag_query(
        "Describe how the Turbo-Encabulator 3000 works.",
        mm_index,
        knowledge_base,
    )
    t_mm_rag = time.time() - t0

    # Timing summary
    print("Execution time summary:")
    print(f"  Document vectorization   : {t_vectorize:.2f}s")
    print(f"  Naïve cosine search      : {t_naive:.4f}s")
    print(f"  FAISS index build        : {t_faiss_build:.4f}s")
    print(f"  FAISS search             : {t_faiss_search:.4f}s")
    print(f"  RAG with Gemma 3         : {t_rag:.2f}s")
    print("  --- Multimodal ---")
    print(f"  Multimodal vectorization : {t_mm_vectorize:.2f}s")
    print(f"  Multimodal FAISS build   : {t_mm_build:.4f}s")
    print(f"  Multimodal RAG           : {t_mm_rag:.2f}s")
