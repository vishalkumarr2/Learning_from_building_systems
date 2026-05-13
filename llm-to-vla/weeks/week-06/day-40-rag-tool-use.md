# Day 40: RAG & Tool Use

> **Phase III — LLMs: Training & Alignment | Week 6 | 2.5 hours**
> *"Retrieval-augmented generation: why memorize the encyclopedia when you can look it up?" — Patrick Lewis*

## Navigation
- **Previous:** [Day 39: Long Context & Reasoning](day-39-long-context-reasoning.md)
- **Next:** [Day 41: LLM for Robotics](day-41-llm-for-robotics.md)
- **Week:** [Week 6 Overview](README.md)
- **Phase:** [Phase III: LLM Engineering](../../study-notes/07-llm-engineering.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 40.1 RAG Architecture

Retrieval-Augmented Generation combines an LLM with an external knowledge base:

```
┌──────────────────────────────────────────────────────┐
│                    RAG Pipeline                       │
│                                                       │
│  User Query ──→ Embedder ──→ Vector Search ──→ Top-K │
│       │                           │            docs   │
│       │                      ┌────┴────┐              │
│       │                      │ Vector  │              │
│       │                      │   DB    │              │
│       │                      └─────────┘              │
│       │                                               │
│       └──→ Prompt = Query + Retrieved Docs ──→ LLM   │
│                                                  │    │
│                                            Answer ◄───┘│
└──────────────────────────────────────────────────────┘
```

**Indexing pipeline (offline):**

$$
\text{Documents} \xrightarrow{\text{chunk}} \text{Chunks} \xrightarrow{\text{embed}} \text{Vectors} \xrightarrow{\text{store}} \text{Vector DB}
$$

**Query pipeline (online):**

$$
\text{Query} \xrightarrow{\text{embed}} q \xrightarrow{\text{search}} \text{Top-K} \xrightarrow{\text{augment}} \text{Prompt} \xrightarrow{\text{LLM}} \text{Answer}
$$

### 40.2 RAG vs Fine-Tuning vs Long Context

| Approach | When to Use | Pros | Cons |
|----------|------------|------|------|
| **RAG** | Dynamic/updated knowledge | Fresh data, attributable | Retrieval quality limits output |
| **Fine-tuning** | Behavioral changes, format | Internalized knowledge | Stale, expensive to update |
| **Long context** | Full document analysis | No retrieval needed | Expensive, needle-in-haystack |

```
Decision tree:
  Does the knowledge change frequently?
    Yes → RAG (easy to update vector DB)
    No → Does the model need to change behavior?
      Yes → Fine-tuning (e.g., new output format)
      No → Does the full context fit in the window?
        Yes → Long context (stuff it all in)
        No → RAG (retrieve relevant parts)
```

### 40.3 Chunking Strategies

How you split documents dramatically affects retrieval quality:

```
Fixed-size chunks (simple):
  Split every 500 tokens with 50-token overlap
  ✅ Easy to implement
  ❌ Cuts mid-sentence, loses structure

Semantic chunks:
  Split on paragraph/section boundaries
  ✅ Preserves meaning
  ❌ Variable chunk sizes

Recursive character splitting:
  Try to split on: "\n\n" → "\n" → ". " → " " → ""
  ✅ Good balance
  ❌ Doesn't understand document structure

Parent-child chunks:
  Index small chunks (retrieval) but return parent chunk (context)
  ✅ Precise retrieval + full context
  ❌ More complex implementation
```

### 40.4 Embedding Models

| Model | Dimensions | Context | Quality (MTEB) |
|-------|-----------|---------|----------------|
| text-embedding-3-small | 1536 | 8191 | 62.3 |
| text-embedding-3-large | 3072 | 8191 | 64.6 |
| BGE-large-en-v1.5 | 1024 | 512 | 64.2 |
| E5-mistral-7b | 4096 | 32768 | 66.6 |
| Nomic-embed-text | 768 | 8192 | 62.4 |

**Similarity search:** given query embedding $q$ and document embedding $d$:

$$
\text{cosine\_sim}(q, d) = \frac{q \cdot d}{\|q\| \|d\|}
$$

### 40.5 Function Calling / Tool Use

LLMs can be trained to invoke external tools:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_robot_status",
        "description": "Get current status of a warehouse robot",
        "parameters": {
          "type": "object",
          "properties": {
            "robot_id": {"type": "string", "description": "Robot identifier"},
            "include_battery": {"type": "boolean", "default": true}
          },
          "required": ["robot_id"]
        }
      }
    }
  ]
}
```

**Tool use for robotics:**

```
User: "What's the battery level of robot OKS-42?"

LLM reasoning:
  1. I need to call get_robot_status(robot_id="OKS-42")
  2. [TOOL CALL] → {"battery": 67, "status": "navigating", ...}
  3. "Robot OKS-42 has 67% battery and is currently navigating."
```

### 40.6 RAG for Robotics Applications

```
Robot maintenance RAG:
  Knowledge base: service manuals, past incident reports, error codes
  Query: "Robot shows NAV_ESTIMATED_STATE_NOT_FINITE"
  Retrieved: 3 past incidents with this error + manual section
  Answer: Structured diagnosis with likely causes and resolution steps

Fleet monitoring RAG:
  Knowledge base: real-time telemetry, shift reports, alert history
  Query: "Why are robots in Zone B slow today?"
  Retrieved: Recent alerts for Zone B + environmental sensor data
  Answer: "Floor sensor detected wet surface in Zone B at 14:00,
           causing speed reduction protocol activation."
```

---

## Implementation (60 min)

### Build a Simple RAG System

```python
"""
Day 40 Implementation: Build a RAG system from scratch.
Uses sentence-transformers for embedding and FAISS for vector search.
"""
import json
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: np.ndarray | None = None

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


class SimpleEmbedder:
    """Embedding using bag-of-words TF-IDF (no GPU needed)."""

    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def fit(self, documents: list[str]):
        """Build vocabulary and compute IDF."""
        # Build vocabulary
        for doc in documents:
            for token in self._tokenize(doc):
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)

        # Compute IDF
        n_docs = len(documents)
        doc_freq = np.zeros(len(self.vocabulary))
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                if token in self.vocabulary:
                    doc_freq[self.vocabulary[token]] += 1

        self.idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1

    def embed(self, text: str) -> np.ndarray:
        """Compute TF-IDF embedding for a text."""
        tokens = self._tokenize(text)
        tf = np.zeros(len(self.vocabulary))
        for token in tokens:
            if token in self.vocabulary:
                tf[self.vocabulary[token]] += 1
        if tokens:
            tf /= len(tokens)

        tfidf = tf * (self.idf if self.idf is not None else 1.0)
        norm = np.linalg.norm(tfidf)
        return tfidf / norm if norm > 0 else tfidf


class VectorStore:
    """Simple vector store with cosine similarity search."""

    def __init__(self):
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None

    def add(self, documents: list[Document]):
        self.documents.extend(documents)
        vecs = np.array([d.embedding for d in documents])
        if self.embeddings is None:
            self.embeddings = vecs
        else:
            self.embeddings = np.vstack([self.embeddings, vecs])

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[SearchResult]:
        if self.embeddings is None:
            return []
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                document=self.documents[i],
                score=float(scores[i]),
                rank=rank,
            )
            for rank, i in enumerate(top_indices)
        ]


class RAGPipeline:
    """Complete RAG pipeline: index, retrieve, augment, generate."""

    def __init__(self, embedder: SimpleEmbedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def index_documents(self, texts: list[str], metadata: list[dict] | None = None):
        """Index documents into the vector store."""
        self.embedder.fit(texts)
        docs = []
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata else {"id": i}
            doc = Document(
                text=text,
                metadata=meta,
                embedding=self.embedder.embed(text),
            )
            docs.append(doc)
        self.store.add(docs)

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_emb = self.embedder.embed(query)
        return self.store.search(query_emb, top_k)

    def build_prompt(self, query: str, results: list[SearchResult]) -> str:
        context = "\n\n".join(
            f"[Source {r.rank+1}] {r.document.text}" for r in results
        )
        return (
            "Answer the question using ONLY the provided context. "
            "If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def query(self, question: str, top_k: int = 3) -> dict:
        results = self.retrieve(question, top_k)
        prompt = self.build_prompt(question, results)
        return {
            "prompt": prompt,
            "sources": [
                {"text": r.document.text[:100], "score": r.score}
                for r in results
            ],
        }


# --- Function calling demo ---
ROBOT_TOOLS = [
    {
        "name": "get_robot_status",
        "description": "Get current status of a robot",
        "parameters": {"robot_id": "string"},
    },
    {
        "name": "send_robot_command",
        "description": "Send navigation command to a robot",
        "parameters": {"robot_id": "string", "command": "string", "target": "string"},
    },
    {
        "name": "query_error_log",
        "description": "Search robot error logs",
        "parameters": {"robot_id": "string", "time_range": "string", "error_type": "string"},
    },
]

def format_tool_call_prompt(query: str, tools: list[dict]) -> str:
    tools_json = json.dumps(tools, indent=2)
    return (
        f"You have access to these tools:\n{tools_json}\n\n"
        f"User: {query}\n\n"
        "Respond with a JSON tool call if needed, or answer directly.\n"
        "Format: {\"tool\": \"name\", \"args\": {...}}\n"
    )


# --- Demo ---
if __name__ == "__main__":
    # Build RAG system
    knowledge_base = [
        "The sensorbar uses SPI communication at 10MHz. Common failure modes "
        "include stiction from debris and firmware version mismatch.",
        "Battery exchange robots dock using IR alignment sensors. The docking "
        "sequence takes approximately 45 seconds including verification.",
        "NAV_ESTIMATED_STATE_NOT_FINITE indicates the navigation estimator "
        "received NaN values. Check IMU calibration and wheel encoder signals.",
        "The OKS robot uses differential drive with two powered wheels and "
        "two caster wheels. Maximum speed is 1.5 m/s in open areas.",
        "LiDAR-based SLAM provides centimeter-level localization accuracy. "
        "Degradation occurs in large open areas with few features.",
        "The guardian node monitors robot health. It triggers emergency stop "
        "when critical errors exceed threshold within a time window.",
    ]

    embedder = SimpleEmbedder()
    store = VectorStore()
    rag = RAGPipeline(embedder, store)
    rag.index_documents(knowledge_base)

    # Query
    result = rag.query("Why is my robot showing NaN errors in navigation?")
    print("RAG Query Result:")
    for src in result["sources"]:
        print(f"  Score: {src['score']:.3f} | {src['text']}...")
    print(f"\nPrompt length: {len(result['prompt'])} chars")

    # Tool call
    print("\n" + "=" * 60)
    tool_prompt = format_tool_call_prompt(
        "Check the error logs for robot OKS-42 from the last hour",
        ROBOT_TOOLS,
    )
    print(tool_prompt)
```

---

## Exercise (45 min)

### E40.1 — Chunking Comparison (25 min)
Take a 2000-word document and:
1. Chunk with fixed 200-word windows (50-word overlap)
2. Chunk on paragraph boundaries
3. Index both and query — which retrieves more relevant chunks?
4. Implement parent-child: index sentences, return parent paragraphs

### E40.2 — Hybrid Search (20 min)
Combine keyword (BM25) and semantic (embedding) search:
1. Implement a simple BM25 scorer
2. Combine: `score = α * bm25_score + (1-α) * cosine_score`
3. Find the optimal α on 5 test queries — is hybrid better than either alone?

---

## Key Takeaways

1. **RAG = retrieval + generation** — the model answers using retrieved documents
2. **Chunking strategy** is the most important design decision in a RAG system
3. **RAG vs fine-tuning** is a false dichotomy — use both (RAG for dynamic knowledge, fine-tuning for behavior)
4. **Function calling** extends LLMs to interact with external systems — critical for robotics
5. **Embedding quality** directly bounds RAG quality — garbage retrieval → garbage answers

---

## Connection to the Thread

RAG is how you give a robot LLM access to its manuals, past incidents, and fleet telemetry without fine-tuning on every update. Function calling is how an LLM-based planner translates high-level goals ("deliver package to Zone C") into API calls to the robot's navigation stack. This is exactly what we build in the capstone.

---

## Further Reading

- [Retrieval-Augmented Generation for NLP (RAG)](https://arxiv.org/abs/2005.11401) — Lewis et al., 2020
- [Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — Karpukhin et al., 2020
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — Schick et al., 2023
- [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) — Edge et al., 2024
- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) — Evaluation framework
