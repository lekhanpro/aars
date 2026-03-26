# AARS API Reference

## Query Endpoint

### `POST /api/v1/query`

Execute an adaptive retrieval query.

**Request Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | The user query (1-2000 chars) |
| `collection` | string | `"default"` | Document collection to search |
| `top_k` | integer | `5` | Number of results (1-50) |
| `enable_planner` | boolean | `true` | Enable LLM-based strategy selection |
| `enable_reflection` | boolean | `true` | Enable retrieval sufficiency evaluation |
| `enable_fusion` | boolean | `true` | Enable reciprocal-rank fusion |
| `enable_mmr` | boolean | `true` | Enable diversity reranking |
| `enable_keyword` | boolean | `true` | Allow BM25 keyword retrieval |
| `enable_graph` | boolean | `true` | Allow graph traversal retrieval |
| `default_strategy` | string | `"vector"` | Fallback strategy when planner is off |
| `enable_trace` | boolean | `true` | Include execution trace in response |

**Response:**

```json
{
  "answer": "string",
  "confidence": 0.92,
  "retrieval_plan": {
    "query_type": "factual|analytical|multi_hop|opinion|conversational",
    "complexity": "simple|moderate|complex",
    "strategy": "keyword|vector|graph|hybrid|none",
    "rewritten_query": "string",
    "decomposed_queries": [],
    "reasoning": "string"
  },
  "reflection_results": [...],
  "documents": [...],
  "citations": [...],
  "trace": {...}
}
```

## Ingest Endpoint

### `POST /api/v1/ingest`

Upload and index a document.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Document file (PDF, TXT, MD, images, video) |
| `collection` | string | Target collection name |
| `chunk_size` | integer | Optional chunk size override |
| `chunk_overlap` | integer | Optional chunk overlap override |

**Supported formats:** `.pdf`, `.txt`, `.md`, `.rst`, `.csv`, `.log`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`, `.tiff`, `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

## Health Endpoint

### `GET /api/v1/health`

Returns API and ChromaDB connectivity status.

### `GET /api/v1/collections`

List all available document collections.

### `DELETE /api/v1/collections/{name}`

Delete a collection and its documents.

## Debug Endpoint

### `GET /api/v1/debug/trace/{trace_id}`

Fetch a stored pipeline execution trace.

## Retrieval Strategies

| Strategy | When Used | Description |
|----------|-----------|-------------|
| `keyword` | Factual queries with strong lexical cues | BM25 sparse retrieval |
| `vector` | Semantic or paraphrased queries | Dense embedding similarity via ChromaDB |
| `graph` | Multi-hop or entity-relationship queries | NetworkX graph traversal with BFS |
| `hybrid` | Mixed or complex queries | All strategies + RRF fusion + MMR |
| `none` | Conversational or off-topic queries | Direct LLM response without retrieval |

## Configuration

All settings are configurable via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Anthropic API key for LLM calls |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8001` | ChromaDB port |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | LLM model ID |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `512` | Default chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Default chunk overlap |
| `RRF_K` | `60` | RRF fusion constant |
| `MMR_LAMBDA` | `0.5` | MMR diversity-relevance tradeoff |
