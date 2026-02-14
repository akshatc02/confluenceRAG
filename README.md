# Production-Ready Confluence RAG System

A sophisticated RAG (Retrieval-Augmented Generation) system for building an effective Confluence chatbot using Claude Sonnet 4.

## 🎯 Key Features

- **Semantic-Aware Chunking**: Preserves tables, code blocks, lists, and section structure
- **Hybrid Search**: Combines vector similarity (semantic) + BM25 (keyword) search
- **Rich Metadata**: Tracks space, hierarchy, content type, dates, labels
- **Query Enhancement**: Uses Claude to expand and clarify queries before retrieval
- **Reranking**: Optional LLM-based reranking for improved precision
- **Context-Aware Generation**: Provides citations and preserves document hierarchy

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Query Enhancement (Claude)                        │
│  • Expand acronyms                                          │
│  • Identify intent                                          │
│  • Generate alternative phrasings                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               Hybrid Retrieval                               │
│  ┌──────────────────┐    ┌──────────────────┐             │
│  │  Vector Search   │    │   BM25 Search    │             │
│  │  (Semantic)      │    │   (Keyword)      │             │
│  └────────┬─────────┘    └────────┬─────────┘             │
│           └───────────┬────────────┘                        │
│                       │                                     │
│           Metadata Filtering (space, date, labels)         │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Reranking (Optional, Claude)                        │
│  • Score relevance of each document                         │
│  • Select top K most relevant                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Answer Generation (Claude Sonnet 4)                 │
│  • Synthesize information from retrieved docs               │
│  • Provide citations                                        │
│  • Maintain conversation context                            │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Why This Architecture Works Better

### Problems with Your Original Approach:

1. **Naive text splitting** → destroys semantic boundaries
   - Tables split mid-row → useless
   - Code blocks fragmented → broken examples
   - Headings separated from content → no context

2. **Loss of metadata** → can't filter effectively
   - No space information → can't narrow by team
   - No hierarchy → can't understand context
   - No content type tags → can't prioritize tables/code

3. **Vector-only search** → misses exact matches
   - "PTO policy" might not embed near "time off policy"
   - Product codes, IDs, specific terms get lost

4. **FAISS limitations** → poor production experience
   - No metadata filtering
   - No hybrid search
   - Difficult to update

### Our Improvements:

1. **Semantic chunking** → preserves structure
   - Tables kept whole with markdown formatting
   - Code blocks preserved with language tags
   - Sections maintain heading hierarchy

2. **Rich metadata** → enables powerful filtering
   - Filter by space, date, author, labels
   - Track content type (table, code, paragraph)
   - Preserve document hierarchy

3. **Hybrid search** → best of both worlds
   - Semantic for conceptual queries
   - Keyword for exact terms
   - Combined scoring with weights

4. **Production-ready stores** → Chroma + BM25 (or Pinecone)
   - Easy metadata filtering
   - Built-in hybrid search
   - Simple updates

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**:
```
anthropic>=0.18.0
langchain>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.22
sentence-transformers>=2.3.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
rank-bm25>=0.2.2
requests>=2.31.0
python-dotenv>=1.0.0
```

### 2. Configuration

Create `config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Confluence settings
BASE_URL = "https://your-domain.atlassian.net/wiki"
SPACE_KEY = "YOUR_SPACE"
AUTH_TOKEN = os.getenv("CONFLUENCE_TOKEN")

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Paths
CONFLUENCE_JSON_PATH = "./confluence_pages.json"
VECTOR_STORE_PATH = "./chroma_confluence"
BM25_INDEX_PATH = "./bm25_confluence.pkl"
```

Create `.env` file:
```bash
CONFLUENCE_TOKEN=your_confluence_token
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 3. Build the Knowledge Base

```bash
# Step 1: Crawl Confluence pages
python confluence_crawler_improved.py

# Step 2: Process and chunk documents
python build_knowledge_base.py
```

**build_knowledge_base.py**:
```python
import json
from document_splitter_improved import split_documents
from vector_store_hybrid import HybridVectorStore
from config import CONFLUENCE_JSON_PATH, VECTOR_STORE_PATH, BM25_INDEX_PATH

# Load crawled pages
print("Loading crawled pages...")
with open(CONFLUENCE_JSON_PATH, 'r', encoding='utf-8') as f:
    pages = json.load(f)

print(f"Loaded {len(pages)} pages")

# Split into semantic chunks
print("\nSplitting documents into semantic chunks...")
documents = split_documents(pages, max_chunk_size=2000, chunk_overlap=200)

print(f"Created {len(documents)} chunks")

# Build hybrid vector store
print("\nBuilding hybrid vector store...")
hybrid_store = HybridVectorStore(
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    persist_directory=VECTOR_STORE_PATH
)

hybrid_store.add_documents(documents)
hybrid_store.save_bm25(BM25_INDEX_PATH)

print("\n✅ Knowledge base built successfully!")
```

### 4. Run the Chatbot

```bash
python rag_pipeline.py
```

## 🎨 Advanced Features

### 1. Handle Images in Confluence

Add to `SemanticConfluenceSplitter`:

```python
def _extract_image_descriptions(self, soup: BeautifulSoup, page_data: Dict) -> List[ContentChunk]:
    """Extract and describe images using Claude Vision"""
    chunks = []
    
    for img in soup.find_all('img'):
        img_url = img.get('src', '')
        alt_text = img.get('alt', '')
        
        # Download image
        if img_url.startswith('http'):
            # Use Claude Vision to describe
            image_description = self._describe_image_with_claude(img_url)
            
            chunks.append(ContentChunk(
                text=f"Image: {alt_text}\nDescription: {image_description}",
                chunk_type='image',
                heading_hierarchy=current_hierarchy.copy(),
                metadata={'image_url': img_url}
            ))
    
    return chunks

def _describe_image_with_claude(self, image_url: str) -> str:
    """Use Claude to describe an image"""
    # Download image and convert to base64
    import base64
    import requests
    
    response = requests.get(image_url)
    image_data = base64.b64encode(response.content).decode('utf-8')
    
    # Call Claude Vision
    message = self.claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this image from a Confluence page. What information does it convey?"
                }
            ]
        }]
    )
    
    return message.content[0].text
```

### 2. Table Summarization

Improve table handling with Claude:

```python
def _summarize_table_with_claude(self, table_markdown: str) -> str:
    """Generate a natural language summary of a table"""
    prompt = f"""Summarize this table in 2-3 sentences, highlighting the key information:

{table_markdown}

Focus on what the table shows and any important patterns or values."""

    message = self.claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text
```

Then in `_extract_table_chunk`:
```python
summary = self._summarize_table_with_claude(markdown_table)
full_text = f"{summary}\n\n{markdown_table}"
```

### 3. Periodic Updates

Create `update_knowledge_base.py`:

```python
"""
Incremental update script - only processes changed pages
"""
import json
from datetime import datetime, timedelta
from confluence_crawler_improved import ImprovedConfluenceCrawler
from document_splitter_improved import split_documents
from vector_store_hybrid import HybridVectorStore

def get_recently_updated_pages(crawler, days=7):
    """Get pages updated in last N days"""
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    # Confluence CQL query for recent updates
    url = f"{crawler.base_url}/rest/api/content/search"
    params = {
        'cql': f'space={crawler.space_key} AND lastModified >= "{cutoff_date}"',
        'expand': 'body.storage,version,space,ancestors,metadata.labels'
    }
    
    resp = requests.get(url, headers=crawler.headers, params=params, verify=False)
    resp.raise_for_status()
    
    results = resp.json()['results']
    return [crawler.extract_page_structure(page) for page in results]

# Main update logic
crawler = ImprovedConfluenceCrawler(BASE_URL, SPACE_KEY, AUTH_TOKEN)
updated_pages = get_recently_updated_pages(crawler, days=7)

if updated_pages:
    print(f"Found {len(updated_pages)} updated pages")
    
    # Process updates
    new_documents = split_documents([p.to_dict() for p in updated_pages])
    
    # Delete old versions from vector store
    # (Delete by page_id, then add new versions)
    
    # Add new versions
    vector_store = HybridVectorStore(persist_directory=VECTOR_STORE_PATH)
    vector_store.load_vector_store()
    vector_store.add_documents(new_documents)
    
    print("✅ Knowledge base updated")
else:
    print("No updates needed")
```

Run this daily via cron or scheduler.

### 4. Advanced Metadata Filtering

```python
# Example: Filter by space and date range
from datetime import datetime, timedelta

recent_date = (datetime.now() - timedelta(days=30)).isoformat()

results = rag.retrieve_documents(
    query="deployment process",
    k=5,
    filter_dict={
        "space_key": "ENG",  # Engineering space only
        "last_updated": {"$gte": recent_date}  # Recent docs only
    }
)

# Example: Filter by content type
results = rag.retrieve_documents(
    query="authentication code example",
    k=5,
    filter_dict={
        "has_code": True,  # Only chunks with code blocks
        "chunk_type": "code"
    }
)

# Example: Filter by labels
results = rag.retrieve_documents(
    query="onboarding checklist",
    k=5,
    filter_dict={
        "labels": {"$in": ["onboarding", "new-hire"]}
    }
)
```

## 📈 Performance Optimization

### 1. Embedding Model Selection

For better results, consider:

```python
# For English-only, high quality
embedding_model = "sentence-transformers/all-mpnet-base-v2"  # 768 dims, good balance

# For multilingual support
embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# For very long documents
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims, faster

# For code-heavy content
embedding_model = "microsoft/codebert-base"  # Better for technical docs
```

### 2. Caching

Add caching to reduce API calls:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_retrieve(query_hash, k):
    """Cache retrieval results"""
    # Your retrieval logic here
    pass

def retrieve_with_cache(query, k=5):
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return cached_retrieve(query_hash, k)
```

### 3. Async Processing

For production web apps:

```python
import asyncio
from anthropic import AsyncAnthropic

class AsyncConfluenceRAG:
    def __init__(self, ...):
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
    
    async def query_async(self, user_query: str):
        # Retrieve documents (can be parallelized)
        documents = await asyncio.to_thread(
            self.retrieve_documents, user_query
        )
        
        # Generate answer asynchronously
        result = await self.generate_answer_async(user_query, documents)
        return result
```

## 🔒 Production Deployment

### Using Pinecone (Recommended for Production)

```python
from vector_store_hybrid import PineconeHybridStore

# Initialize Pinecone
vector_store = PineconeHybridStore(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-east1-gcp",
    index_name="confluence-rag"
)

# Add documents
vector_store.add_documents(documents)

# Search with metadata filtering
results = vector_store.search(
    query="deployment guide",
    k=5,
    filter_dict={"space_key": {"$eq": "ENG"}}
)
```

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - CONFLUENCE_TOKEN=${CONFLUENCE_TOKEN}
    volumes:
      - ./chroma_confluence:/app/chroma_confluence
      - ./bm25_confluence.pkl:/app/bm25_confluence.pkl
```

## 📊 Monitoring & Evaluation

### Track Query Performance

```python
import time
from collections import defaultdict

class RAGMetrics:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def log_query(self, query, retrieval_time, generation_time, num_docs):
        self.metrics['queries'].append({
            'query': query,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + generation_time,
            'num_docs': num_docs,
            'timestamp': time.time()
        })
    
    def get_stats(self):
        if not self.metrics['queries']:
            return {}
        
        queries = self.metrics['queries']
        return {
            'total_queries': len(queries),
            'avg_retrieval_time': sum(q['retrieval_time'] for q in queries) / len(queries),
            'avg_generation_time': sum(q['generation_time'] for q in queries) / len(queries),
            'avg_total_time': sum(q['total_time'] for q in queries) / len(queries),
        }
```

### User Feedback Collection

```python
def collect_feedback(query: str, answer: str, helpful: bool, comment: str = ""):
    """Store user feedback for continuous improvement"""
    feedback = {
        'query': query,
        'answer': answer,
        'helpful': helpful,
        'comment': comment,
        'timestamp': datetime.now().isoformat()
    }
    
    # Store in database or file
    with open('feedback.jsonl', 'a') as f:
        f.write(json.dumps(feedback) + '\n')
```

## 🎯 Best Practices Summary

1. **Chunking**: Use semantic boundaries, not fixed sizes
2. **Metadata**: Track everything - space, hierarchy, type, dates
3. **Retrieval**: Hybrid search beats vector-only
4. **Query Enhancement**: Let Claude clarify before searching
5. **Reranking**: Optional but improves precision
6. **Context**: Always prepend page/section info to chunks
7. **Updates**: Implement incremental refresh, not full rebuild
8. **Monitoring**: Track performance and collect feedback
9. **Caching**: Cache embeddings and common queries
10. **Testing**: Build eval set with real user queries

## 🚨 Common Pitfalls to Avoid

1. ❌ Don't split tables or code blocks
2. ❌ Don't discard images without processing
3. ❌ Don't ignore Confluence hierarchy
4. ❌ Don't use vector search alone
5. ❌ Don't skip metadata extraction
6. ❌ Don't forget to handle Confluence macros
7. ❌ Don't rebuild index for every update
8. ❌ Don't ignore user feedback
9. ❌ Don't forget to cite sources
10. ❌ Don't use FAISS for production

## 📚 Further Reading

- [Anthropic Claude Documentation](https://docs.anthropic.com)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Pinecone Hybrid Search](https://docs.pinecone.io/docs/hybrid-search)
- [Confluence REST API](https://developer.atlassian.com/cloud/confluence/rest/v2/intro/)

## 📝 License

MIT License - feel free to use and modify for your needs.
