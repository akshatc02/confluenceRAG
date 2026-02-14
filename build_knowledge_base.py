#!/usr/bin/env python3
"""
Complete setup script for Confluence RAG system
Runs the full pipeline: crawl → chunk → embed → store
"""
import json
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from confluence_crawler_improved import ImprovedConfluenceCrawler
from document_splitter_improved import split_documents
from vector_store_hybrid import HybridVectorStore
from config import (
    BASE_URL, SPACE_KEY, AUTH_TOKEN, ANTHROPIC_API_KEY,
    CONFLUENCE_JSON_PATH, VECTOR_STORE_PATH, BM25_INDEX_PATH
)


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def validate_config():
    """Validate configuration before starting"""
    print_header("Validating Configuration")
    
    issues = []
    
    if not BASE_URL:
        issues.append("❌ BASE_URL not set")
    else:
        print(f"✅ Confluence URL: {BASE_URL}")
    
    if not SPACE_KEY:
        issues.append("❌ SPACE_KEY not set")
    else:
        print(f"✅ Space Key: {SPACE_KEY}")
    
    if not AUTH_TOKEN:
        issues.append("❌ CONFLUENCE_TOKEN not set in environment")
    else:
        print(f"✅ Confluence token: {'*' * 20}")
    
    if not ANTHROPIC_API_KEY:
        issues.append("❌ ANTHROPIC_API_KEY not set in environment")
    else:
        print(f"✅ Anthropic API key: {'*' * 20}")
    
    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   {issue}")
        print("\nPlease check your .env file and config.py")
        return False
    
    print("\n✅ Configuration validated successfully!")
    return True


def crawl_confluence():
    """Step 1: Crawl Confluence space"""
    print_header("Step 1: Crawling Confluence Space")
    
    try:
        crawler = ImprovedConfluenceCrawler(BASE_URL, SPACE_KEY, AUTH_TOKEN)
        
        # Get space name
        space_name = crawler.get_space_name(SPACE_KEY)
        print(f"Space: {space_name} ({SPACE_KEY})")
        
        # Get root page
        print("\nEnter the title of the root page to start crawling from:")
        print("(Press Enter to crawl entire space)")
        root_title = input("> ").strip()
        
        if root_title:
            print(f"\nCrawling from: {root_title}")
            root_id = crawler.get_page_id_from_title(root_title)
            pages = crawler.crawl_page(root_id)
        else:
            print("\n⚠️  Crawling entire space (this may take a while)...")
            # Get all pages in space
            # This is simplified - you'd implement get_all_pages_in_space
            print("ERROR: Full space crawling not implemented.")
            print("Please specify a root page title.")
            return None
        
        # Save to JSON
        print(f"\nSaving {len(pages)} pages to {CONFLUENCE_JSON_PATH}")
        crawler.save_to_json(pages, CONFLUENCE_JSON_PATH)
        
        print(f"\n✅ Successfully crawled {len(pages)} pages!")
        
        # Show some stats
        spaces = set(p.space_name for p in pages)
        has_tables = sum(1 for p in pages if p.has_tables)
        has_code = sum(1 for p in pages if p.has_code)
        has_images = sum(1 for p in pages if p.has_images)
        
        print(f"\n📊 Statistics:")
        print(f"   • Total pages: {len(pages)}")
        print(f"   • Spaces: {', '.join(spaces)}")
        print(f"   • Pages with tables: {has_tables}")
        print(f"   • Pages with code: {has_code}")
        print(f"   • Pages with images: {has_images}")
        
        return pages
    
    except Exception as e:
        print(f"\n❌ Error crawling Confluence: {e}")
        import traceback
        traceback.print_exc()
        return None


def chunk_documents(pages):
    """Step 2: Split pages into semantic chunks"""
    print_header("Step 2: Chunking Documents")
    
    try:
        # Convert pages to dicts if they're objects
        pages_data = [p.to_dict() if hasattr(p, 'to_dict') else p for p in pages]
        
        print(f"Splitting {len(pages_data)} pages into semantic chunks...")
        print("Configuration:")
        print(f"   • Max chunk size: 2000 characters")
        print(f"   • Chunk overlap: 200 characters")
        print(f"   • Preserving: tables, code blocks, lists, sections\n")
        
        documents = split_documents(
            pages_data,
            max_chunk_size=2000,
            chunk_overlap=200
        )
        
        print(f"\n✅ Created {len(documents)} semantic chunks!")
        
        # Show chunk type distribution
        from collections import Counter
        chunk_types = Counter(doc.metadata['chunk_type'] for doc in documents)
        
        print(f"\n📊 Chunk Type Distribution:")
        for chunk_type, count in chunk_types.most_common():
            print(f"   • {chunk_type}: {count}")
        
        # Show example chunks
        print(f"\n📝 Example Chunks:")
        for i, doc in enumerate(documents[:3]):
            print(f"\n   Chunk {i+1}:")
            print(f"   Type: {doc.metadata['chunk_type']}")
            print(f"   Page: {doc.metadata['page_title']}")
            print(f"   Section: {' > '.join(doc.metadata.get('section_hierarchy', []))}")
            print(f"   Preview: {doc.page_content[:100]}...")
        
        return documents
    
    except Exception as e:
        print(f"\n❌ Error chunking documents: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_vector_store(documents):
    """Step 3: Build hybrid vector store"""
    print_header("Step 3: Building Hybrid Vector Store")
    
    try:
        print("Initializing hybrid vector store...")
        print(f"   • Embedding model: sentence-transformers/all-mpnet-base-v2")
        print(f"   • Vector store: ChromaDB")
        print(f"   • Keyword index: BM25")
        print(f"   • Storage location: {VECTOR_STORE_PATH}\n")
        
        hybrid_store = HybridVectorStore(
            embedding_model_name="sentence-transformers/all-mpnet-base-v2",
            persist_directory=VECTOR_STORE_PATH
        )
        
        print(f"Adding {len(documents)} documents to store...")
        print("(This may take several minutes depending on corpus size)\n")
        
        hybrid_store.add_documents(documents)
        
        print(f"\nSaving BM25 index to {BM25_INDEX_PATH}")
        hybrid_store.save_bm25(BM25_INDEX_PATH)
        
        print(f"\n✅ Hybrid vector store built successfully!")
        
        return hybrid_store
    
    except Exception as e:
        print(f"\n❌ Error building vector store: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_retrieval(vector_store):
    """Step 4: Test the system with sample queries"""
    print_header("Step 4: Testing Retrieval System")
    
    test_queries = [
        "What is our PTO policy?",
        "How do I deploy to production?",
        "Show me API authentication examples"
    ]
    
    print("Running test queries to validate setup...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"Test Query {i}: {query}")
        print('─'*70)
        
        try:
            # Test hybrid search
            results = vector_store.search(
                query=query,
                k=3,
                method='hybrid'
            )
            
            print(f"\nTop 3 Results:")
            for j, doc in enumerate(results, 1):
                print(f"\n   {j}. {doc.metadata.get('page_title', 'Unknown')}")
                print(f"      Section: {' > '.join(doc.metadata.get('section_hierarchy', []))}")
                print(f"      Type: {doc.metadata.get('chunk_type', 'unknown')}")
                print(f"      Preview: {doc.page_content[:100]}...")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n{'─'*70}")
    print("\n✅ Test queries completed!")


def main():
    """Run the complete setup pipeline"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "Confluence RAG Setup" + " "*28 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Step 0: Validate configuration
    if not validate_config():
        sys.exit(1)
    
    # Ask user what to do
    print("\n" + "─"*70)
    print("Select build mode:")
    print("  1. Full build (crawl + chunk + embed)")
    print("  2. Skip crawling (use existing confluence_pages.json)")
    print("  3. Skip crawling and chunking (rebuild index only)")
    print("─"*70)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    pages = None
    documents = None
    
    if choice == "1":
        # Full build
        pages = crawl_confluence()
        if not pages:
            print("\n❌ Build failed at crawling stage")
            sys.exit(1)
        
        documents = chunk_documents(pages)
        if not documents:
            print("\n❌ Build failed at chunking stage")
            sys.exit(1)
    
    elif choice == "2":
        # Skip crawling
        if not os.path.exists(CONFLUENCE_JSON_PATH):
            print(f"\n❌ Error: {CONFLUENCE_JSON_PATH} not found")
            print("Please run with option 1 first to crawl pages")
            sys.exit(1)
        
        print_header("Loading Existing Pages")
        with open(CONFLUENCE_JSON_PATH, 'r', encoding='utf-8') as f:
            pages_data = json.load(f)
        print(f"✅ Loaded {len(pages_data)} pages from {CONFLUENCE_JSON_PATH}")
        
        documents = chunk_documents(pages_data)
        if not documents:
            print("\n❌ Build failed at chunking stage")
            sys.exit(1)
    
    elif choice == "3":
        # Skip crawling and chunking
        print("\n⚠️  Mode 3 not fully implemented")
        print("Please ensure you have pre-chunked documents")
        sys.exit(1)
    
    else:
        print(f"\n❌ Invalid choice: {choice}")
        sys.exit(1)
    
    # Build vector store
    vector_store = build_vector_store(documents)
    if not vector_store:
        print("\n❌ Build failed at vector store stage")
        sys.exit(1)
    
    # Test the system
    print("\n" + "─"*70)
    print("Would you like to test the system with sample queries? (y/n)")
    test_choice = input("> ").strip().lower()
    
    if test_choice in ['y', 'yes']:
        test_retrieval(vector_store)
    
    # Success!
    print_header("🎉 Setup Complete!")
    print("Your Confluence RAG system is ready!")
    print("\nNext steps:")
    print("  1. Run 'python rag_pipeline.py' to start the chatbot")
    print("  2. Or integrate into your application using the ConfluenceRAG class")
    print("\nFiles created:")
    print(f"  • {CONFLUENCE_JSON_PATH} - Crawled pages")
    print(f"  • {VECTOR_STORE_PATH}/ - Vector store")
    print(f"  • {BM25_INDEX_PATH} - BM25 keyword index")
    print("\n" + "─"*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
