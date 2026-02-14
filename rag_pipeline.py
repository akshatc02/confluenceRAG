"""
Complete RAG Pipeline for Confluence Chatbot
Includes query enhancement, retrieval, reranking, and generation with Claude
"""
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import anthropic
import os
from vector_store_hybrid import HybridVectorStore


class QueryEnhancer:
    """Enhances user queries before retrieval using Claude"""
    
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
    
    def enhance_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Enhance query by:
        1. Expanding acronyms
        2. Identifying intent
        3. Extracting key entities
        4. Generating alternative phrasings
        
        Returns:
            Dictionary with enhanced query components
        """
        prompt = f"""Analyze this user query for a Confluence knowledge base search:

Query: "{query}"

Please provide:
1. **Expanded query**: Expand any acronyms and add clarifying terms
2. **Search intent**: What is the user trying to find? (factual_lookup, how_to, policy, troubleshooting, etc.)
3. **Key entities**: Extract important terms, names, or concepts
4. **Alternative phrasings**: 2-3 ways to rephrase this query
5. **Suggested filters**: Any metadata filters that might help (e.g., space, content type)

Format your response as JSON:
{{
    "expanded_query": "...",
    "intent": "...",
    "key_entities": ["...", "..."],
    "alternative_queries": ["...", "..."],
    "suggested_filters": {{"key": "value"}},
    "requires_specific_page": true/false
}}"""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response
        import json
        try:
            response_text = message.content[0].text
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            enhanced = json.loads(response_text)
            return enhanced
        except Exception as e:
            print(f"Error parsing query enhancement: {e}")
            return {
                "expanded_query": query,
                "intent": "unknown",
                "key_entities": [],
                "alternative_queries": [query],
                "suggested_filters": {},
                "requires_specific_page": False
            }


class ConfluenceRAG:
    """Complete RAG system for Confluence chatbot"""
    
    def __init__(self, 
                 vector_store: HybridVectorStore,
                 anthropic_api_key: str,
                 enable_query_enhancement: bool = True):
        """
        Initialize RAG system
        
        Args:
            vector_store: Initialized hybrid vector store
            anthropic_api_key: Anthropic API key for Claude
            enable_query_enhancement: Whether to use query enhancement
        """
        self.vector_store = vector_store
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.enable_query_enhancement = enable_query_enhancement
        
        if enable_query_enhancement:
            self.query_enhancer = QueryEnhancer(anthropic_api_key)
    
    def retrieve_documents(self, 
                          query: str, 
                          k: int = 5,
                          filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve relevant documents with optional query enhancement
        
        Args:
            query: User query
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
        
        Returns:
            List of relevant documents
        """
        # Enhance query if enabled
        if self.enable_query_enhancement:
            enhanced = self.query_enhancer.enhance_query(query)
            
            # Use expanded query for retrieval
            search_query = enhanced['expanded_query']
            
            # Apply suggested filters if no filters provided
            if not filter_dict and enhanced.get('suggested_filters'):
                filter_dict = enhanced['suggested_filters']
            
            print(f"Original query: {query}")
            print(f"Enhanced query: {search_query}")
            print(f"Intent: {enhanced.get('intent')}")
        else:
            search_query = query
        
        # Retrieve documents using hybrid search
        documents = self.vector_store.search(
            query=search_query,
            k=k,
            method='hybrid',
            filter_dict=filter_dict
        )
        
        return documents
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Rerank documents using Claude to find most relevant ones.
        This is optional but improves precision.
        
        Args:
            query: User query
            documents: Retrieved documents
            top_k: Number of top documents to return
        
        Returns:
            Reranked documents
        """
        if len(documents) <= top_k:
            return documents
        
        # Create reranking prompt
        docs_text = ""
        for i, doc in enumerate(documents):
            docs_text += f"\n[Document {i+1}]\n"
            docs_text += f"Title: {doc.metadata.get('page_title', 'Unknown')}\n"
            docs_text += f"Section: {' > '.join(doc.metadata.get('section_hierarchy', []))}\n"
            docs_text += f"Content: {doc.page_content[:300]}...\n"
        
        prompt = f"""Given this user query:
"{query}"

And these candidate documents:
{docs_text}

Rank these documents by relevance to the query. Return only the document numbers of the top {top_k} most relevant documents, in order.
Format: [1, 3, 5] (just the numbers in a list)"""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            # Parse ranking
            response_text = message.content[0].text
            import json
            # Extract list from response
            ranking = json.loads(response_text.strip())
            
            # Reorder documents
            reranked = []
            for idx in ranking[:top_k]:
                if 1 <= idx <= len(documents):
                    reranked.append(documents[idx - 1])
            
            return reranked if reranked else documents[:top_k]
        
        except Exception as e:
            print(f"Error in reranking: {e}")
            return documents[:top_k]
    
    def generate_answer(self, 
                       query: str, 
                       documents: List[Document],
                       conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Generate answer using Claude with retrieved context
        
        Args:
            query: User query
            documents: Retrieved documents
            conversation_history: Previous conversation turns
        
        Returns:
            Dictionary with answer and sources
        """
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"""
[Source {i+1}]
Page: {doc.metadata.get('page_title', 'Unknown')}
URL: {doc.metadata.get('url', 'N/A')}
Section: {' > '.join(doc.metadata.get('section_hierarchy', []))}
Content:
{doc.page_content}
---""")
        
        context = "\n".join(context_parts)
        
        # Build system prompt
        system_prompt = """You are a helpful assistant that answers questions based on information from a Confluence knowledge base.

Your task:
1. Answer the user's question using ONLY the information provided in the context below
2. If the answer cannot be found in the context, say so clearly
3. Cite your sources by referring to [Source N] when making claims
4. Be concise but thorough
5. If there are multiple relevant pieces of information, synthesize them
6. Maintain a professional, helpful tone

Format your response as:
**Answer**: [Your answer here]

**Sources**:
- [Page Title]: [URL]
"""

        # Build user message
        user_message = f"""Context from knowledge base:
{context}

User question: {query}

Please provide a helpful answer based on the context above."""

        # Add conversation history if provided
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system_prompt,
            messages=messages
        )
        
        answer_text = message.content[0].text
        
        # Build sources list
        sources = []
        for doc in documents:
            sources.append({
                'page_title': doc.metadata.get('page_title', 'Unknown'),
                'url': doc.metadata.get('url', ''),
                'section': ' > '.join(doc.metadata.get('section_hierarchy', [])),
                'space': doc.metadata.get('space_name', ''),
            })
        
        return {
            'answer': answer_text,
            'sources': sources,
            'documents': documents  # Include full documents for reference
        }
    
    def query(self, 
             user_query: str,
             k: int = 5,
             rerank: bool = True,
             conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Main query interface - complete RAG pipeline
        
        Args:
            user_query: User's question
            k: Number of documents to retrieve
            rerank: Whether to rerank documents
            conversation_history: Previous conversation context
        
        Returns:
            Dictionary with answer and sources
        """
        print(f"\n{'='*60}")
        print(f"Query: {user_query}")
        print(f"{'='*60}")
        
        # Step 1: Retrieve documents
        print("\n[1/3] Retrieving relevant documents...")
        documents = self.retrieve_documents(user_query, k=k)
        print(f"Retrieved {len(documents)} documents")
        
        # Step 2: Rerank if enabled
        if rerank and len(documents) > 3:
            print("\n[2/3] Reranking documents...")
            documents = self.rerank_documents(user_query, documents, top_k=3)
            print(f"Reranked to top {len(documents)} documents")
        else:
            print("\n[2/3] Skipping reranking")
        
        # Step 3: Generate answer
        print("\n[3/3] Generating answer with Claude...")
        result = self.generate_answer(user_query, documents, conversation_history)
        
        return result


def main():
    """Example usage of the complete RAG system"""
    import json
    
    # Load environment variables
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    if not ANTHROPIC_API_KEY:
        raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
    
    # Initialize vector store (assumes it's already built)
    vector_store = HybridVectorStore(
        persist_directory="./chroma_confluence"
    )
    vector_store.load_vector_store()
    vector_store.load_bm25("./bm25_confluence.pkl")
    
    # Initialize RAG system
    rag = ConfluenceRAG(
        vector_store=vector_store,
        anthropic_api_key=ANTHROPIC_API_KEY,
        enable_query_enhancement=True
    )
    
    # Interactive query loop
    print("\n" + "="*60)
    print("Confluence Chatbot Ready!")
    print("="*60)
    print("Type your questions below (or 'quit' to exit)\n")
    
    conversation_history = []
    
    while True:
        user_query = input("\nYou: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        try:
            # Query the RAG system
            result = rag.query(
                user_query=user_query,
                k=8,  # Retrieve more, then rerank
                rerank=True,
                conversation_history=conversation_history[-4:]  # Last 2 turns
            )
            
            # Print answer
            print(f"\nAssistant: {result['answer']}")
            
            # Update conversation history
            conversation_history.append({
                "role": "user",
                "content": user_query
            })
            conversation_history.append({
                "role": "assistant",
                "content": result['answer']
            })
        
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
