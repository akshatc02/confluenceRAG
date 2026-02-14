"""
Semantic-aware document splitter for Confluence content.
Intelligently chunks based on structure, not arbitrary character counts.
"""
from typing import List, Dict, Any
from bs4 import BeautifulSoup, NavigableString, Tag
from dataclasses import dataclass
import re
from langchain.schema import Document


@dataclass
class ContentChunk:
    """Represents a semantically meaningful chunk of content"""
    text: str
    chunk_type: str  # 'section', 'table', 'code', 'list', 'paragraph'
    heading_hierarchy: List[str]  # ['H1 Title', 'H2 Subtitle', ...]
    metadata: Dict[str, Any]


class SemanticConfluenceSplitter:
    """
    Splits Confluence HTML content based on semantic structure.
    Preserves tables, code blocks, and section boundaries.
    """
    
    def __init__(self, max_chunk_size: int = 2000, chunk_overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_page(self, page_data: Dict[str, Any]) -> List[Document]:
        """
        Split a Confluence page into semantic chunks.
        
        Args:
            page_data: Dictionary containing page information including 'content_html'
        
        Returns:
            List of LangChain Document objects with rich metadata
        """
        content_html = page_data.get('content_html', '')
        
        if not content_html:
            return []
        
        # Parse HTML
        soup = BeautifulSoup(content_html, 'lxml')
        
        # Extract semantic chunks
        chunks = self._extract_semantic_chunks(soup, page_data)
        
        # Convert to LangChain Documents
        documents = []
        for i, chunk in enumerate(chunks):
            # Build hierarchical context
            hierarchy_text = " > ".join(chunk.heading_hierarchy) if chunk.heading_hierarchy else ""
            
            # Create context-aware chunk text
            chunk_text = self._build_chunk_text(chunk, page_data, hierarchy_text)
            
            # Build metadata
            metadata = {
                'page_id': page_data.get('page_id'),
                'page_title': page_data.get('title'),
                'space_key': page_data.get('space_key'),
                'space_name': page_data.get('space_name'),
                'url': page_data.get('url'),
                'last_updated': page_data.get('last_updated'),
                'author': page_data.get('author'),
                'labels': page_data.get('labels', []),
                'chunk_index': i,
                'chunk_type': chunk.chunk_type,
                'section_hierarchy': chunk.heading_hierarchy,
                'has_table': chunk.chunk_type == 'table',
                'has_code': chunk.chunk_type == 'code',
                **chunk.metadata
            }
            
            # Add ancestor information for breadcrumb context
            if page_data.get('ancestors'):
                ancestor_titles = [a.get('title', '') for a in page_data['ancestors']]
                metadata['ancestor_pages'] = ancestor_titles
            
            documents.append(Document(
                page_content=chunk_text,
                metadata=metadata
            ))
        
        return documents
    
    def _extract_semantic_chunks(self, soup: BeautifulSoup, page_data: Dict[str, Any]) -> List[ContentChunk]:
        """Extract semantic chunks from parsed HTML"""
        chunks = []
        current_hierarchy = []
        
        # Process top-level elements
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table', 'pre', 'ul', 'ol', 'div'], recursive=False):
            
            # Update heading hierarchy
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                heading_text = element.get_text(strip=True)
                
                # Trim hierarchy to current level
                current_hierarchy = current_hierarchy[:level-1]
                current_hierarchy.append(heading_text)
                continue
            
            # Extract tables as complete units
            if element.name == 'table':
                chunk = self._extract_table_chunk(element, current_hierarchy.copy())
                if chunk:
                    chunks.append(chunk)
                continue
            
            # Extract code blocks as complete units
            if element.name == 'pre':
                chunk = self._extract_code_chunk(element, current_hierarchy.copy())
                if chunk:
                    chunks.append(chunk)
                continue
            
            # Extract lists
            if element.name in ['ul', 'ol']:
                chunk = self._extract_list_chunk(element, current_hierarchy.copy())
                if chunk:
                    chunks.append(chunk)
                continue
            
            # Handle paragraphs and divs
            if element.name in ['p', 'div']:
                text = element.get_text(strip=True)
                if text and len(text) > 20:  # Skip very short paragraphs
                    chunks.append(ContentChunk(
                        text=text,
                        chunk_type='paragraph',
                        heading_hierarchy=current_hierarchy.copy(),
                        metadata={}
                    ))
        
        # Merge small consecutive paragraph chunks
        chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    def _extract_table_chunk(self, table_element: Tag, hierarchy: List[str]) -> ContentChunk:
        """Extract table as markdown with description"""
        try:
            # Convert table to markdown
            rows = []
            headers = []
            
            # Extract headers
            header_row = table_element.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                rows.append('| ' + ' | '.join(headers) + ' |')
                rows.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
            
            # Extract data rows
            for tr in table_element.find_all('tr')[1:]:  # Skip header row
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append('| ' + ' | '.join(cells) + ' |')
            
            markdown_table = '\n'.join(rows)
            
            # Create a descriptive summary (you could use Claude here in production)
            summary = f"Table with {len(headers)} columns: {', '.join(headers)}"
            
            full_text = f"{summary}\n\n{markdown_table}"
            
            return ContentChunk(
                text=full_text,
                chunk_type='table',
                heading_hierarchy=hierarchy,
                metadata={
                    'table_columns': headers,
                    'table_rows': len(table_element.find_all('tr')) - 1
                }
            )
        except Exception as e:
            print(f"Error extracting table: {e}")
            return None
    
    def _extract_code_chunk(self, code_element: Tag, hierarchy: List[str]) -> ContentChunk:
        """Extract code block with language detection"""
        code_text = code_element.get_text(strip=True)
        
        # Try to detect language
        language = 'unknown'
        class_attr = code_element.get('class', [])
        for cls in class_attr:
            if 'language-' in cls:
                language = cls.replace('language-', '')
                break
        
        return ContentChunk(
            text=f"```{language}\n{code_text}\n```",
            chunk_type='code',
            heading_hierarchy=hierarchy,
            metadata={
                'code_language': language,
                'code_lines': len(code_text.split('\n'))
            }
        )
    
    def _extract_list_chunk(self, list_element: Tag, hierarchy: List[str]) -> ContentChunk:
        """Extract list items"""
        items = []
        for li in list_element.find_all('li', recursive=False):
            item_text = li.get_text(strip=True)
            if item_text:
                items.append(f"• {item_text}")
        
        if not items:
            return None
        
        return ContentChunk(
            text='\n'.join(items),
            chunk_type='list',
            heading_hierarchy=hierarchy,
            metadata={
                'list_items': len(items)
            }
        )
    
    def _merge_small_chunks(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        """Merge consecutive small paragraph chunks that share the same heading"""
        if not chunks:
            return chunks
        
        merged = []
        current_merge = None
        
        for chunk in chunks:
            # Don't merge tables, code, or lists
            if chunk.chunk_type != 'paragraph':
                if current_merge:
                    merged.append(current_merge)
                    current_merge = None
                merged.append(chunk)
                continue
            
            # Start a new merge group
            if current_merge is None:
                current_merge = chunk
                continue
            
            # Check if we can merge with current group
            same_hierarchy = current_merge.heading_hierarchy == chunk.heading_hierarchy
            combined_length = len(current_merge.text) + len(chunk.text)
            
            if same_hierarchy and combined_length < self.max_chunk_size:
                # Merge
                current_merge.text += "\n\n" + chunk.text
            else:
                # Save current merge and start new one
                merged.append(current_merge)
                current_merge = chunk
        
        # Don't forget the last merge group
        if current_merge:
            merged.append(current_merge)
        
        return merged
    
    def _build_chunk_text(self, chunk: ContentChunk, page_data: Dict[str, Any], 
                         hierarchy_text: str) -> str:
        """
        Build the final chunk text with context prepended.
        This helps the LLM understand WHERE this chunk came from.
        """
        context_parts = []
        
        # Add page context
        context_parts.append(f"Page: {page_data.get('title', 'Unknown')}")
        
        # Add space context
        if page_data.get('space_name'):
            context_parts.append(f"Space: {page_data['space_name']}")
        
        # Add section hierarchy
        if hierarchy_text:
            context_parts.append(f"Section: {hierarchy_text}")
        
        # Add ancestor breadcrumb if available
        if page_data.get('ancestors'):
            ancestor_titles = [a.get('title', '') for a in page_data['ancestors'] if a.get('title')]
            if ancestor_titles:
                context_parts.append(f"Path: {' > '.join(ancestor_titles)} > {page_data.get('title', '')}")
        
        context_header = '\n'.join(context_parts)
        
        # Build final text
        return f"{context_header}\n\n{chunk.text}"


def split_documents(pages_data: List[Dict[str, Any]], 
                    max_chunk_size: int = 2000,
                    chunk_overlap: int = 200) -> List[Document]:
    """
    Split multiple Confluence pages into semantic chunks.
    
    Args:
        pages_data: List of page dictionaries from crawler
        max_chunk_size: Maximum size for paragraph chunks
        chunk_overlap: Overlap for paragraph chunks
    
    Returns:
        List of LangChain Document objects
    """
    splitter = SemanticConfluenceSplitter(max_chunk_size, chunk_overlap)
    
    all_documents = []
    for page_data in pages_data:
        documents = splitter.split_page(page_data)
        all_documents.extend(documents)
    
    return all_documents


if __name__ == "__main__":
    # Example usage
    import json
    
    # Load crawled pages
    with open('confluence_pages.json', 'r', encoding='utf-8') as f:
        pages = json.load(f)
    
    # Split into semantic chunks
    documents = split_documents(pages)
    
    print(f"Created {len(documents)} semantic chunks from {len(pages)} pages")
    
    # Show example
    if documents:
        print("\nExample chunk:")
        print(f"Type: {documents[0].metadata['chunk_type']}")
        print(f"Section: {' > '.join(documents[0].metadata['section_hierarchy'])}")
        print(f"Text preview: {documents[0].page_content[:200]}...")
