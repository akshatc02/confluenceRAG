"""
Improved Confluence Crawler with structure-aware extraction
"""
import requests
import json
import time
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re


@dataclass
class ConfluencePage:
    """Structured representation of a Confluence page"""
    page_id: str
    title: str
    space_key: str
    space_name: str
    url: str
    content_html: str
    content_text: str
    last_updated: str
    author: str
    labels: List[str]
    ancestors: List[Dict[str, str]]  # Parent pages
    has_tables: bool
    has_code: bool
    has_images: bool
    
    def to_dict(self):
        return asdict(self)


class ImprovedConfluenceCrawler:
    def __init__(self, base_url: str, space_key: str, auth_token: str):
        self.base_url = base_url.rstrip('/')
        self.space_key = space_key
        self.headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
        
    def get_space_name(self, space_key: str) -> str:
        """Get human-readable space name"""
        url = f"{self.base_url}/rest/api/content"
        params = {'spaceKey': space_key, 'limit': 1}
        resp = requests.get(url, headers=self.headers, params=params, verify=False)
        resp.raise_for_status()
        data = resp.json()
        if data.get('results'):
            return data['results'][0].get('space', {}).get('name', space_key)
        return space_key
    
    def get_page_id_from_title(self, title: str) -> Optional[str]:
        """Get page ID from title"""
        url = f"{self.base_url}/rest/api/content"
        params = {
            "spaceKey": self.space_key,
            "title": title,
            "expand": "version"
        }
        resp = requests.get(url, headers=self.headers, params=params, verify=False)
        resp.raise_for_status()
        data = resp.json()
        if data["results"]:
            return data["results"][0]["id"]
        else:
            raise Exception(f"Page '{title}' not found in space '{self.space_key}'")
    
    def get_page_details(self, page_id: str) -> Dict[str, Any]:
        """Get full page details with all metadata"""
        url = f"{self.base_url}/rest/api/content/{page_id}"
        params = {
            "expand": "body.storage,version,space,ancestors,metadata.labels,history"
        }
        resp = requests.get(url, headers=self.headers, params=params, verify=False)
        resp.raise_for_status()
        return resp.json()
    
    def extract_page_structure(self, page_data: Dict[str, Any]) -> ConfluencePage:
        """Extract structured information from page"""
        page_id = page_data['id']
        title = page_data['title']
        content_html = page_data['body']['storage']['value']
        url = f"{self.base_url}{page_data['_links']['webui']}"
        
        # Extract metadata
        space_key = page_data['space']['key']
        space_name = page_data['space'].get('name', space_key)
        last_updated = page_data['version']['when']
        author = page_data['version']['by'].get('displayName', 'Unknown')
        
        # Extract labels
        labels = [label['name'] for label in page_data.get('metadata', {}).get('labels', {}).get('results', [])]
        
        # Extract ancestor hierarchy
        ancestors = []
        for ancestor in page_data.get('ancestors', []):
            ancestors.append({
                'id': ancestor['id'],
                'title': ancestor['title']
            })
        
        # Analyze content structure
        soup = BeautifulSoup(content_html, 'lxml')
        has_tables = bool(soup.find_all('table'))
        has_code = bool(soup.find_all(['code', 'pre']))
        has_images = bool(soup.find_all('img'))
        
        # Convert to clean text (we'll do better chunking later)
        content_text = self.html_to_text_structured(content_html)
        
        return ConfluencePage(
            page_id=page_id,
            title=title,
            space_key=space_key,
            space_name=space_name,
            url=url,
            content_html=content_html,
            content_text=content_text,
            last_updated=last_updated,
            author=author,
            labels=labels,
            ancestors=ancestors,
            has_tables=has_tables,
            has_code=has_code,
            has_images=has_images
        )
    
    def html_to_text_structured(self, html_content: str) -> str:
        """
        Convert HTML to text while preserving some structure.
        This is just for basic text extraction - we'll do semantic chunking separately.
        """
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, "lxml")
        
        # Remove script and style elements
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        # Get text with line breaks preserved
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text
    
    def crawl_page(self, page_id: str, path: Optional[List[str]] = None, 
                   results: Optional[List[ConfluencePage]] = None) -> List[ConfluencePage]:
        """
        Recursively crawl a page and its children
        """
        if results is None:
            results = []
        if path is None:
            path = []
        
        # Get page details
        page_data = self.get_page_details(page_id)
        
        # Extract structured information
        page = self.extract_page_structure(page_data)
        
        # Add path information (breadcrumb trail)
        page.ancestors = [{'title': p} for p in path]
        
        results.append(page)
        
        # Crawl children
        children_url = f"{self.base_url}/rest/api/content/{page_id}/child/page"
        resp = requests.get(children_url, headers=self.headers, verify=False)
        resp.raise_for_status()
        children_data = resp.json()
        
        if children_data.get('results'):
            new_path = path + [page.title]
            for child in children_data['results']:
                child_id = child['id']
                time.sleep(0.2)  # Rate limiting
                self.crawl_page(child_id, new_path, results)
        
        return results
    
    def save_to_json(self, pages: List[ConfluencePage], filename: str):
        """Save crawled pages to JSON"""
        data = [page.to_dict() for page in pages]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    # Example usage
    from config import BASE_URL, SPACE_KEY, AUTH_TOKEN, CONFLUENCE_JSON_PATH
    
    crawler = ImprovedConfluenceCrawler(BASE_URL, SPACE_KEY, AUTH_TOKEN)
    
    # Start from root page
    root_page_title = "SOC Workstreams"
    root_id = crawler.get_page_id_from_title(root_page_title)
    
    print(f"Starting crawl from: {root_page_title}")
    pages = crawler.crawl_page(root_id)
    
    crawler.save_to_json(pages, CONFLUENCE_JSON_PATH)
    print(f"Crawled {len(pages)} pages and saved to {CONFLUENCE_JSON_PATH}")


if __name__ == "__main__":
    main()
