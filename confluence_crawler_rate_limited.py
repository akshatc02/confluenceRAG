"""
Rate-Limited Confluence Crawler
Handles 429 errors with exponential backoff and configurable rate limiting
"""
import requests
import json
import time
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re
import os
from datetime import datetime


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
    ancestors: List[Dict[str, str]]
    has_tables: bool
    has_code: bool
    has_images: bool
    
    def to_dict(self):
        return asdict(self)


class RateLimitedConfluenceCrawler:
    def __init__(self, base_url: str, space_key: str, auth_token: str,
                 delay_between_requests: float = 1.0,
                 max_retries: int = 5):
        """
        Initialize crawler with rate limiting
        
        Args:
            base_url: Confluence base URL
            space_key: Space to crawl
            auth_token: API token
            delay_between_requests: Seconds to wait between requests (default: 1.0)
            max_retries: Maximum retry attempts for rate limit errors (default: 5)
        """
        self.base_url = base_url.rstrip('/')
        self.space_key = space_key
        self.headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
        self.delay_between_requests = delay_between_requests
        self.max_retries = max_retries
        self.request_count = 0
        self.start_time = time.time()
        
        print(f"\n🔍 Testing connection to: {self.base_url}")
        print(f"⏱️  Rate limit: {delay_between_requests}s between requests")
        
        try:
            test_url = f"{self.base_url}/rest/api/content"
            test_params = {'spaceKey': space_key, 'limit': 1}
            resp = self._make_request_with_retry(test_url, params=test_params)
            print(f"✅ Connection successful!")
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection failed: {e}")
            raise
    
    def _make_request_with_retry(self, url: str, params: Dict = None, 
                                 method: str = 'GET') -> requests.Response:
        """
        Make HTTP request with exponential backoff retry for rate limits
        
        Args:
            url: Request URL
            params: Query parameters
            method: HTTP method (GET, POST, etc.)
        
        Returns:
            Response object
        """
        retry_count = 0
        base_delay = 2  # Start with 2 second delay
        
        while retry_count < self.max_retries:
            try:
                # Add delay between requests to avoid rate limits
                if self.request_count > 0:
                    time.sleep(self.delay_between_requests)
                
                # Make request
                if method.upper() == 'GET':
                    resp = requests.get(url, headers=self.headers, params=params, verify=False)
                else:
                    resp = requests.post(url, headers=self.headers, json=params, verify=False)
                
                self.request_count += 1
                
                # Handle rate limiting (429)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get('Retry-After', base_delay * (2 ** retry_count)))
                    
                    print(f"\n⚠️  Rate limit hit (429) - Request #{self.request_count}")
                    print(f"   Waiting {retry_after} seconds before retry {retry_count + 1}/{self.max_retries}...")
                    
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                
                # Raise for other errors
                resp.raise_for_status()
                
                # Success!
                return resp
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # This shouldn't happen (we handle above) but just in case
                    retry_count += 1
                    wait_time = base_delay * (2 ** retry_count)
                    print(f"⚠️  Rate limit error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Other HTTP error, don't retry
                    raise
            except requests.exceptions.RequestException as e:
                # Network error, retry with backoff
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise
                wait_time = base_delay * (2 ** retry_count)
                print(f"⚠️  Request error: {e}")
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        # Max retries exceeded
        raise Exception(f"Max retries ({self.max_retries}) exceeded for URL: {url}")
    
    def get_stats(self):
        """Get crawling statistics"""
        elapsed = time.time() - self.start_time
        rate = self.request_count / elapsed if elapsed > 0 else 0
        return {
            'requests': self.request_count,
            'elapsed_seconds': round(elapsed, 2),
            'requests_per_second': round(rate, 2)
        }
    
    def get_space_name(self, space_key: str) -> str:
        """Get human-readable space name"""
        try:
            url = f"{self.base_url}/rest/api/content"
            params = {'spaceKey': space_key, 'limit': 1}
            resp = self._make_request_with_retry(url, params=params)
            data = resp.json()
            if data.get('results'):
                return data['results'][0].get('space', {}).get('name', space_key)
            return space_key
        except Exception as e:
            print(f"Warning: Could not get space name: {e}")
            return space_key
    
    def get_page_id_from_title(self, title: str) -> Optional[str]:
        """Get page ID from title"""
        print(f"\n🔍 Searching for page: '{title}'")
        url = f"{self.base_url}/rest/api/content"
        params = {
            "spaceKey": self.space_key,
            "title": title,
            "expand": "version"
        }
        
        try:
            resp = self._make_request_with_retry(url, params=params)
            data = resp.json()
            
            if data["results"]:
                page_id = data["results"][0]["id"]
                print(f"✅ Found page ID: {page_id}")
                return page_id
            else:
                print(f"❌ Page '{title}' not found in space '{self.space_key}'")
                print("\nAvailable pages in this space:")
                self.list_pages_in_space()
                return None
        except Exception as e:
            print(f"❌ Error searching for page: {e}")
            return None
    
    def list_pages_in_space(self, limit: int = 10):
        """List some pages in the space to help user find the right title"""
        try:
            url = f"{self.base_url}/rest/api/content"
            params = {
                'spaceKey': self.space_key,
                'limit': limit,
                'expand': 'version'
            }
            resp = self._make_request_with_retry(url, params=params)
            data = resp.json()
            
            if data.get('results'):
                print(f"\nFirst {limit} pages in space '{self.space_key}':")
                for i, page in enumerate(data['results'], 1):
                    print(f"  {i}. {page['title']}")
            else:
                print(f"No pages found in space '{self.space_key}'")
        except Exception as e:
            print(f"Could not list pages: {e}")
    
    def get_page_details(self, page_id: str) -> Dict[str, Any]:
        """Get full page details with all metadata"""
        url = f"{self.base_url}/rest/api/content/{page_id}"
        params = {
            "expand": "body.storage,version,space,ancestors,metadata.labels,history"
        }
        
        try:
            resp = self._make_request_with_retry(url, params=params)
            return resp.json()
        except Exception as e:
            print(f"❌ Error getting page details for ID {page_id}: {e}")
            raise
    
    def extract_page_structure(self, page_data: Dict[str, Any]) -> ConfluencePage:
        """Extract structured information from page"""
        page_id = page_data['id']
        title = page_data['title']
        content_html = page_data.get('body', {}).get('storage', {}).get('value', '')
        url = f"{self.base_url}{page_data['_links']['webui']}"
        
        # Extract metadata
        space_key = page_data['space']['key']
        space_name = page_data['space'].get('name', space_key)
        last_updated = page_data['version']['when']
        author = page_data['version']['by'].get('displayName', 'Unknown')
        
        # Extract labels
        labels = []
        try:
            label_results = page_data.get('metadata', {}).get('labels', {}).get('results', [])
            labels = [label['name'] for label in label_results]
        except:
            pass
        
        # Extract ancestor hierarchy
        ancestors = []
        try:
            for ancestor in page_data.get('ancestors', []):
                ancestors.append({
                    'id': ancestor['id'],
                    'title': ancestor['title']
                })
        except:
            pass
        
        # Analyze content structure
        has_tables = False
        has_code = False
        has_images = False
        
        if content_html:
            try:
                soup = BeautifulSoup(content_html, 'lxml')
                has_tables = bool(soup.find_all('table'))
                has_code = bool(soup.find_all(['code', 'pre']))
                has_images = bool(soup.find_all('img'))
            except:
                pass
        
        # Convert to clean text
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
        """Convert HTML to text while preserving some structure"""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, "lxml")
            
            # Remove script and style elements
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            # Get text with line breaks preserved
            text = soup.get_text(separator="\n", strip=True)
            
            # Clean up excessive whitespace
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            
            return text
        except Exception as e:
            print(f"Warning: Error converting HTML to text: {e}")
            return html_content
    
    def get_child_pages(self, page_id: str) -> List[Dict[str, str]]:
        """Get list of child pages (non-recursive)"""
        try:
            children_url = f"{self.base_url}/rest/api/content/{page_id}/child/page"
            resp = self._make_request_with_retry(children_url)
            children_data = resp.json()
            
            children = []
            if children_data.get('results'):
                for child in children_data['results']:
                    children.append({
                        'id': child['id'],
                        'title': child['title']
                    })
            
            return children
        except Exception as e:
            print(f"   ⚠️  Error getting children: {e}")
            return []
    
    def crawl_page(self, page_id: str, path: Optional[List[str]] = None, 
                   results: Optional[List[ConfluencePage]] = None,
                   max_depth: int = 10) -> List[ConfluencePage]:
        """
        Recursively crawl a page and its children with rate limiting
        
        Args:
            page_id: ID of the page to crawl
            path: Current path (for breadcrumbs)
            results: Accumulated results
            max_depth: Maximum depth to crawl
        """
        if results is None:
            results = []
        if path is None:
            path = []
        
        # Check depth limit
        if len(path) >= max_depth:
            print(f"⚠️  Reached maximum depth ({max_depth}), stopping crawl")
            return results
        
        try:
            # Get page details
            indent = "  " * len(path)
            print(f"{indent}📄 Crawling: {' > '.join(path) if path else 'Root'}")
            
            page_data = self.get_page_details(page_id)
            
            # Extract structured information
            page = self.extract_page_structure(page_data)
            
            # Add path information (breadcrumb trail)
            page.ancestors = [{'title': p} for p in path]
            
            results.append(page)
            print(f"{indent}   ✅ {page.title}")
            
            # Show progress
            stats = self.get_stats()
            print(f"{indent}   📊 Progress: {len(results)} pages, {stats['requests']} requests, {stats['requests_per_second']:.2f} req/s")
            
            # Get children
            children = self.get_child_pages(page_id)
            
            if children:
                print(f"{indent}   📂 Found {len(children)} child pages")
                new_path = path + [page.title]
                
                for i, child in enumerate(children, 1):
                    child_id = child['id']
                    print(f"{indent}   ├─ Child {i}/{len(children)}: {child['title']}")
                    self.crawl_page(child_id, new_path, results, max_depth)
            
        except Exception as e:
            print(f"{indent}   ❌ Error crawling page {page_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def save_to_json(self, pages: List[ConfluencePage], filename: str):
        """Save crawled pages to JSON"""
        data = [page.to_dict() for page in pages]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved {len(pages)} pages to: {filename}")


def main():
    """Interactive main function with rate limiting"""
    print("\n" + "="*70)
    print("  Rate-Limited Confluence Crawler")
    print("  (Handles 429 errors automatically)")
    print("="*70 + "\n")
    
    # Get configuration
    print("📋 Configuration Setup\n")
    
    base_url = os.getenv('CONFLUENCE_URL') or input("Confluence URL: ").strip()
    space_key = os.getenv('CONFLUENCE_SPACE') or input("Space Key: ").strip()
    auth_token = os.getenv('CONFLUENCE_TOKEN') or input("API Token: ").strip()
    
    # Ask about rate limiting
    print("\n⏱️  Rate Limiting Configuration")
    print("   Confluence Cloud typically allows ~100 requests/minute")
    print("   Recommended: 1.0 second delay (60 requests/minute)")
    
    delay_input = input("Delay between requests in seconds [default: 1.0]: ").strip()
    delay = float(delay_input) if delay_input else 1.0
    
    if not base_url or not space_key or not auth_token:
        print("\n❌ Error: Missing required configuration")
        return
    
    try:
        # Initialize crawler with rate limiting
        crawler = RateLimitedConfluenceCrawler(
            base_url=base_url,
            space_key=space_key,
            auth_token=auth_token,
            delay_between_requests=delay,
            max_retries=5
        )
        
        # Get space name
        space_name = crawler.get_space_name(space_key)
        print(f"\n📚 Space: {space_name} ({space_key})")
        
        # Get root page
        print("\n" + "-"*70)
        root_title = input("Root page title (or Enter to list pages): ").strip()
        
        if not root_title:
            crawler.list_pages_in_space(limit=20)
            print("\nPlease run again and enter one of the page titles above.")
            return
        
        root_id = crawler.get_page_id_from_title(root_title)
        if not root_id:
            return
        
        # Ask for max depth
        depth_input = input("\nMax depth [default: 10]: ").strip()
        max_depth = int(depth_input) if depth_input.isdigit() else 10
        
        # Crawl pages
        print(f"\n🚀 Starting crawl")
        print(f"   Root: {root_title}")
        print(f"   Max depth: {max_depth}")
        print(f"   Rate limit: {delay}s between requests")
        print("-"*70 + "\n")
        
        start_time = time.time()
        pages = crawler.crawl_page(root_id, max_depth=max_depth)
        elapsed = time.time() - start_time
        
        if not pages:
            print("\n⚠️  No pages were crawled")
            return
        
        # Show statistics
        stats = crawler.get_stats()
        print("\n" + "="*70)
        print(f"  Crawl Complete!")
        print("="*70)
        print(f"\n📊 Statistics:")
        print(f"   • Total pages crawled: {len(pages)}")
        print(f"   • Total requests: {stats['requests']}")
        print(f"   • Time elapsed: {stats['elapsed_seconds']}s")
        print(f"   • Average rate: {stats['requests_per_second']:.2f} requests/second")
        print(f"   • Pages with tables: {sum(1 for p in pages if p.has_tables)}")
        print(f"   • Pages with code: {sum(1 for p in pages if p.has_code)}")
        print(f"   • Pages with images: {sum(1 for p in pages if p.has_images)}")
        
        # Save to JSON
        output_file = input(f"\nSave to file [confluence_pages.json]: ").strip() or "confluence_pages.json"
        crawler.save_to_json(pages, output_file)
        
        print("\n✅ Done! You can now use this JSON file with the document splitter.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Crawl interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Suppress SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    main()
