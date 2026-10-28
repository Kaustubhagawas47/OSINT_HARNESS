import requests
import os
import string
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict
from config import DOMAIN_RELIABILITY_TIERS, DEFAULT_UNKNOWN_DOMAIN_RELIABILITY, ConvergenceConfig
from tavily import TavilyClient

load_dotenv()

_SEARCH_CACHE: Dict[str, List[Dict[str, str]]]={}

def score_domain_reliability(url:str)->float:
    domain= urlparse(url).netloc.lower().replace("www.","")
    if domain in DOMAIN_RELIABILITY_TIERS:
        return DOMAIN_RELIABILITY_TIERS[domain]
    tld= domain.split(".")[-1]
    if tld in DOMAIN_RELIABILITY_TIERS:
        return DOMAIN_RELIABILITY_TIERS[tld]
    return DEFAULT_UNKNOWN_DOMAIN_RELIABILITY

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def fetch_page_text(url: str, max_chars: int = 1000) -> str:
    """Pulls document text using Requests + BeautifulSoup."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (OSINT-Research-Harness/1.0)"}
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return text[:max_chars]
    except Exception:
        return ""  # Return empty string so we don't feed HTTP errors to the LLM


def search_external_sources(query: str, seen_urls: set = None) -> List[Dict[str, str]]:
    cleaned_query = query.strip(string.whitespace + string.punctuation)
    if len(cleaned_query) < 3:
        return []
    if cleaned_query in _SEARCH_CACHE:
        print(f"  [Cache Hit] Returning cached results for: '{cleaned_query}'", flush=True)
        cached_results = _SEARCH_CACHE[cleaned_query]
    else:
        print(f"  [Tavily] Searching: '{cleaned_query}'...", flush=True)
        try:
            response = tavily_client.search(
                query=cleaned_query,
                search_depth="basic",
                max_results=3,
                include_raw_content=True
            )
            
            results = []
            max_chars = ConvergenceConfig.max_scrape_chars
            
            for r in response.get("results", []):
                url = r.get("url", "")
                if not url:
                    continue
                    
                snippet = r.get("content", "")
                raw_text = r.get("raw_content")
                
                if raw_text:
                    full_text = raw_text[:max_chars]
                else:
                    full_text = fetch_page_text(url, max_chars=max_chars)
                    
                combined_content = f"Snippet: {snippet}\nExtracted: {full_text}" if full_text else f"Snippet: {snippet}"
                
                results.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "base_reliability": score_domain_reliability(url),
                    "content": combined_content[:max_chars]
                })
                
            _SEARCH_CACHE[cleaned_query] = results
            cached_results = results
            
        except Exception as e:
            print(f"Search failed for query '{cleaned_query}': {e}")
            return []

    # Apply seen_urls filtering right before returning
    final_results = []
    for res in cached_results:
        url = res["url"]
        if seen_urls is not None and url in seen_urls:
            continue
        if seen_urls is not None:
            seen_urls.add(url)
        final_results.append(res)
        
    return final_results