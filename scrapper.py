# scrape_apple_docs.py
import os
import pickle
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain.schema import Document
from playwright.sync_api import sync_playwright
import re

BASE_URL = "https://support.apple.com"
START_PAGE = "https://support.apple.com/en-us/iphone"
OUTPUT_PATH = "iphone_docs.pkl"

visited = set()
documents = []

def is_support_article(url: str) -> bool:
    # Accept both /en-us/HTxxxx and /en-us/123456 formats
    return bool(re.match(r"^/en-us/(HT)?\d+$", url))

def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def scrape_article(relative_url):
    if not is_support_article(relative_url):
        return
    full_url = urljoin(BASE_URL, relative_url)
    if full_url in visited:
        return
    visited.add(full_url)

    try:
        res = requests.get(full_url, timeout=10)
        res.raise_for_status()
    except Exception as e:
        print(f"Failed: {full_url} – {e}")
        return

    soup = BeautifulSoup(res.text, "html.parser")
    title_tag = soup.find("h1")
    title = title_tag.text.strip() if title_tag else "Untitled"
    content = clean_text(res.text)

    if not content or len(content) < 100:  # avoid junk pages
        print(f"Skipped (too short): {full_url}")
        return

    doc = Document(
        page_content=content,
        metadata={"title": title, "url": full_url}
    )
    documents.append(doc)

def crawl_support_articles(max_articles=200, headless=True):
    global documents
    visited_links = set()
    article_links = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(START_PAGE, timeout=10000)

        print("Extracting subpages to explore...")
        subpages = page.eval_on_selector_all(
            "a[href^='/en-us/']",
            "elements => elements.map(e => e.getAttribute('href'))"
        )
        subpages = list({urljoin(BASE_URL, link) for link in subpages if link.startswith("/en-us/")})
        print(f"Found {len(subpages)} subpages")

        # Crawl subpages for article links
        for subpage_url in subpages:
            if len(article_links) >= max_articles:
                break
            try:
                print(f"Visiting: {subpage_url}")
                page.goto(subpage_url, timeout=10000)
                links = page.eval_on_selector_all(
                    "a[href^='/en-us/']",
                    "elements => elements.map(e => e.getAttribute('href'))"
                )
                for link in links:
                    if is_support_article(link):
                        article_links.add(link)
            except Exception as e:
                print(f"Failed to load subpage {subpage_url}: {e}")

        browser.close()

    print(f"Found {len(article_links)} unique article links")
    for link in list(article_links)[:max_articles]:
        print(f"Scraping: {link}")
        scrape_article(link)

    print(f"Scraped {len(documents)} articles.")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(documents, f)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    crawl_support_articles(headless=True)  # Set to False to debug visually
