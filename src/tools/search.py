"""Web search tools for the Researcher agent."""

from langchain_core.tools import tool
from src.config import settings


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information on a topic.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).

    Returns:
        Formatted search results with titles, URLs, and snippets.
    """
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=query, max_results=max_results)

    results = []
    for item in response.get("results", []):
        results.append(
            f"**{item['title']}**\nURL: {item['url']}\nContent: {item['content']}\n"
        )

    return "\n---\n".join(results) if results else "No results found."


@tool
def scrape_url(url: str) -> str:
    """Fetch and extract the main text content from a URL.

    Args:
        url: The URL to scrape.

    Returns:
        Extracted text content from the page (truncated to 3000 chars).
    """
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.extract(urls=[url])

    if response.get("results"):
        content = response["results"][0].get("raw_content", "")
        return content[:3000] if content else "Could not extract content."

    return "Failed to fetch URL content."
