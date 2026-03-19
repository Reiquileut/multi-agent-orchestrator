"""Text processing tools for the Analyst and Writer agents."""

from langchain_core.tools import tool


@tool
def summarize_text(text: str, max_sentences: int = 5) -> str:
    """Extract the most important sentences from a text block.

    This is a fast, heuristic-based summarizer (no LLM call).
    It scores sentences by word frequency and returns the top ones.

    Args:
        text: The text to summarize.
        max_sentences: Maximum sentences in the summary (default: 5).

    Returns:
        A condensed summary of the input text.
    """
    import re
    from collections import Counter

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= max_sentences:
        return text

    # Score by word frequency (simple extractive summarization)
    words = re.findall(r"\w+", text.lower())
    freq = Counter(words)

    scored = []
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r"\w+", sent.lower())
        score = sum(freq.get(w, 0) for w in sent_words) / max(len(sent_words), 1)
        scored.append((score, i, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = sorted(scored[:max_sentences], key=lambda x: x[1])

    return " ".join(s[2] for s in top)


@tool
def extract_key_points(text: str) -> str:
    """Extract structured key points from unstructured text.

    Identifies bullet-point-worthy facts and claims.

    Args:
        text: The text to extract key points from.

    Returns:
        A bullet-point list of key findings.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Heuristic: sentences with numbers, comparisons, or strong verbs
    # are more likely to be key points
    indicators = [
        r"\d+%",
        r"\$[\d,.]+",
        r"\d{4}",
        r"significant|important|key|critical|major|notably",
        r"increase|decrease|grow|decline|rise|fall",
        r"first|largest|smallest|best|worst|most|least",
    ]

    key_points = []
    for sent in sentences:
        score = sum(1 for pattern in indicators if re.search(pattern, sent, re.I))
        if score >= 1 and len(sent.split()) >= 5:
            key_points.append(f"• {sent.strip()}")

    if not key_points:
        # Fallback: return first few sentences
        key_points = [f"• {s.strip()}" for s in sentences[:5]]

    return "\n".join(key_points[:10])
