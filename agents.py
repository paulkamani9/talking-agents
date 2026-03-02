# agents.py

from llm import call_llm
from prompts import MODERATOR_PROMPT, EXPERT_PROMPT
from typing import Optional
import re
from dataclasses import dataclass
import json


@dataclass
class SearchDecision:
    should_search: bool
    reason: str
    confidence: float
    source: str # "hueristic" | "llm" | "fallback"


# ==== Tool decision ===
# this keywords trigger a web search

SEARCH_KEYWORDS = [
    "latest", "statistics", "data", "numbers", "recent",
    "current", "how much", "how many", "percentage",
    "trend", "year", "2024", "2025", "2026", "factors","2027", "2028","2029","2030" "report",
                   ]

def heuristic_needs_search(question:  str) -> bool:
    """
    Decide if the expert needs to search the web. 
    Simple keyword matching. No LLM call needed.
    """
    question_lower= question.lower()
    
    has_keyword = any(kw in question_lower for kw in SEARCH_KEYWORDS)
    has_year = bool(re.search(r"\b20\d{2}\b", question_lower))
    
    return has_keyword or has_year


def parse_decision_json(raw:str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        return None


def needs_search_withLLm(question, call_llm)-> SearchDecision:
    """
    Decide if the expert needs to search the web, with llm intuition
    """
    
    if heuristic_needs_search(question):
        return SearchDecision(
            True, "Hueristic keyword match", 0.9, "hueristic"
        )
        
    prompt = (
        "Return Only valid JSON with keys: should_search (boolean), confidence (0-1), reason (string). \n\n"
        "Rule: should_search=true when correctness depends on recent facts, statistics, dates,"
        " rankings, prices, laws, elections, releases, or other time sensitive information. \n\n"
        f"Question: {question}"
    )
    
    raw = call_llm("You are a strict classifier for web search neccesity.", prompt, 0.2) or "" 
    
    parsed = parse_decision_json(raw.strip())
    
    if parsed and isinstance(parsed.get("should_search"), bool):
        return SearchDecision(
            confidence=parsed.get("confidence", 0.5),
            reason=parsed.get("reason", "No reason provided"),
            source="llm",
            should_search=parsed["should_search"]
        )
    
    
    return SearchDecision(
        should_search=True,
        reason="Fallback decision: default to searching",
        confidence=0.3,
        source="fallback"
    )



def generate_search_query(question: str, call_llm) -> str:
    prompt = (
        "Return ONLY JSON: {\"query\": \"...\"}.\n"
        "Query should be 4-10 words, specific, factual, and searchable.\n"
        f"Question: {question}"
    )
    raw = call_llm("You generate web search queries.", prompt, 0.2) or ""

    try:
        data = json.loads(raw)
        query = str(data.get("query", "")).strip()
        if query:
            return query
    except Exception:
        pass

    # Fallback
    return question.strip()[:120]
     

# === Agent functions ===
def get_moderator_question(conversation_text):
    """The moderator generates the next question"""
    if not conversation_text:
        user_msg = "This is the start of the interview ask your first question."
    else: 
        user_msg = (
            f"Here is the conversation so far:\n\n"
            f"{conversation_text}\n\n"
            f"Ask your next question. Do not repeat any previous questions."
        )
    return call_llm(MODERATOR_PROMPT, user_msg, 0.7).strip()
    
    
def get_expert_answer(question, converstion_text, search_context=""):
     """The Expert generates an answer, optionally using search results."""
     user_msg= ""
     
     if converstion_text:
         user_msg += f"Conversations so far: \n\n{converstion_text}\n\n"
         
     user_msg += f"Question {question}"
     
     if search_context:
         user_msg += (
             f"\n\n Web search results (use these facts in your answer):\n"
             f"{search_context}"
         )
         
    
     return call_llm(EXPERT_PROMPT, user_msg, 0.7).strip()
        