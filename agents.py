# agents.py

from llm import call_llm
from config import TOPIC

# ==== System Prompts ===
MODERATOR_PROMPT = f"""You are a podcast moderator interviewing an expert.
Topic: {TOPIC}
Rules:
- Ask clear, focused questions about the topic
- Progress from basic issues to advanced issues
- Never repeat a question already asked in the conversation
- Ask one question at a time
- Keep questions short (1-2 sentences)
- Let your questions be guided by the expert's answers, suct that there is a flow to the conversation
- Keep your questions about the as latest as possible
- If the convesrtion as covered the basics, move to causes, effects and solutions
- Be friendly

Respond with ONLY the question. No preamble. No "Great answer!" Just the question. 
"""


EXPERT_PROMPT = f"""You are an expert being interviewed on a podcast.

Topic: {TOPIC}

Rules:
- Answer the question clearly and concisely
- Use facts and data when available
- If search results are provided, incorporate them into your answer
- Keep answers to 3-5 sentences
- Be informative but conversational
- Feel free to use illustrations and anecdotes

Respond with ONLY your answer. No "That's a great question!" Just answer."""

# ==== Tool decision ===
# this keywords trigger a web search

SEARCH_KEYWORDS = [
    "latest", "statistics", "data", "numbers", "recent",
    "current", "how much", "how many", "percentage",
    "trend", "year", "2024", "2025", "2026", "report",
                   ]

def needs_search(question):
    """
    Decide if the expert needs to search the web. 
    Simple keyword matching. No LLM call needed.
    """
    question_lower= question.lower()
    
    return any(kw in question_lower for kw in SEARCH_KEYWORDS)


def generate_search_query(question):
     """Use the LLM to turn the moderator's question into a search query."""
     prompt = (
         "Convert this interview question to a concise web search query"
         "(3-6 words). Respond with ONLY the search query, nothing else. \n\n"
         f"Question:{question}"
     )
     search_query = call_llm("You generate web search queries.", prompt)
     
     if not search_query:
         return "Search query not generated"
     
     return search_query.strip().strip("'")
     

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
    return call_llm(MODERATOR_PROMPT, user_msg).strip()
    
    
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
         
    
     return call_llm(EXPERT_PROMPT, user_msg).strip()
        