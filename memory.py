# memory.py

# --- Conversation Memory (Short-term) ---
# A list of every message exchanged.
# This is what the agents "see" when they generate responses.
conversation= []

# --- Knowledge Memory (Structured, Long-Term) ---
# Explicit facts extracted from the conversation.
# This is what the system "knows" at any point.
knowledge = {
    "definations":[],
    "causes":[],
    "effects":[],
    "statistics":[],
    "solutions":[]
}

def add_message(role, content):
    """Add a message to the conversation memory"""
    conversation.append({"role":role, "content":content})
    
def get_conversation_text():
    """Format converstion memory as redable text"""
    if not conversation:
        return ""
    
    
    text = ""
    
    for msg in conversation:
        label = msg["role"].upper()
        text += f"[{label}]: {msg['content']}\n\n"
    return text.strip()

def add_knowledge(category, fact):
    """Add a fact to knowledge memory. Avoid duplicates."""
    if category in knowledge and fact not in knowledge[category]:
        knowledge[category].append(fact)