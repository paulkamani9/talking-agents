# main.py

import json
from config import MAX_TURNS, TOPIC
from llm import call_llm
from tools import search_web
from memory import (
    conversation,
    knowledge,
    add_message,
    get_conversation_text,
    add_knowledge
)
from agents import (
    get_moderator_question,
    get_expert_answer,
    needs_search,
    generate_search_query
)


# === Knowledge extraction ===
EXTRACT_PROMPT= """
Extract key facts from the following text.
Categorise each fact as excatly on of: defination, cause, effect, statistic, solution.

Respond in this exact format (one fact per line):
category: fact

Example:
defination: Fossil fuels are carbon-rich deposits formed from ancient organisms
statistic: Coal amounts for 27 percent of global energy production
cause: Burning fossil fuels release carbon dioxide into the atmosphere


Rules: 
- One fact per line
- Use ONLY these categories: defination, cause, effect, statistic and solution
- Keep each fact to one sentence
- If there are no clear facts, respond with: none: 

Text to extract from
"""

def extract_knowledge(answer):
      """
    Use the LLM to extract structured facts from the expert's answer.
    Store them in knowledge memory.
    """
      response = call_llm(
        "You extrat and categorise facts from text. Follow the format exactly.",
        EXTRACT_PROMPT + answer
    )
      
      category_map = {
        "definition": "definitions",
        "cause": "causes",
        "effect": "effects",
        "statistic": "statistics",
        "solution": "solutions",

      }
      
      for line in response.strip().split("\n"):
          if ":" in line:
              parts = line.split(":", 1)
              category = parts[0].strip().lower()
              fact = parts[1].strip()
              if category in category_map and fact:
                  add_knowledge(category_map[category],fact)
                  
                  
# === Main Loop ===
def run():
    print("=" * 60)
    print(f" Podcast: {TOPIC}")
    print("=" * 60)
    print()
    
    for turn in range(1, MAX_TURNS + 1):
        print(f"--- Turn {turn}/{MAX_TURNS} ", "-" * 40)
        print()
        
        # Step 1: Moderator ask question
        conv_text = get_conversation_text()
        question = get_moderator_question(conv_text)
        add_message("moderator", question)
        print(f" [Moderator]: {question}")
        print()
        
        # Step 2: Decide if search is needed
        search_context = ""
        if needs_search(question):
            query = generate_search_query(question)
            print(f"[Search] looking up '{query}'...")
            search_context = search_web(query)
            print(f"[Search]: found {search_context.count(chr(10))} results")
            print()
            
        
        # Step 3: Expert answers
        conv_text = get_conversation_text
        answer = get_expert_answer(question, conv_text, search_context)
        add_message("expert", answer)
        print(f"[Expert]: {answer}")
        print()
        
        # Step 4: Extract and store knowledge
        extract_knowledge(answer)
        
        print()
        
        
    # Final output
    print("=" * 60)
    print("  ACCUMULATED KNOWLEDGE")
    print("=" * 60)
    print()
    with open("knowledge.json", "w") as f:
        json.dump(knowledge, f, indent=2)
    print()
    
    print("=" * 60)
    print("  FULL CONVERSATION")
    print("=" * 60)
    print()
    print(get_conversation_text())
    
    

if __name__ == "__main__":
    run()