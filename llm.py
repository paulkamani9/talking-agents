# llm.py

from openai import OpenAI
from config import MODEL, BASE_URL, API_KEY

# Create one client. Reuse it for every call
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)


def call_llm(system_prompt, user_message, temperature:float) -> str:
    """
    Call the LLM with a system prompt and a user message.
    Returns the model's response as a string
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role":"system", "content": system_prompt
                },
                {
                    "role":"user", "content": user_message
                }
            ], 
            temperature=temperature if temperature is not None else 0.2, # Controls randomness. Higher values = more random, lower values = more deterministic
        )
        
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""
    except Exception as e:
        print(f"LLM call failed: {e}")
        return ""
    
    