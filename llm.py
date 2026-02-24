# llm.py

from openai import OpenAI
from config import MODEL, BASE_URL, API_KEY

# Create one client. Reuse it for every call
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)


def call_llm(system_prompt, user_message):
    """
    Call the LLM with a system prompt and a user message.
    Returns the model's response as a string
    """
    
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
        temperature=0.7, # Controls randomness. Higher values = more random, lower values = more deterministic
    )
    
    return response.choices[0].message.content