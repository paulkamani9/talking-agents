# cofiguration file 

# --- LLM Settings ---
# Ollama exposes an OpenAI-compatible API at this URL.
# If you switch to OpenAI, change these two values.
MODEL = "gemma3:4b"
BASE_URL = "http://localhost:11434/v1"  # Ollama API URL
API_KEY = "ollama" # Ollama doesnt need this, but OpenAI does. You can set it to your OpenAI API key if you switch.


# Podcast settings
TOPIC = "The state of Artificial Intelligence in 2026. US China Competition, AI Comapnies and LLM models"
MAX_TURNS = 5
