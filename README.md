# Talking Agents

A small Python experiment where two agents run a podcast-style interview:

- a **moderator** asks questions
- an **expert** answers

The script can optionally do a quick web search for recent facts, then saves extracted knowledge to `knowledge.json`.

## What it does

- Runs a turn-based conversation (default: 5 turns)
- Uses an OpenAI-compatible chat API (works with Ollama out of the box)
- Triggers web search for "latest/data/statistics" type questions
- Extracts facts from answers into categories
- Prints the full conversation at the end

## Project files

- `main.py` – main loop
- `agents.py` – moderator/expert prompts + search decision
- `llm.py` – shared LLM client
- `tools.py` – DuckDuckGo search helper
- `memory.py` – conversation + structured knowledge store
- `config.py` – model and topic settings

## Quick start

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure your model endpoint is running.
   - Default config expects Ollama at `http://localhost:11434/v1`

4. Run:

```bash
python main.py
```

## Config

Edit `config.py` to change:

- `MODEL`
- `BASE_URL`
- `API_KEY`
- `TOPIC`
- `MAX_TURNS`

## Notes

- Web search uses DuckDuckGo (`ddgs`).
- Output knowledge is written to `knowledge.json`.
- If search package import fails, install/update with:

```bash
pip install ddgs
```
