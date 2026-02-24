# Building Agentic AI From Scratch

A hands-on tutorial. No frameworks. No magic. Just Python, an LLM, and a search tool.

By the end, you will understand how tools, memory, and control loops create agentic behavior.

---

## What We're Building

A CLI podcast simulator where two agents discuss fossil fuels:

- **Moderator Agent** — asks progressive questions
- **Expert Agent** — answers questions, uses web search when needed

```
┌─────────────────────────────────────────────────┐
│                 ORCHESTRATOR (main.py)           │
│                                                  │
│   ┌─────────────┐          ┌──────────────┐     │
│   │  MODERATOR   │          │   EXPERT      │     │
│   │  Agent       │─question─▶  Agent        │     │
│   │              │          │               │     │
│   │  Role prompt │          │  Role prompt  │     │
│   │  + memory    │◀─answer──│  + memory     │     │
│   └─────────────┘          │  + tools      │     │
│                             └──────┬───────┘     │
│                                    │              │
│                             ┌──────▼───────┐     │
│                             │  SEARCH TOOL  │     │
│                             │  (DuckDuckGo) │     │
│                             └──────────────┘     │
│                                                  │
│   ┌──────────────────────────────────────────┐   │
│   │              MEMORY                       │   │
│   │  conversation: [{role, content}, ...]     │   │
│   │  knowledge: {definitions, causes, ...}    │   │
│   └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

What makes this "agentic":

| Component     | What it does                              |
| ------------- | ----------------------------------------- |
| Role prompts  | Give each agent a distinct behavior       |
| Tool (search) | Let the expert fetch external information |
| Memory        | Track conversation + accumulate knowledge |
| Control loop  | Orchestrate the turn-by-turn interaction  |

That's it. That's an agent system.

---

## Prerequisites

- **Python 3.10+** installed
- **Ollama** installed and running with a model pulled

If you don't have Ollama yet:

```bash
# Install Ollama (macOS)
brew install ollama

# Start the Ollama server (runs in background)
ollama serve

# Pull a model (in a separate terminal)
ollama pull phi3:mini
```

Verify it works:

```bash
ollama run phi3:mini "Say hello"
```

You should see a response. If yes, you're ready.

> **Note:** You can use any Ollama model. `phi3:mini` is small and fast.
> If you want better quality, try `llama3.1:8b` or `mistral`.
> Just change the model name in `config.py` later.

---

## Project Structure

Here's what we'll create:

```
talking-agents/
├── venv/               # Virtual environment (generated)
├── config.py           # Settings: model, topic, turns
├── llm.py              # Single function to call the LLM
├── tools.py            # Web search tool
├── memory.py           # Conversation + knowledge storage
├── agents.py           # Moderator and Expert logic
├── main.py             # The orchestrator (control loop)
└── requirements.txt    # Dependencies
```

Each file maps to one concept. That's intentional.

---

## Step 1: Project Setup

Open your terminal in the `talking-agents` directory.

### Create Virtual Environment

```bash
python3 -m venv venv
```

This creates an isolated Python environment inside `venv/`.

### Activate It

```bash
source venv/bin/activate
```

Your terminal prompt should now show `(venv)` at the beginning.

> Every time you open a new terminal to work on this project,
> run `source venv/bin/activate` again.

### Create `requirements.txt`

```
openai
duckduckgo-search
```

Two dependencies. That's all.

- `openai` — the OpenAI Python client. We use it to talk to Ollama because Ollama exposes an OpenAI-compatible API. This means the same code works with OpenAI, Ollama, LM Studio, or any compatible server. You learn one API, it works everywhere.
- `duckduckgo-search` — free web search. No API key needed.

### Install

```bash
pip install -r requirements.txt
```

---

## Step 2: Configuration — `config.py`

This file holds all settings in one place. No magic strings scattered across files.

```python
# config.py

# --- LLM Settings ---
# Ollama exposes an OpenAI-compatible API at this URL.
# If you switch to OpenAI, change these two values.
MODEL = "phi3:mini"
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"  # Ollama ignores this, but the openai package requires it

# --- Podcast Settings ---
TOPIC = "fossil fuels and their impact on climate change"
MAX_TURNS = 5  # Number of question-answer rounds
```

**Why a config file?**

Because when you want to change the model, topic, or number of turns, you change one file. Not five.

**Why `openai` package for Ollama?**

Ollama runs a server at `localhost:11434`. It speaks the same protocol as OpenAI's API. So the `openai` Python package connects to it directly. You're learning the industry-standard API format. If you ever switch to OpenAI, Claude, or any other provider, you change two lines in this file.

---

## Step 3: The LLM Caller — `llm.py`

This is the most important file to understand. Every agent call goes through here.

```python
# llm.py

from openai import OpenAI
from config import MODEL, BASE_URL, API_KEY

# Create one client. Reuse it for every call.
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


def call_llm(system_prompt, user_message):
    """
    Call the LLM with a system prompt and a user message.
    Returns the model's response as a string.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content
```

**What's happening here:**

1. We create an OpenAI client pointed at Ollama's local server.
2. The `call_llm` function takes two strings:
   - `system_prompt` — tells the model WHO it is (its role)
   - `user_message` — tells the model WHAT to do right now
3. The model returns a completion. We extract the text.

**This is how every LLM API works.** Messages go in, text comes out.

The `messages` list uses three roles:

| Role        | Purpose                                        |
| ----------- | ---------------------------------------------- |
| `system`    | Sets the model's behavior and identity         |
| `user`      | The input/question for this turn               |
| `assistant` | The model's previous responses (not used here) |

We keep it simple: one system message, one user message. The conversation history gets embedded inside the user message as formatted text. This avoids complicated role-mapping logic.

**`temperature=0.7`** — controls randomness. Lower = more deterministic. Higher = more creative. 0.7 is a good default for conversational output.

---

## Step 4: The Search Tool — `tools.py`

This is where "agentic" begins. A tool is just a function that the system can call.

```python
# tools.py

from duckduckgo_search import DDGS


def search_web(query, max_results=3):
    """
    Search the web using DuckDuckGo.
    Returns a formatted string of results.
    """
    results = DDGS().text(query, max_results=max_results)

    if not results:
        return "No results found."

    output = ""
    for r in results:
        output += f"- {r['title']}: {r['body']}\n"

    return output
```

**What's happening here:**

1. We call DuckDuckGo's search API with a query string.
2. We get back a list of results, each with `title`, `href`, and `body`.
3. We format them into a simple text block.
4. That text block gets injected into the expert's prompt.

**Why is this a "tool"?**

Because the agent system decides when to call it. The function itself is dumb — it just searches. The intelligence is in the decision to search.

That's the key insight: **tools are simple functions. The agent logic decides when and how to use them.**

**Test it** (optional, in Python REPL):

```bash
python3 -c "from tools import search_web; print(search_web('fossil fuels climate impact'))"
```

You should see a few search results printed.

---

## Step 5: Memory — `memory.py`

Memory is what separates a chatbot from an agent. Without memory, every turn is independent. With memory, the system builds context over time.

We use two types:

```python
# memory.py

# --- Conversation Memory (Short-Term) ---
# A list of every message exchanged.
# This is what the agents "see" when they generate responses.
conversation = []


# --- Knowledge Memory (Structured, Long-Term) ---
# Explicit facts extracted from the conversation.
# This is what the system "knows" at any point.
knowledge = {
    "definitions": [],
    "causes": [],
    "effects": [],
    "statistics": [],
    "solutions": [],
}


def add_message(role, content):
    """Add a message to conversation memory."""
    conversation.append({"role": role, "content": content})


def get_conversation_text():
    """Format conversation memory as readable text."""
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
```

**Two types of memory. Here's why:**

### Conversation Memory

```python
[
    {"role": "moderator", "content": "What are fossil fuels?"},
    {"role": "expert", "content": "Fossil fuels are..."},
]
```

This is just a running transcript. Every turn, the agents read it to understand what was already discussed. This prevents repetition and enables progressive questions.

### Knowledge Memory

```python
{
    "definitions": ["Fossil fuels are carbon-rich deposits..."],
    "causes": ["Burning fossil fuels releases CO2..."],
    "statistics": ["Coal accounts for 27% of global energy..."],
}
```

This is structured information extracted from each answer. The conversation memory is "what was said." The knowledge memory is "what was learned."

**Why not just use conversation memory?**

Because re-reading raw conversation text is not knowledge. It's just text replay. Structured extraction forces the system to identify and categorize facts. That's the difference between a chat log and a knowledge base.

> This is also what real agent systems do. They don't just keep a chat history.
> They extract, categorize, and store structured information.

---

## Step 6: The Agents — `agents.py`

Now we define the two agents. Each agent is:

- A **system prompt** (its role/personality)
- **Functions** that call the LLM with that prompt

```python
# agents.py

from llm import call_llm
from config import TOPIC


# ============================================================
#  SYSTEM PROMPTS
# ============================================================

MODERATOR_PROMPT = f"""You are a podcast moderator interviewing an expert.

Topic: {TOPIC}

Rules:
- Ask clear, focused questions about the topic
- Progress from basic concepts to advanced issues
- Never repeat a question already asked in the conversation
- Ask ONE question at a time
- Keep questions short (1-2 sentences)
- If the conversation has covered basics, move to causes, effects, solutions

Respond with ONLY the question. No preamble. No "Great answer!" Just the question."""


EXPERT_PROMPT = f"""You are an expert being interviewed on a podcast.

Topic: {TOPIC}

Rules:
- Answer the question clearly and concisely
- Use facts and data when available
- If search results are provided, incorporate them into your answer
- Keep answers to 3-5 sentences
- Be informative but conversational

Respond with ONLY your answer. No "That's a great question!" Just answer."""


# ============================================================
#  TOOL DECISION
# ============================================================

# These keywords trigger a web search.
# This is the simplest possible "tool use" logic.
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
    question_lower = question.lower()
    return any(kw in question_lower for kw in SEARCH_KEYWORDS)


def generate_search_query(question):
    """Use the LLM to turn the moderator's question into a search query."""
    prompt = (
        "Convert this interview question into a concise web search query "
        "(3-6 words). Respond with ONLY the search query, nothing else.\n\n"
        f"Question: {question}"
    )
    return call_llm("You generate web search queries.", prompt).strip().strip('"')


# ============================================================
#  AGENT FUNCTIONS
# ============================================================

def get_moderator_question(conversation_text):
    """The Moderator generates the next question."""
    if not conversation_text:
        user_msg = "This is the start of the interview. Ask your first question."
    else:
        user_msg = (
            f"Here is the conversation so far:\n\n"
            f"{conversation_text}\n\n"
            f"Ask your next question. Do not repeat any previous question."
        )
    return call_llm(MODERATOR_PROMPT, user_msg).strip()


def get_expert_answer(question, conversation_text, search_context=""):
    """The Expert generates an answer, optionally using search results."""
    user_msg = ""

    if conversation_text:
        user_msg += f"Conversation so far:\n\n{conversation_text}\n\n"

    user_msg += f"Question: {question}"

    if search_context:
        user_msg += (
            f"\n\nWeb search results (use these facts in your answer):\n"
            f"{search_context}"
        )

    return call_llm(EXPERT_PROMPT, user_msg).strip()
```

**Let's break down what's happening:**

### System Prompts

Each agent has a system prompt that defines:

- WHO it is (moderator vs expert)
- WHAT it should do
- HOW it should respond
- Constraints (length, format, behavior)

The prompts end with **"Respond with ONLY..."** — this is critical. Without this constraint, models add filler like "That's a great question!" or "Let me think about this..." You want clean output.

### Tool Decision: `needs_search()`

This is the simplest possible tool-use logic:

```
Does the question contain "statistics"? → Search.
Does it contain "latest"? → Search.
Otherwise → No search.
```

It's a keyword check. That's it.

In more advanced systems, the LLM itself decides whether to use tools (function calling, structured output). But for learning, keyword matching teaches the concept without the complexity.

**The concept is:** the system inspects the input and decides whether to invoke a tool. That's tool use.

### `generate_search_query()`

The moderator might ask: _"What are the latest statistics on global CO2 emissions from fossil fuels?"_

That's too long for a search query. So we use the LLM to compress it:
→ `"global CO2 emissions fossil fuels 2025"`

This is a tiny but real example of using an LLM as a utility function — not as an agent, but as a text transformer.

### Agent Functions

Each function:

1. Formats the conversation history into a string
2. Adds the current task (question to ask / question to answer)
3. Calls the LLM with the appropriate system prompt
4. Returns the response

Notice: **both agents use the same `call_llm()` function.** The only difference is the system prompt. That's what makes them "different agents." Same engine, different instructions.

---

## Step 7: The Orchestrator — `main.py`

This is the control loop. The brain of the system.

```python
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
    add_knowledge,
)
from agents import (
    get_moderator_question,
    get_expert_answer,
    needs_search,
    generate_search_query,
)


# ============================================================
#  KNOWLEDGE EXTRACTION
# ============================================================

EXTRACT_PROMPT = """Extract key facts from the following text.
Categorize each fact as exactly one of: definition, cause, effect, statistic, solution.

Respond in this exact format (one fact per line):
category: fact

Example:
definition: Fossil fuels are carbon-rich deposits formed from ancient organisms
statistic: Coal accounts for 27% of global energy production
cause: Burning fossil fuels releases carbon dioxide into the atmosphere

Rules:
- One fact per line
- Use ONLY these categories: definition, cause, effect, statistic, solution
- Keep each fact to one sentence
- If there are no clear facts, respond with: none

Text to extract from:
"""


def extract_knowledge(answer):
    """
    Use the LLM to extract structured facts from the expert's answer.
    Store them in knowledge memory.
    """
    response = call_llm(
        "You extract and categorize facts from text. Follow the format exactly.",
        EXTRACT_PROMPT + answer,
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
                add_knowledge(category_map[category], fact)


# ============================================================
#  MAIN LOOP
# ============================================================

def run():
    print("=" * 60)
    print(f"  PODCAST: {TOPIC.upper()}")
    print("=" * 60)
    print()

    for turn in range(1, MAX_TURNS + 1):
        print(f"--- Turn {turn}/{MAX_TURNS} ", "-" * 40)
        print()

        # ----- STEP 1: Moderator asks a question -----
        conv_text = get_conversation_text()
        question = get_moderator_question(conv_text)
        add_message("moderator", question)
        print(f"  [MODERATOR]: {question}")
        print()

        # ----- STEP 2: Decide if search is needed -----
        search_context = ""
        if needs_search(question):
            query = generate_search_query(question)
            print(f"  [SEARCH]: looking up '{query}'...")
            search_context = search_web(query)
            print(f"  [SEARCH]: found {search_context.count(chr(10))} results")
            print()

        # ----- STEP 3: Expert answers -----
        conv_text = get_conversation_text()
        answer = get_expert_answer(question, conv_text, search_context)
        add_message("expert", answer)
        print(f"  [EXPERT]: {answer}")
        print()

        # ----- STEP 4: Extract and store knowledge -----
        extract_knowledge(answer)

        print()

    # ----- FINAL OUTPUT -----
    print("=" * 60)
    print("  ACCUMULATED KNOWLEDGE")
    print("=" * 60)
    print()
    print(json.dumps(knowledge, indent=2))
    print()

    print("=" * 60)
    print("  FULL CONVERSATION")
    print("=" * 60)
    print()
    print(get_conversation_text())


if __name__ == "__main__":
    run()
```

**This is the entire agentic system. Let's trace through one turn:**

```
Turn 1:
  1. Get conversation so far           → "" (empty, first turn)
  2. Moderator generates question      → "What are fossil fuels?"
  3. Save to conversation memory       → [{"role": "moderator", ...}]
  4. Check if search needed            → "fossil fuels" has no search keywords → NO
  5. Expert answers                    → "Fossil fuels are..."
  6. Save to conversation memory       → [moderator msg, expert msg]
  7. Extract knowledge from answer     → definitions: ["Fossil fuels are..."]

Turn 2:
  1. Get conversation so far           → "[MODERATOR]: What are...\n[EXPERT]: Fossil fuels are..."
  2. Moderator reads history           → Sees basics covered
  3. Moderator generates question      → "What are the latest statistics on fossil fuel emissions?"
  4. Check if search needed            → "latest" + "statistics" → YES
  5. Generate search query             → "fossil fuel emissions statistics 2025"
  6. Call search tool                  → Returns web results
  7. Expert answers using results     → "According to recent data..."
  8. Save + extract knowledge          → statistics: ["CO2 emissions reached..."]
```

**That's the loop.** Each turn builds on the previous. Memory grows. Knowledge accumulates. Tools get invoked when needed.

### The Three LLM Calls Per Turn

Each turn makes up to 3 LLM calls:

1. **Moderator** — generate the question
2. **Search query** — (only when search is needed) compress question into query
3. **Expert** — generate the answer

Plus 1 more for knowledge extraction:

4. **Extractor** — pull structured facts from the answer

So each turn is 3-4 LLM calls. With 5 turns, that's 15-20 total calls. On a local model, this takes 1-3 minutes depending on your hardware.

---

## Step 8: Run It

Make sure Ollama is running:

```bash
# In a separate terminal, if not already running
ollama serve
```

Then run the podcast:

```bash
# Make sure you're in the talking-agents directory
# Make sure venv is activated
python3 main.py
```

### Expected Output

```
============================================================
  PODCAST: FOSSIL FUELS AND THEIR IMPACT ON CLIMATE CHANGE
============================================================

--- Turn 1/5 ----------------------------------------

  [MODERATOR]: What exactly are fossil fuels and how are they formed?

  [EXPERT]: Fossil fuels are natural energy sources formed from the
  remains of ancient plants and organisms buried deep underground
  millions of years ago. The three main types are coal, oil, and
  natural gas. Over millions of years, heat and pressure transformed
  these organic materials into carbon-rich deposits that we extract
  and burn for energy today.

--- Turn 2/5 ----------------------------------------

  [MODERATOR]: What are the latest statistics on global fossil fuel
  consumption and CO2 emissions?

  [SEARCH]: looking up 'global fossil fuel CO2 emissions 2025'...
  [SEARCH]: found 3 results

  [EXPERT]: According to recent data, global CO2 emissions from fossil
  fuels reached approximately 37.4 billion tonnes in 2024. Coal remains
  the largest single source, followed by oil and natural gas. Despite
  growth in renewables, fossil fuels still account for roughly 80% of
  global primary energy consumption.

...

============================================================
  ACCUMULATED KNOWLEDGE
============================================================

{
  "definitions": [
    "Fossil fuels are natural energy sources formed from ancient organisms"
  ],
  "causes": [
    "Burning fossil fuels releases CO2 into the atmosphere"
  ],
  "effects": [
    "Rising global temperatures due to greenhouse gas accumulation"
  ],
  "statistics": [
    "Global CO2 emissions reached 37.4 billion tonnes in 2024",
    "Fossil fuels account for roughly 80% of global primary energy"
  ],
  "solutions": [
    "Transitioning to renewable energy sources like solar and wind"
  ]
}
```

---

## What You Just Built

Let's map what you implemented to agent concepts:

| Concept               | Your Implementation                                              |
| --------------------- | ---------------------------------------------------------------- |
| **Agent**             | A system prompt + functions that call the LLM                    |
| **Role**              | Moderator prompt vs Expert prompt (same LLM, different behavior) |
| **Tool**              | `search_web()` — a function the system calls conditionally       |
| **Tool Use Logic**    | `needs_search()` — keyword check that triggers the tool          |
| **Short-Term Memory** | `conversation` list — raw message history                        |
| **Long-Term Memory**  | `knowledge` dict — structured facts extracted from answers       |
| **Control Loop**      | The `for` loop in `main.py` — orchestrates everything            |
| **Orchestrator**      | `main.py` itself — decides who speaks, when to search            |

### Key Insight

The "agents" are not separate programs. They are **different prompts fed to the same model**. The orchestrator switches between them. That's multi-agent systems at their core.

```
call_llm(MODERATOR_PROMPT, ...)  →  Agent A speaks
call_llm(EXPERT_PROMPT, ...)     →  Agent B speaks
```

Same function. Different system prompt. Different behavior.

---

## What to Experiment With

Now that it works, try these modifications to deepen your understanding:

### 1. Change the Topic

In `config.py`, change `TOPIC` to anything:

```python
TOPIC = "the history of artificial intelligence"
```

Run again. Same system, completely different conversation.

### 2. Add More Turns

```python
MAX_TURNS = 10
```

Watch how the moderator progresses from basics to advanced topics.

### 3. Add a Third Agent

Create a `SKEPTIC_PROMPT` in `agents.py`:

```python
SKEPTIC_PROMPT = f"""You are a skeptic on a podcast panel.
Topic: {TOPIC}
After the expert answers, raise one counterpoint or challenge.
Keep it to 1-2 sentences. Be respectful but critical."""
```

Add a skeptic turn in the main loop after the expert answers.

### 4. Improve Tool Decision

Replace keyword matching with an LLM-based decision:

```python
def needs_search(question):
    response = call_llm(
        "Answer YES or NO only. Does this question require recent data or statistics?",
        question
    )
    return "YES" in response.upper()
```

Now the model itself decides. That's closer to real tool calling.

### 5. Save Knowledge to File

At the end of `main.py`, add:

```python
with open("knowledge.json", "w") as f:
    json.dump(knowledge, f, indent=2)
```

Now your knowledge persists across runs.

### 6. Load Previous Knowledge

At the start, load existing knowledge:

```python
import os
if os.path.exists("knowledge.json"):
    with open("knowledge.json") as f:
        loaded = json.load(f)
        for key in knowledge:
            knowledge[key] = loaded.get(key, [])
```

Now the system "remembers" across sessions. That's persistent memory.

---

## Common Errors and Fixes

### `Connection refused` on LLM call

Ollama is not running. Start it:

```bash
ollama serve
```

### `Model not found`

Pull the model first:

```bash
ollama pull phi3:mini
```

### Search returns empty results

DuckDuckGo may rate-limit. Wait a few seconds and retry. Or reduce `max_results` to 2.

### Model outputs garbage / doesn't follow format

Small models sometimes ignore instructions. Options:

- Use a larger model (`llama3.1:8b`)
- Add "Respond with ONLY..." to your prompts (already done)
- Lower `temperature` to 0.3 for more deterministic output

### Very slow responses

Local models depend on your hardware. Options:

- Use a smaller model (`phi3:mini` instead of larger ones)
- Reduce `MAX_TURNS`
- Be patient — this is local inference without a GPU cluster

---

## Recap: The Mental Model

```
Agent = Role Prompt + Tools + Memory + Loop
```

You now have all four.

The entire system is ~150 lines of Python across 6 files. No frameworks. No abstractions you don't understand. Every line does something you can trace.

That's the foundation. Every agent framework (LangChain, CrewAI, AutoGen) is just layers on top of exactly this pattern.
