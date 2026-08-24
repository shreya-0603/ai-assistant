# 🤖 AI Assistant — Powered by qwen/qwen3.6-27b

A conversational AI assistant with memory, tools, and web search. 
Built with LangChain, Groq, and Streamlit.

🔗 **Live Demo:** [https://ai-assistant-v11.streamlit.app/]

---

## What it does

- 💬 Remembers your entire conversation
- 🧮 Calculates math expressions accurately
- 📅 Tells you the current date and time
- 🔍 Searches the web for recent information
- 🤖 Answers general knowledge questions

---

## How it works

The assistant uses a **ReAct agent loop** — on each message it decides whether to:
1. Answer directly from its own knowledge
2. Use a tool to get accurate information

This means it never guesses on math or dates — it always uses the right tool.

---

## Tech Stack

- **LLM:** qwen/qwen3.6-27b via Groq API
- **Framework:** LangChain
- **Memory:** Conversation history with HumanMessage/AIMessage
- **Tools:** Calculator, Date, Web Search (DuckDuckGo)
- **UI:** Streamlit

---

## Features

- ✅ Persistent conversation memory across turns
- ✅ ReAct agent loop for tool decision making
- ✅ Calculator tool for accurate math
- ✅ Real-time date tool
- ✅ Web search for current information
- ✅ Clean sidebar with tool list and examples
- ✅ Clear conversation button

---

## Run locally

```bash
git clone https://github.com/shreya-0603/ai-assistant
cd ai-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run project2.py
```

Add your Groq API key in `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-key-here"
```

---

## What I learned

- How to build a ReAct agent loop from scratch
- Tool calling via prompt engineering
- LangChain chains and memory with MessagesPlaceholder
- Difference between answering from memory vs using tools
- Building and deploying conversational AI apps
