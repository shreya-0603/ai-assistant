from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import json
import datetime
import urllib.request
import urllib.parse

GROQ_API_KEY =  st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="qwen/qwen3.6-27b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant with memory."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

parser = StrOutputParser()
chain = prompt | llm | parser

def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_date() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %y")

def search_web(query: str) -> str:
    """Search using DuckDuckGo Lite"""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
        
        results = []
        
        # Try abstract first (Wikipedia summary)
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
        
        # Try answer (quick facts)
        if data.get("Answer"):
            results.append(f"Answer: {data['Answer']}")
            
        # Try related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(topic["Text"])
        
        if results:
            return "\n".join(results[:3])
        else:
            return f"No direct results found for '{query}'. The agent will answer from its training knowledge."
            
    except Exception as e:
        return f"Search unavailable: {str(e)}"
def execute_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "calculate":
        return calculate(tool_args["expression"])
    elif tool_name == "get_current_date":
        return get_current_date()
    elif tool_name == "search_web":
        return search_web(tool_args["query"])
    else:
        return f"Unknown tool: {tool_name}"

def chat(question: str, chat_history: list) -> str:
    print(f"DEBUG: chat() called with: {question}")
    print(f"DEBUG: history length: {len(chat_history)}")
    system_prompt = """You are a helpful AI assistant with memory and tools.

IMPORTANT: You MUST use tools in these situations:
- ANY math calculation → use calculate tool
- Current date/time → use get_current_date tool  
- Recent news, current events → use search_web tool

DO NOT answer math or date questions from memory.
ALWAYS use the tool first.

Available tools:
1. calculate(expression) - for ANY math
   Format: TOOL: calculate | ARGS: {"expression": "2+2"}

2. get_current_date() - for current date/time
   Format: TOOL: get_current_date | ARGS: {}

3. search_web(query) - for recent news/events
   Format: TOOL: search_web | ARGS: {"query": "your search"}

RULES:
- If you need a tool, respond with ONLY the TOOL: line, nothing else
- Multiple tools = one per line
- Only answer directly if NO tool is needed"""
    # Build messages manually for tool loop
    messages_for_llm = [{"role": "system", "content": system_prompt}]
    
    # Add chat history
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages_for_llm.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages_for_llm.append({"role": "assistant", "content": msg.content})
    
    # Add current question
    messages_for_llm.append({"role": "user", "content": question})
    
    # Agent loop
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    for step in range(5):
        response = groq_client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=messages_for_llm,
            temperature=0
        )
        
        reply = response.choices[0].message.content
        
        if "TOOL:" in reply:
            print(f"🔧 TOOL DETECTED: {reply}")
            lines = [l.strip() for l in reply.strip().split("\n")
                     if l.strip().startswith("TOOL:")]
            
            all_results = []
            for line in lines:
                try:
                    parts = line.split(" | ARGS: ")
                    tool_name = parts[0].replace("TOOL:", "").strip()
                    tool_args = json.loads(parts[1].strip())
                    tool_result = execute_tool(tool_name, tool_args)
                    all_results.append(f"{tool_name}: {tool_result}")
                except Exception as e:
                    all_results.append(f"Tool error: {str(e)}")
            
            if all_results:
                messages_for_llm.append({"role": "assistant", "content": reply})
                messages_for_llm.append({
                    "role": "user",
                    "content": "Tool results:\n" + "\n".join(all_results) +
                               "\n\nNow answer the original question."
                })
        else:
            # Save to LangChain memory
            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=reply))
            return reply
    
    return "I could not complete that task."


# ─── STREAMLIT UI ─────────────────────────────────────────

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    .stChatInputContainer {
        border-radius: 20px;
    }
    .tool-badge {
        background-color: #1f2937;
        border-left: 3px solid #6366f1;
        padding: 8px 12px;
        border-radius: 5px;
        font-size: 0.85em;
        color: #a5b4fc;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🤖")
with col2:
    st.title("AI Assistant")
    st.caption("Powered by LLaMA 3.3 • Memory • Tools • Web Search")

st.divider()

# Sidebar with info
with st.sidebar:
    st.markdown("## 🛠️ Available Tools")
    st.markdown("""
    - 🧮 **Calculator** — math expressions
    - 📅 **Date** — current date and time  
    - 🔍 **Web Search** — recent information
    """)
    
    st.divider()
    
    st.markdown("## 💡 Try asking:")
    st.markdown("""
    - *What is 1847362 × 9284756?*
    - *What is today's date?*
    - *Search for latest AI news*
    - *My name is [name], remember it*
    """)
    
    st.divider()
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Built with LangChain + Groq + Streamlit")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []

# Welcome message
if not st.session_state.display_history:
    st.markdown("""
    <div style='text-align: center; padding: 40px; color: #6b7280;'>
        <h3>👋 Hello! I'm your AI Assistant</h3>
        <p>I can remember our conversation, do math, check the date, and search the web.</p>
        <p>Try asking me something!</p>
    </div>
    """, unsafe_allow_html=True)

# Display conversation
for message in st.session_state.display_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "tool" in message:
            st.markdown(f"""
            <div class='tool-badge'>
                🔧 Used tool: {message["tool"]}
            </div>
            """, unsafe_allow_html=True)

# Handle input
query = st.chat_input("Ask me anything...")

if query:
    with st.spinner("🤔 Thinking..."):
        answer = chat(query, st.session_state.chat_history)
        st.session_state.display_history.append({
            "role": "user",
            "content": query
        })
        st.session_state.display_history.append({
            "role": "assistant",
            "content": answer
        })
    st.rerun()
