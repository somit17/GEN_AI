from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

load_dotenv()
groq_llm_model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    streaming=True
    # other params...
)

search = GoogleSerperAPIWrapper()
tools = [search.run]
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []
print(st.session_state.memory)

agent = create_agent(
    model=groq_llm_model,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are amazing AI agent and can search on google as well !"
    )
thread_config = {"configurable": {"thread_id": "1"}}

## UI

st.subheader("AI ---- BOT")

query = st.chat_input("ASK Anything !!!")

for message in st.session_state.history:
    role,msg = message["role"],message["content"]
    st.chat_message(role).markdown(msg)
# while query:
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    res = agent.stream(
        {"messages":[{"role":"user","content":query}]},thread_config,stream_mode="messages"
    )
    #print(res)
    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""
        for chunk in res:
            message = message + chunk[0].content
            space.write(message)

        st.session_state.history.append({"role": "ai", "content": message})
    # answer = res["messages"][-1].content
    # st.chat_message("ai").markdown(answer)
    #st.session_state.history.append({"role": "ai", "content": answer})




