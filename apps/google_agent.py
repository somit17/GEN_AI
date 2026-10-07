from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import  GoogleSerperAPIWrapper
from langchain.agents import  create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
llm = ChatGroq(
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

agent = create_agent(
                     model=llm,
                     tools=[search.run],
                     system_prompt="You're an agent and can search anything on google",
                     checkpointer=MemorySaver()
                     )
thread_config = {"configurable": {"thread_id": "1"}}
while True:
    query = input("User : ")

    if query.lower() in ["quit","exit","bye"]:
        print("GoodBye ✌️!!")
        break

    res = agent.invoke({"messages":[{"role": "user", "content": query}]},thread_config)
    print(res["messages"][-1].content)