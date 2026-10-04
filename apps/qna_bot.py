from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

llm_model = ChatHuggingFace(llm=llm)

# while True:
#     query = input("User : ")

#     if query.lower() in ["quit","exit","bye"]:
#         print("GoodBye ✌️!!")
#         break

#     res = llm_model.invoke(query)
#     print(f"AI : {res.content}")


st.title("Ask Buddy - AI QnA Bot")
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    role,content = msg["role"],msg["content"]
    st.chat_message(role).markdown(content)


query = st.chat_input("Ask Anything ?")
if query:
    #print(query)
    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message(f"user").markdown(query)
    res = llm_model.invoke(query)
    st.chat_message(f"ai").markdown(res.content)
    st.session_state.messages.append({"role":"ai","content":res.content})