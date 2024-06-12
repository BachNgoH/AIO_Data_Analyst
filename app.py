import chainlit as cl
import pandas as pd
from pandasai import Agent
from pandasai.llm import OpenAI
from pandasai.llm.local_llm import LocalLLM
from pandasai.llm.google_gemini import GoogleGemini
from chainlit_response import ChainlitResponse
from dotenv import load_dotenv
import os

load_dotenv(override=True)

@cl.on_chat_start
async def on_chat_start():
    os.makedirs("charts", exist_ok=True)
    user_defined_path = os.path.join(os.getcwd(), "charts")
    
    
    llm = LocalLLM(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama3-8b-8192",
        api_base="https://api.groq.com/openai/v1",
    )
    
    df = pd.read_csv("./data/train.csv")
    agent = Agent(df, config={
        "llm": llm, 
        "response_parser": ChainlitResponse, 
        "verbose": True, 
        "save_charts": True,
        "save_charts_path": user_defined_path})
    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent") 
    
    response = agent.chat(message.content)
    await cl.Message(content=response).send()

# @cl.on_file_upload
# async def on_file_upload(file: cl.File):
#     # Handle file upload logic here
#     await cl.Message(content=f"File {file.name} uploaded successfully!").send()
