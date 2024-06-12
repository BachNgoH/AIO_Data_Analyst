import chainlit as cl
import pandas as pd
from pandasai import Agent
from pandasai.llm import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

@cl.on_chat_start
async def on_chat_start():
    llm = OpenAI(
        api_token=os.environ.get("OPENAI_API_KEY"),
    )
    df = pd.read_csv("./data/train.csv")
    agent = Agent(df, config={"llm": llm})
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
