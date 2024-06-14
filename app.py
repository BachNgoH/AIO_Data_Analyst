import chainlit as cl
from dotenv import load_dotenv
from src.agents.pandasai_agent.agent import PandasAIAgent
from src.agents.llm_compiler_agent.agent import LLMCompilerAgent
from src.const import AGENT_TYPE

load_dotenv(override=True)

@cl.on_chat_start
async def on_chat_start():
    if AGENT_TYPE == "LLMCompilerAgent":
        await LLMCompilerAgent.aon_start()
    else:
        await PandasAIAgent.aon_start()

@cl.on_message
async def on_message(message: cl.Message):
    if AGENT_TYPE == "LLMCompilerAgent":
        await LLMCompilerAgent.aon_message(message)
    else:
        await PandasAIAgent.aon_message(message)
    

# @cl.on_file_upload
# async def on_file_upload(file: cl.File):
#     # Handle file upload logic here
#     await cl.Message(content=f"File {file.name} uploaded successfully!").send()
