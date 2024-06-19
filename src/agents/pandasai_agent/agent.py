import os
import pandas as pd
import chainlit as cl
from pandasai import Agent
from pandasai.llm import OpenAI
from pandasai.llm.local_llm import LocalLLM
from pandasai.llm.google_gemini import GoogleGemini
from src.utils.chainlit_response import ChainlitResponse
from src.agents.base import BaseChainlitAgent
from src.const import LLM_PROVIDER

class PandasAIAgent(BaseChainlitAgent):
    
    def __init__(self):
        pass
    
    @classmethod
    async def aon_start(cls, *args, **kwargs):
        os.makedirs("charts", exist_ok=True)
        user_defined_path = "./charts"
        
        if LLM_PROVIDER == "groq":
            llm = LocalLLM(
                api_key=os.environ.get("GROQ_API_KEY"),
                model="llama3-8b-8192",
                api_base="https://api.groq.com/openai/v1",
            )
        elif LLM_PROVIDER == "openai":
            llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif LLM_PROVIDER == "gemini":
            llm  = GoogleGemini(api_key=os.environ.get("GOOGLE_API_KEY"))
        else:
            raise ValueError("Invalid LLM provider. The supported providers are groq, openai, and gemini.")
        
        df = pd.read_csv("./data/train.csv")
        agent = Agent(df, config={
            "llm": llm, 
            "response_parser": ChainlitResponse, 
            "verbose": True, 
            "save_charts": True,
            "save_charts_path": user_defined_path})
        cl.user_session.set("agent", agent)
    
    @classmethod
    async def aon_message(cls, message: cl.Message, *args, **kwargs):
        agent = cl.user_session.get("agent") 
    
        response = agent.chat(message.content)
        await cl.Message(content=response).send()
    
    @classmethod
    async def aon_resume(cls, f, *args, **kwargs):
        pass