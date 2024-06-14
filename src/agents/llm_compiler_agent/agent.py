import chainlit as cl
from llama_index.agent.llm_compiler import LLMCompilerAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from src.agents.base import BaseChainlitAgent
from src.utils.llm_utils import load_model

load_dotenv(override=True)

class LLMCompilerAgent(BaseChainlitAgent):
    
    _agent: AgentRunner
    
    @staticmethod
    def _init_tools():
        from src.tools.pandas_tool import load_pandas_tool
        return load_pandas_tool()
    
    @classmethod
    async def aon_start(cls, *args, **kwargs):
        llm = load_model()
        tools = cls._init_tools()

        agent_worker = LLMCompilerAgentWorker.from_tools(
            tools, llm=llm, verbose=True
        )
        agent = AgentRunner(agent_worker)
        cls._agent = agent
        await cl.Message(content="Ask me anything about the dataframe!!").send()
    
    @classmethod
    async def aon_message(cls, message: cl.Message, *args, **kwargs):
        content = message.content
        response = cls._agent.chat(content)
        await cl.Message(content=response.response).send()
                
                
    