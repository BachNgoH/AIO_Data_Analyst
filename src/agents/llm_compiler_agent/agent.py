import chainlit as cl
from llama_index.agent.llm_compiler import LLMCompilerAgentWorker
from llama_index.core.agent import AgentRunner, ReActAgent
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from src.agents.base import BaseChainlitAgent
from src.utils.llm_utils import load_model
from .prompts import WELCOME_MESSAGE
from src.const import MAX_ITERATIONS

load_dotenv(override=True)

class LLMCompilerAgent(BaseChainlitAgent):
    
    _agent: AgentRunner
    _df_path: str
    
    @classmethod
    def _init_tools(cls):
        from src.tools.pandas_tool import load_pandas_tool
        return load_pandas_tool(df_path=LLMCompilerAgent._df_path)
    
    @classmethod
    async def _ask_file_handler(cls):
        files = None

        # Wait for the user to upload a file
        while files == None:
            files = await cl.AskFileMessage(
                content=WELCOME_MESSAGE, 
                accept=["text/csv"], 
                max_size_mb=25
            ).send()

        text_file = files[0]
        LLMCompilerAgent._df_path = text_file.path
        return text_file.path
        
    
    @classmethod
    async def aon_start(cls, *args, **kwargs):
        
        llm = load_model()

        await LLMCompilerAgent._ask_file_handler()
        tools = LLMCompilerAgent._init_tools()

        # agent_worker = LLMCompilerAgentWorker.from_tools(
        #     tools, llm=llm, verbose=True
        # )
        # agent = AgentRunner(agent_worker)
        agent = ReActAgent.from_tools(
            tools, llm=llm, verbose=True, max_iterations=MAX_ITERATIONS
        )
        LLMCompilerAgent._agent = agent
    
    @classmethod
    async def aon_message(cls, message: cl.Message, *args, **kwargs):
        content = message.content
        response = LLMCompilerAgent._agent.stream_chat(content)
        msg = cl.Message(content = "")
        for token in response.response_gen:
            await msg.stream_token(token)
        await msg.send()                
                
    