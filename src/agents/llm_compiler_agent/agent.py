from typing import List
import chainlit as cl
import pandas as pd
from llama_index.core.agent import AgentRunner, ReActAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from dotenv import load_dotenv
from src.agents.base import BaseChainlitAgent
from src.utils.llm_utils import load_model
from .prompts import WELCOME_MESSAGE, BASE_SYSTEM_PROMPT, SYSTEM_PROMPT
from src.const import MAX_ITERATIONS
# from src.tools.data_analysis.model_selection import analyze_data, select_model
from src.tools.data_analysis.tool import DataAnalysisToolSuite
from PIL import Image
load_dotenv(override=True)

class LLMCompilerAgent(BaseChainlitAgent):
    
    agent: AgentRunner
    _df_path: str
    _AGENT_IDENTIFIER: str = "LLMAnalyzerAgent"
    _HISTORY_IDENTIFIER: str = f"{_AGENT_IDENTIFIER}_chat_history"
    _df: pd.DataFrame

    
    @staticmethod
    def _get_chat_history() -> list[dict]:
        chat_history = cl.user_session.get(key=LLMCompilerAgent._HISTORY_IDENTIFIER, default=[])
        return chat_history

    @staticmethod
    def _set_chat_history(chat_history: list[dict]) -> None:
        cl.user_session.set(key=LLMCompilerAgent._HISTORY_IDENTIFIER, value=chat_history)
    
    @classmethod
    def _construct_message_history(self, message_history: List[dict] = None) -> List[ChatMessage]:
        self._agent.memory.reset()
        memory = [
            ChatMessage(content=BASE_SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
        ]

        if message_history:
            memory.extend([ChatMessage(**message) for message in message_history])

        self._agent.memory.set(messages=memory)
        return memory
        
    @classmethod
    def _init_tools(cls):
        llm= load_model()
        data_tools = DataAnalysisToolSuite(cls._df, llm=llm).get_tools()
        return data_tools
    @classmethod
    async def _ask_file_handler(cls):
        files = None

        # Wait for the user to upload a file
        while files is None:
            files = await cl.AskFileMessage(
                content="Please upload a file for analysis.", 
                accept=[
                    "text/csv", 
                    "text/plain", 
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                    "image/png", 
                    "image/jpeg",
                    "application/octet-stream"  # Common MIME type for .dat files
                ],
                max_size_mb=25
            ).send()

        uploaded_file = files[0]
        LLMCompilerAgent._df_path = uploaded_file.path
        file_path= uploaded_file.path
        file_name= uploaded_file.name

        if file_name.endswith(".csv"):
            # Handle CSV file
            LLMCompilerAgent._df = pd.read_csv(file_path)
            await cl.Message(content=f"{LLMCompilerAgent._df.head().to_markdown()}\n\nCSV file uploaded successfully!").send()

        elif file_name.endswith(".txt"):
            # Handle text file
            with open(file_path, 'r') as file:
                content = file.read()
            await cl.Message(content=f"Text file content:\n\n{content}").send()

        elif file_name.endswith(".xlsx"):
            # Handle Excel file
            LLMCompilerAgent._df = pd.read_excel(file_path)
            await cl.Message(content=f"{LLMCompilerAgent._df.head().to_markdown()}\n\nExcel file uploaded successfully!").send()

        elif file_name.endswith(".png") or file_name.endswith(".jpeg") or file_name.endswith(".jpg"):
            # Handle image file
            img = Image.open(file_path)
            img.show()
            await cl.Message(content=f"Image file uploaded successfully!").send()

        elif file_name.endswith(".dat"):
            # Handle .dat file (assuming it's a text-based .dat file)
            with open(file_path, 'r') as file:
                content = file.read()
            await cl.Message(content=f".dat file content:\n\n{content}").send()

        # df = pd.read_csv(text_file.path)
        
        # await cl.Message(f"{df.head().to_markdown()}\n\nFile uploaded successfully!").send()
        
        
        return file_path
        
    @classmethod
    async def aon_start(cls, *args, **kwargs):
        LLMCompilerAgent._set_chat_history([])
        
        llm = load_model()

        await LLMCompilerAgent._ask_file_handler()
        tools = LLMCompilerAgent._init_tools()

        agent = ReActAgent.from_tools(
            tools, llm=llm, verbose=True, max_iterations=MAX_ITERATIONS
        )
        LLMCompilerAgent._agent = agent
        cl.user_session.set(LLMCompilerAgent._AGENT_IDENTIFIER, agent)

    
    @classmethod
    async def aon_message(cls, message: cl.Message, *args, **kwargs):
        chat_history = LLMCompilerAgent._get_chat_history()
        LLMCompilerAgent._construct_message_history(chat_history)

        chat_history.append({
            "content": message.content,
            "role": MessageRole.USER
        })

        content = message.content
        # await LLMCompilerAgent.show_action_buttons()
        # Pass the user prompt to the LLM
        response = LLMCompilerAgent._agent.chat(content)
        markdown_response = f"\n{response}\n"
    
        msg = cl.Message(content=markdown_response)
        await msg.send()
        # msg = cl.Message(content=response)
        # await msg.send()

        chat_history.append({
            "content": msg.content,
            "role": MessageRole.ASSISTANT
        })
        LLMCompilerAgent._set_chat_history(chat_history)