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

load_dotenv(override=True)

class LLMCompilerAgent(BaseChainlitAgent):
    
    _agent: AgentRunner
    _df_path: str
    _AGENT_IDENTIFIER: str = "LLMAnalyzerAgent"
    _HISTORY_IDENTIFIER: str = f"{_AGENT_IDENTIFIER}_chat_history"
    
    
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
        fixed_path = "./data/dataframe.csv"
    
        # Lưu file với tên và đường dẫn cố định
        with open(fixed_path, 'wb') as f:
            with open(text_file.path, 'rb') as temp_file:
                f.write(temp_file.read())  # Đọc nội dung từ file tạm thời và ghi vào file cố định
        
        LLMCompilerAgent._df_path = fixed_path
        
        df = pd.read_csv(text_file.path)
        
        await cl.Message(f"{df.head().to_markdown()}\n\nFile uploaded successfully! Ask anything about the data!").send()
        return text_file.path
        
    
    @classmethod
    async def aon_start(cls, *args, **kwargs):
        LLMCompilerAgent._set_chat_history([])
        
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
        response = LLMCompilerAgent._agent.stream_chat(content)
        msg = cl.Message(content = "")
        for token in response.response_gen:
            await msg.stream_token(token)
        await msg.send()
        
        chat_history.append({
            "content": msg.content,
            "role": MessageRole.ASSISTANT
        })
        LLMCompilerAgent._set_chat_history(chat_history)
                
    