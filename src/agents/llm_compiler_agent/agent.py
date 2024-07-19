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
import re
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
        LLMCompilerAgent._df_path = text_file.path
        
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
        
        # Lấy lịch sử chat từ phiên người dùng
        chat_history = LLMCompilerAgent._get_chat_history()
        # Xây dựng lại lịch sử tin nhắn
        LLMCompilerAgent._construct_message_history(chat_history)

        # Thêm tin nhắn của người dùng vào lịch sử
        chat_history.append({
            "content": message.content,
            "role": MessageRole.USER
        })
        
        # Nội dung tin nhắn từ người dùng
        content = message.content
        # Nhận phản hồi từ agent
        response = LLMCompilerAgent._agent.stream_chat(content)
        
        # Khởi tạo tin nhắn phản hồi từ agent
        response_content = ""
        
        # Xử lý từng token trong phản hồi
        for token in response.response_gen:
            response_content += token
        
        # Sử dụng regex để lấy phần sau "Answer: "
        match = re.search(r'Answer: (.*)', response_content)
        if match:
            response_content = match.group(1).strip()
        
        # Gửi tin nhắn phản hồi sau khi nhận đủ toàn bộ phản hồi từ agent
        msg = cl.Message(content=response_content)
        await msg.send()
        
        # Thêm phản hồi của agent vào lịch sử
        chat_history.append({
            "content": response_content,
            "role": MessageRole.ASSISTANT
        })
        
        # Cập nhật lại lịch sử chat trong phiên người dùng
        LLMCompilerAgent._set_chat_history(chat_history)
                    
    