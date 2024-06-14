import chainlit as cl

class BaseChainlitAgent:
    
    @classmethod  
    async def aon_start(cls, *args, **kwargs):
        pass
    
    @classmethod
    async def aon_message(cls, message: cl.Message, *args, **kwargs):
        pass
    
    @classmethod
    async def aon_resume(cls, *args, **kwargs):
        pass