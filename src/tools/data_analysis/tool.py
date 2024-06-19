import logging
import pandas as pd
from llama_index.core.llms import LLM 
from llama_index.core.base.response.schema import Response
from llama_index.core.tools import FunctionTool
from src.tools.data_analysis.output_parser import InstructionParser
from src.tools.data_analysis.prompts import (
    DEFAULT_PANDAS_PROMPT, 
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT, 
    DEFAULT_INSTRUCTION_STR,
    DEFAULT_PANDAS_EXCEPTION_PROMPT
)
import chainlit as cl
from chainlit import run_sync

class DataAnalysisToolSuite:
        
    def __init__(self, df: pd.DataFrame, llm : LLM) -> None:
        self._llm = llm
        self._pandas_prompt = DEFAULT_PANDAS_PROMPT
        self._pandas_exception_prompt = DEFAULT_PANDAS_EXCEPTION_PROMPT
        self._response_synthesis_prompt = DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        self._instruction_str = DEFAULT_INSTRUCTION_STR
        self._verbose = True
        self._synthesize_response = False
        self._instruction_parser = InstructionParser(df)
        self._df = df 
        self._head = 5

    def _get_table_context(self) -> str:
        """Get table context."""
        return str(self._df.head(self._head))

    def retry_generate_code(self, prev_code: str, exception_str: str):
        """Retry generating code."""
        """
        Generate code for a given query.

        Args:
            query_str (str): The query string in natural language.

        Returns:
            dict: A dictionary containing the response and metadata.
        """
        context = self._get_table_context()

        pandas_response_str = self._llm.predict(
            self._pandas_exception_prompt,
            df_str=context,
            query_str=prev_code,
            instruction_str=self._instruction_str,
            error_msg=exception_str
        )

        if self._verbose:
            logging.info(f"> Instructions:\n" f"\n{pandas_response_str}\n\n")
            pandas_response_str = "```\n" + pandas_response_str + "\n```\n" \
                if not pandas_response_str.startswith("```") else pandas_response_str
            run_sync(cl.Message(content=f"Generated Instructions:\n\n{pandas_response_str}\n").send())


        pandas_output = self._instruction_parser.parse(pandas_response_str)
        if self._verbose:
            logging.info(f"> Execution Output: {pandas_output}\n")
            run_sync(cl.Message(
                content=f"Execution Output: \n ```{pandas_output}```\n").send())


        response_metadata = {
            "pandas_instruction_str": pandas_response_str,
            "raw_pandas_output": pandas_output,
        }
        
        if self._synthesize_response:
            response_str = str(
                self._llm.predict(
                    self._response_synthesis_prompt,
                    query_str=prev_code,
                    pandas_instructions=pandas_response_str,
                    pandas_output=pandas_output,
                )
            )
        else:
            response_str = str(pandas_output)

        return Response(response=response_str, metadata=response_metadata)
    
    
    def generate_code(self, query_str) -> dict:
        """
        Generate code for a given query.

        Args:
            query_str (str): The query string in natural language.

        Returns:
            dict: A dictionary containing the response and metadata.
        """
        context = self._get_table_context()

        pandas_response_str = self._llm.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n" f"\n{pandas_response_str}\n\n")
            pandas_response_str = "```\n" + pandas_response_str + "\n```\n" \
                if not pandas_response_str.startswith("```") else pandas_response_str
            run_sync(cl.Message(content=f"Generated Instructions:\n\n{pandas_response_str}\n").send())

        response_metadata = {
            "instruction_str": pandas_response_str,
        }

        return Response(response=pandas_response_str, metadata=response_metadata)
    
    def execute_code(self, code_str: str) -> dict:
        """
        Execute code to analyze the dataframe.

        Args:
            code_str (str): The code string in Python.

        Returns:
            dict: A dictionary containing the response and metadata.
        """
        # context = self._get_table_context()

        pandas_output = self._instruction_parser.parse(code_str)

        if self._verbose:
            logging.info(f"> Execution Output: {pandas_output}\n")
            run_sync(cl.Message(content=f"Execution Output: \n {pandas_output}\n").send())

        response_metadata = {
            "raw_pandas_output": pandas_output,
        }

        return Response(response=pandas_output, metadata=response_metadata)

    def generate_and_run_code(self, query_str) -> dict:
        """
        Generate code for a given query and execute the code to analyze the dataframe.

        Args:
            query_str (str): The query string in natural language.

        Returns:
            dict: A dictionary containing the response and metadata.
        """
        context = self._get_table_context()

        pandas_response_str = self._llm.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n" f"\n{pandas_response_str}\n\n")
            pandas_response_str = "```\n" + pandas_response_str + "\n```\n" \
                if not pandas_response_str.startswith("```") else pandas_response_str
            
            run_sync(cl.Message(content=f"Generated Instructions:\n\n {pandas_response_str}\n").send())

        pandas_output = self._instruction_parser.parse(pandas_response_str)
        if self._verbose:
            logging.info(f"> Execution Output: {pandas_output}\n")
            run_sync(cl.Message(content=f"Execution Output: \n ```{pandas_output}```\n").send())


        response_metadata = {
            "pandas_instruction_str": pandas_response_str,
            "raw_pandas_output": pandas_output,
        }
        
        if self._synthesize_response:
            response_str = str(
                self._llm.predict(
                    self._response_synthesis_prompt,
                    query_str=query_str,
                    pandas_instructions=pandas_response_str,
                    pandas_output=pandas_output,
                )
            )
        else:
            response_str = str(pandas_output)

        return Response(response=response_str, metadata=response_metadata)


    async def agenerate_and_run_code(self, query_str) -> dict:
        """
        Generate code for a given query and execute the code to analyze the dataframe.

        Args:
            query_str (str): The query string in natural language.

        Returns:
            dict: A dictionary containing the response and metadata.
        """
        context = self._get_table_context()

        pandas_response_str = self._llm.apredict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n" f"```\n{pandas_response_str}\n```\n")
            await cl.Message(content=f"Generated Instructions:\n{pandas_response_str}\n").send()
            
        pandas_output = self._instruction_parser.parse(pandas_response_str)
        if self._verbose:
            logging.info(f"> Execution Output: {pandas_output}\n")
            await cl.Message(content=f"Execution Output: {pandas_output}\n").send()

        response_metadata = {
            "generated_code": pandas_response_str,
            "output": pandas_output,
        }
        
        if self._synthesize_response:
            response_str = str(
                self._llm.apredict(
                    self._response_synthesis_prompt,
                    query_str=query_str,
                    pandas_instructions=pandas_response_str,
                    pandas_output=pandas_output,
                )
            )
        else:
            response_str = str(response_metadata)

        return Response(response=response_str, metadata=response_metadata)


    def get_tools(self):
        """Get tools."""
        # return [FunctionTool.from_defaults(fn=self.generate_and_run_code, async_fn=self.agenerate_and_run_code)]
        return [
            FunctionTool.from_defaults(fn=self.execute_code),
            FunctionTool.from_defaults(fn=self.generate_and_run_code),
            FunctionTool.from_defaults(fn=self.retry_generate_code),
        ]