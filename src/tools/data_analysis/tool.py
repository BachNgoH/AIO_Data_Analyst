import logging
import pandas as pd
import io
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50

from llama_index.core.llms import LLM 
from llama_index.core.base.response.schema import Response
from llama_index.core.tools import FunctionTool
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from src.tools.data_analysis.output_parser import InstructionParser
from src.tools.data_analysis.prompts import (
    DEFAULT_PANDAS_PROMPT, 
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT, 
    DEFAULT_INSTRUCTION_STR,
    DEFAULT_PANDAS_EXCEPTION_PROMPT,
    DEFAULT_MODEL_PROMPT,
    DEFAULT_MODEL_EXCEPTION_PROMPT,
    DEFAULT_MODEL_RESPONSE_SYNTHESIS_PROMPT,
    DEFAULT_MODEL_INSTRUCTION_STR,
    DEFAULT_COMPREHENSIVE_ANALYSIS_PROMPT,
    DEFAULT_COMPREHENSIVE_ANALYSIS_INSTRUCTION_STR,
    DEFAULT_SUMMARIZE_PROMPT,
    DEFAULT_SUMMARIZE_INSTRUCTION_STR,
    DEFAULT_ANALYZE_PLOT_PROMPT,
    DEFAULT_ANALYZE_PLOT_INSTRUCTION_STR
)
import chainlit as cl
from chainlit import run_sync

def is_error_message(output: str) -> bool:
    """Check if the output contains an error message."""
    error_indicators = [
        "Traceback",
        "Error:",
        "Exception:",
        "There was an error running the output as Python code"
    ]
    return any(indicator in output for indicator in error_indicators)


def format_analysis_output(output: str) -> str:
    """
    Format the analysis output for better presentation.

    Args:
        output (str): The raw output string.

    Returns:
        str: The formatted output string.
    """
    # Ensure each step and section is clearly separated and formatted.
    formatted_output = output

    # Example formatting: Add extra newlines before headers and code blocks.
    formatted_output = formatted_output.replace('### ', '\n### ')
    formatted_output = formatted_output.replace('```python', '\n```python')
    formatted_output = formatted_output.replace('```\n', '```\n\n')
    
    return formatted_output





# Main function
class DataAnalysisToolSuite:
    
    def __init__(self, df: pd.DataFrame, llm: any) -> None:
        self._df = df
        self._llm = llm
        self._verbose = True
        self._instruction_parser = InstructionParser(df)
        self._head = 5

        # Prompts and instructions
        self._pandas_prompt = DEFAULT_PANDAS_PROMPT
        self._pandas_exception_prompt = DEFAULT_PANDAS_EXCEPTION_PROMPT
        self._response_synthesis_prompt = DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        self._instruction_str = DEFAULT_INSTRUCTION_STR
        self._model_prompt = DEFAULT_MODEL_PROMPT
        self._model_exception_prompt = DEFAULT_MODEL_EXCEPTION_PROMPT
        self._model_response_synthesis_prompt = DEFAULT_MODEL_RESPONSE_SYNTHESIS_PROMPT
        self._model_instruction_str = DEFAULT_MODEL_INSTRUCTION_STR
        self._comprehensive_analysis_prompt = DEFAULT_COMPREHENSIVE_ANALYSIS_PROMPT
        self._comprehensive_analysis_prompt_str = DEFAULT_COMPREHENSIVE_ANALYSIS_INSTRUCTION_STR
        self._summarize_prompt = DEFAULT_SUMMARIZE_PROMPT
        self._summarize_prompt_instruction_str= DEFAULT_SUMMARIZE_INSTRUCTION_STR
        self._plot_analysis_prompt= DEFAULT_ANALYZE_PLOT_PROMPT
        self._plot_analysis_prompt_instruction_str=DEFAULT_ANALYZE_PLOT_INSTRUCTION_STR
        self._synthesize_response = False
        self.error_history = []

    def _get_table_context(self) -> str:
        """Get table context."""
        return str(self._df.head(self._head))

    def generate_description(self,text, features):
        generated_description= self._llm.predict(
            self._plot_analysis_prompt,
            instruction_str= self._plot_analysis_prompt_instruction_str,
            extracted_text= text,
            visual_features=features

        )
        return generated_description

    
    def summarize_outputv2(self, content1: str,content2: str) -> str:
        """Summarize the provided content using the default LLM."""
        summary_response = self._llm.predict(
            self._summarize_prompt,
            instruction_str=self._summarize_prompt_instruction_str,
            content1=content1,
            content2=content2

        )
   
        return summary_response
   
    def summarize_output(self, content: str) -> str:
        """Summarize the provided content using the default LLM."""
        summary_response = self._llm.predict(
            self._summarize_prompt,
            instruction_str=self._summarize_prompt_instruction_str,
            content=content,
     
        )
   
        return summary_response

    def retry_generate_data_analysis_code(self, prev_code: str, exception_str: str) -> dict:
        """Retry generating code."""
        context = self._get_table_context()

        pandas_response_str = self._llm.predict(
            self._pandas_exception_prompt,
            df_str=context,
            query_str=prev_code,
            instruction_str=self._instruction_str,
            error_msg=exception_str
        )

        if self._verbose:
            logging.info(f"> Instructions:\n\n{pandas_response_str}\n\n")
            pandas_response_str = "'''\n" + pandas_response_str + "\n'''\n" if not pandas_response_str.startswith("'''") else pandas_response_str
            #pandas_response_str= self.summarize_output(pandas_response_str)
            run_sync(cl.Message(content=f"Generated Instructions:\n\n{pandas_response_str}\n").send())

        pandas_output = self._instruction_parser.parse(pandas_response_str)
        pandas_output= self.summarize_output(pandas_output)
        if self._verbose:
            logging.info(f"> Execution Output: {pandas_output}\n")
            run_sync(cl.Message(content=f"Execution Output: \n'''{pandas_output}'''\n").send())

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
    
    def generate_code(self, query_str: str) -> dict:
        """Generate code for a given query."""
        context = self._get_table_context()

        pandas_response_str = self._llm.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n\n{pandas_response_str}\n\n")
            pandas_response_str = "```\n" + pandas_response_str + "\n```\n" if not pandas_response_str.startswith("```") else pandas_response_str
            run_sync(cl.Message(content=f"Generated Instructions:\n\n{pandas_response_str}\n").send())

        response_metadata = {
            "instruction_str": pandas_response_str,
        }

        return Response(response=pandas_response_str, metadata=response_metadata)
    
    def execute_data_analysis_code(self, code_str: str) -> dict:
        """Execute code to analyze the dataframe."""
        pandas_output = self._instruction_parser.parse(code_str)

        if self._verbose:
            logging.info(f"> Execution Output: {pandas_output}\n")
            run_sync(cl.Message(content=f"Execution Output: \n {pandas_output}\n").send())

        response_metadata = {
            "raw_pandas_output": pandas_output,
        }

        return Response(response=pandas_output, metadata=response_metadata)

    def generate_and_run_data_analysis_code(self, query_str: str) -> dict:
        """Generate code for a given query and execute the code to analyze the dataframe."""
        context = self._get_table_context()

        pandas_response_str = self._llm.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n\n{pandas_response_str}\n\n")
            pandas_response_str = "'''\n" + pandas_response_str + "\n'''\n" if not pandas_response_str.startswith("'''") else pandas_response_str
            #pandas_response_str= self.summarize_output(pandas_response_str)
            run_sync(cl.Message(content=f"Generated Instructions:\n\n {pandas_response_str}\n").send())

        pandas_output = self._instruction_parser.parsev2(pandas_response_str,self._llm)
        pandas_output=self.summarize_output(pandas_output)
        if self._verbose:
            logging.info(f"> Execution Output: {pandas_output}\n")
            run_sync(cl.Message(content=f"Execution Output: \n '''{pandas_output}'''\n").send())

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

    async def agenerate_and_run_code(self, query_str: str) -> dict:
        """Generate code for a given query and execute the code to analyze the dataframe asynchronously."""
        context = self._get_table_context()

        pandas_response_str = await self._llm.apredict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n```\n{pandas_response_str}\n```\n")
            await cl.Message(content=f"Generated Instructions:\n{pandas_response_str}\n").send()
        
        pandas_output = self._instruction_parser.parse(pandas_response_str)
        if self._verbose:
            logging.info(f"> Execution Output: {pandas_output}\n")
            await cl.Message(content=f"Execution Output: {pandas_output}\n").send()

        response_metadata = {
            "generated_code": pandas_response_str,
            "output": pandas_output,
        }
        
        response_str = str(response_metadata)

        return Response(response=response_str, metadata=response_metadata)
    
    def analyze_data(self, target_column: str) -> str:
        """Analyze the dataframe to determine the most suitable model type."""
        unique_values = self._df[target_column].nunique()

        if unique_values <= 10:
            return "classification"
        else:
            return "regression"

    def generate_model_code(self, query_str: str) -> dict:
        """Generate code for a given query to build a model."""
        context = self._get_table_context()

        model_response_str = self._llm.predict(
            self._model_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._model_instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n{model_response_str}\n")
            model_response_str = f"'''\n{model_response_str}\n'''\n" if not model_response_str.startswith("'''") else model_response_str
            run_sync(cl.Message(content=f"Generated Instructions:\n\n{model_response_str}\n").send())

        response_metadata = {
            "instruction_str": model_response_str,
        }

        return Response(response=model_response_str, metadata=response_metadata)

    def retry_generate_model_code(self, prev_code: str, exception_str: str) -> dict:
        """Retry generating model code after an error."""
        context = self._get_table_context()

        # Add the exception to the error history
        self.error_history.append(exception_str)

        # Format the error history for the LLM prompt
        error_history_str = "\n".join(f"- {err}" for err in self.error_history)

        model_response_str = self._llm.predict(
            self._model_exception_prompt,
            df_str=context,
            query_str=prev_code,
            instruction_str=self._model_instruction_str,
            error_msg=exception_str,
            error_history=error_history_str
        )

        if self._verbose:
            logging.info(f"> Instructions:\n{model_response_str}\n")
            model_response_str = f"'''\n{model_response_str}\n'''\n" if not model_response_str.startswith("'''") else model_response_str
            model_output = self._instruction_parser.parse(model_response_str)
            if not is_error_message(model_output):
                run_sync(cl.Message(content=f"Generated Instructions:\n\n{model_response_str}\n").send())
        else:
            model_output = self._instruction_parser.parse(model_response_str)
        if self._verbose:
            logging.info(f"> Execution Output: {model_output}\n")
            if not is_error_message(model_output):
                run_sync(cl.Message(content=f"Execution Output: \n```{model_output}```\n").send())

        response_metadata = {
            "model_instruction_str": model_response_str,
            "raw_model_output": model_output,
        }
        
        response_str = str(model_output)

        return Response(response=response_str, metadata=response_metadata)

    def execute_model_code(self, code_str: str) -> dict:
        """Execute code to train and evaluate the model."""
        model_output = self._instruction_parser.parse(code_str)

        if self._verbose:
            logging.info(f"> Execution Output: {model_output}\n")
            run_sync(cl.Message(content=f"Execution Output: \n{model_output}\n").send())

        response_metadata = {
            "raw_model_output": model_output,
        }

        return Response(response=model_output, metadata=response_metadata)

    def generate_and_run_model_code(self, query_str: str) -> dict:
        """Generate code for a given query and execute the code to train and evaluate the model."""
        context = self._get_table_context()

        model_response_str = self._llm.predict(
            self._model_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._model_instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n{model_response_str}\n")
            model_response_str = f"'''\n{model_response_str}\n'''\n" if not model_response_str.startswith("'''") else model_response_str
            model_output = self._instruction_parser.parse(model_response_str)
            if not is_error_message(model_output):
                run_sync(cl.Message(content=f"Generated Instructions:\n\n{model_response_str}\n").send())
        else:
            model_output = self._instruction_parser.parse(model_response_str)
        if self._verbose:
            logging.info(f"> Execution Output: {model_output}\n")
            if not is_error_message(model_output):
                run_sync(cl.Message(content=f"Execution Output: \n'''{model_output}'''\n").send())

        response_metadata = {
            "model_instruction_str": model_response_str,
            "raw_model_output": model_output,
        }
        
        response_str = str(model_output)

        return Response(response=response_str, metadata=response_metadata)

    def generate_comprehensive_analysis(self,query_str:str) -> dict:
        """Generate comprehensive data analysis and model building code."""
        context = self._get_table_context()
        # query_str = (
        #     "Perform a full exploratory data analysis (EDA) on the provided dataset. "
        #     "Include data loading, preprocessing, visualization, and feature engineering steps. "
        #     "Provide detailed comments and explanations for each step, including insights and interpretations. "
        #     "Summarize the findings and suggest next steps or further analysis."
        # )
        analysis_response_str = self._llm.predict(
            self._comprehensive_analysis_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._comprehensive_analysis_prompt_str,
        )
        

        if self._verbose:
            logging.info(f"> Comprehensive Analysis Instructions:\n{analysis_response_str}\n")
            analysis_response_str = f"'''\n{analysis_response_str}\n'''\n" if not analysis_response_str.startswith("'''") else analysis_response_str
            # analysis_response_str=format_analysis_output(analysis_response_str)
            run_sync(cl.Message(content=f"Generated Comprehensive Analysis Instructions:\n\n\n{analysis_response_str}\n").send())


        analysis_output= self._instruction_parser.parsev2(analysis_response_str,self._llm)
        #analysis_output= self._instruction_parser.parse(analysis_response_str)
        analysis_output= self.summarize_output(analysis_output)
        #print(f"analysis_output:{analysis_output}")
        if self._verbose:
            logging.info(f"> Comprehensive Analysis Execution Output: {analysis_output}\n")
            analysis_output = f"'''\n{analysis_output}\n'''\n" if not analysis_output.startswith("'''") else analysis_output
            run_sync(cl.Message(content=f"Comprehensive Analysis Execution Output: \n\n\n{analysis_output}\n").send())

        response_metadata = {
            "analysis_instruction_str": analysis_response_str,
            "raw_analysis_output": analysis_output,
        }
        
        response_str = str(analysis_output)

        return Response(response=response_str, metadata=response_metadata)
    def retry_generate_comprehensive_analysis(self,prev_bug:str) -> dict:
        """Generate comprehensive data analysis and model building code."""
        context = self._get_table_context()
        
        analysis_response_str = self._llm.predict(
            self._comprehensive_analysis_prompt,
            df_str=context,
            query_str=prev_bug,
            instruction_str=self._comprehensive_analysis_prompt_str,
        )
        

        if self._verbose:
            logging.info(f"> Comprehensive Analysis Instructions:\n{analysis_response_str}\n")
            analysis_response_str = f"'''\n{analysis_response_str}\n'''\n" if not analysis_response_str.startswith("'''") else analysis_response_str
            # analysis_response_str=format_analysis_output(analysis_response_str)
            run_sync(cl.Message(content=f"Generated Comprehensive Analysis Instructions:\n\n\n{analysis_response_str}\n").send())

    
        analysis_output = self._instruction_parser.parsev2(analysis_response_str)
        summary_output= self.summarize_output(analysis_output)
        #print(f"analysis_output:{analysis_output}")
        if self._verbose:
            logging.info(f"> Comprehensive Analysis Execution Output: {summary_output}\n")
            summary_output = f"'''\n{summary_output}\n'''\n" if not summary_output.startswith("'''") else summary_output
            run_sync(cl.Message(content=f"Comprehensive Analysis Execution Output: \n{summary_output}\n").send())

        response_metadata = {
            "analysis_instruction_str": analysis_response_str,
            "raw_analysis_output": analysis_output,
        }
        
        response_str = str(analysis_output)

        return Response(response=response_str, metadata=response_metadata)
   

    async def agenerate_and_run_code(self, query_str: str) -> dict:
        """Generate code for a given query and execute the code to train and evaluate the model asynchronously."""
        context = self._get_table_context()

        model_response_str = await self._llm.apredict(
            self._model_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._model_instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n{model_response_str}\n")
            await cl.Message(content=f"Generated Instructions:\n{model_response_str}\n").send()
        
        model_output = self._instruction_parser.parse(model_response_str)
        if self._verbose:
            logging.info(f"> Execution Output: {model_output}\n")
            await cl.Message(content=f"Execution Output: {model_output}\n").send()

        response_metadata = {
            "generated_code": model_response_str,
            "output": model_output,
        }
        
        response_str = str(response_metadata)

        return Response(response=response_str, metadata=response_metadata)

    def get_tools(self):
        """Get tools."""
        return [
            # FunctionTool.from_defaults(fn=self.execute_data_analysis_code),
            FunctionTool.from_defaults(fn=self.generate_and_run_data_analysis_code),
            FunctionTool.from_defaults(fn=self.retry_generate_data_analysis_code),
            # FunctionTool.from_defaults(fn=self.execute_model_code),
            FunctionTool.from_defaults(fn=self.generate_and_run_model_code),
            FunctionTool.from_defaults(fn=self.retry_generate_model_code),
            FunctionTool.from_defaults(fn=self.generate_comprehensive_analysis),
            FunctionTool.from_defaults(fn=self.retry_generate_comprehensive_analysis)
        ]
