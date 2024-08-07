import re
import io
import logging
import signal
import sys
import ast
import re
import traceback
import pytesseract
from PIL import Image

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

import importlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chainlit as cl
from enum import Enum
from typing import Any, Dict, Optional, List
# from paddleocr import PaddleOCR

# import holoviews as hv
# import pandas as pd
# from holoviews import opts
# from bokeh.io import output_notebook

from chainlit import run_sync
# from llama_index.experimental.exec_utils import safe_eval, safe_exec
from llama_index.core.output_parsers.base import ChainableOutputParser
from src.tools.data_analysis.prompts import DEFAULT_ANALYZE_PLOT_INSTRUCTION_STR
logger = logging.getLogger(__name__)


class ErrorHistory:
    def __init__(self):
        self.errors = []
    
    def add_error(self, error_message, code):
        self.errors.append({'error': error_message, 'code': code})
    
    def check_error(self, error_message):
        for record in self.errors:
            if record['error'] == error_message:
                return record['code']
        return None

error_history = ErrorHistory()
class TimeoutException(Exception):
    pass


class Status(Enum):
    NO_PLOT = "No plot"
    SHOW_PLOT_SUCCESS = "Show plot successfully!"
    SHOW_PLOT_FAILED = "Show plot failed!"

def parse_code_markdown(text: str, only_last: bool) -> List[str]:
    # Regular expression pattern to match code within triple-backticks with an optional programming language
    pattern = r"```[a-zA-Z]*\n(.*?)```"

    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the last matched group if requested
    code = matches[-1] if matches and only_last else matches

    # If empty, we optimistically assume the output is the code
    if not code:
        # Strip the text to handle cases where the code may start or end with triple backticks or quotes
        candidate = text.strip()

        # Handle cases where the code is surrounded by regular quotes
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1]

        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1]

        if candidate.startswith("`") and candidate.endswith("`"):
            candidate = candidate[1:-1]

        # Handle triple backticks at the start
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate)

        # Handle triple backticks at the end
        if candidate.endswith("```"):
            candidate = candidate[:-3]

        code = candidate.strip()

    return code
def extract_python_code(markdown_text)-> List[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    return "\n".join([match.strip() for match in matches])
def extract_python_codev2(markdown_text: str) -> str:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    extracted_code = "\n".join([match.strip() for match in matches])
    
    # Fix indentation issues
    fixed_code = []
    lines = extracted_code.split('\n')
    indent_level = 0
    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line.startswith("for ") or stripped_line.startswith("if "):
            indent_level += 1
            fixed_code.append(" " * 4 * (indent_level - 1) + stripped_line)
        elif line.strip() == "":
            fixed_code.append(line)
        else:
            fixed_code.append(" " * 4 * (indent_level) + stripped_line)
    return "\n".join(fixed_code)

def extract_python_codev4(markdown_text: str) -> str:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    if not matches:
        # If no Python code is found, return the original markdown text
        return markdown_text
    extracted_code = "\n".join([match.strip() for match in matches])
    
    # Fix indentation issues
    fixed_code = []
    lines = extracted_code.split('\n')
    indent_level = 0
    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line.startswith("for ") or stripped_line.startswith("if "):
            fixed_code.append(" " * 4 * (indent_level) + stripped_line)
            indent_level += 1
        elif line.strip() == "":
            fixed_code.append(line)
        else:
            fixed_code.append(" " * 4 * (indent_level) + stripped_line)
            indent_level = max(0, indent_level - line.count(" " * 4))
    
    # Remove plt.show() statements
    fixed_code = [line for line in fixed_code if not line.strip().startswith("plt.show()")]
    
    return "\n".join(fixed_code)
def show_plot() -> str:
    try:
        # Ensure there's a plot to display
        # if not plt.get_fignums():
        #     return Status.NO_PLOT
        
        # Create an in-memory bytes buffer for the plot image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Clear the plot
        plt.close()
        
        # Display the plot image
        image = cl.Image(
            name="plot", 
            size="large", 
            #display="inline", 
            content=buffer.getvalue())
        

        run_sync(cl.Message(
            content="",
            elements=[image]
        ).send())
        
        return Status.SHOW_PLOT_SUCCESS
    except Exception as e:
        return Status.SHOW_PLOT_FAILED
    
def show_plotv2(vision_llm:any) :
    try:
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        new_buffer=buffer.getvalue()

        plt.close()
        
        image = cl.Image(
            name="plot", 
            size="large", 
            display="inline", 
            content=new_buffer)
        
        run_sync(cl.Message(
            content="",
            elements=[image]
        ).send())

        buffer.seek(0)
        image_sequence = [Image.open(buffer)]

        description = vision_llm.complete(
        prompt=DEFAULT_ANALYZE_PLOT_INSTRUCTION_STR,
        images= image_sequence,
        )

        run_sync(cl.Message(
            content=f"\n\n{description}\n",

        ).send())

        return description
    except Exception as e:
        return Status.SHOW_PLOT_FAILED

def save_plot():#
    
    
    # Create an in-memory bytes buffer for the plot image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Clear the plot
    plt.close()
    
    return buffer

def timeout_handler(signum, frame):
    raise TimeoutException("Timed out!")

class InstructionParser(ChainableOutputParser):
    """Instruction parser for data analysis, model training, and evaluation."""

    def __init__(self, df: pd.DataFrame, error_history: List[Dict[str, str]] = None, output_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.df = df
        self.error_history = error_history if error_history is not None else []
        self.output_kwargs = output_kwargs or {}
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()

    def _prepare_data(self):
        X = self.df.drop(columns=[self.df.columns[-1]]).values
        y = self.df[self.df.columns[-1]].values
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def import_model(self, model_name: str):
        """Dynamically import a model class."""
        module_name, class_name = model_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def import_metric(self, metric_name: str):
        """Dynamically import a metric function."""
        module_name, func_name = metric_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        return self.default_output_processor(output, **self.output_kwargs)
    def parsev2(self, output: str,vision_llm:any) -> Any:
        """Parse, validate, and correct errors programmatically."""
        return self.output_processor_comprehensive_data_analysis(output, vision_llm,**self.output_kwargs)
    
    def default_output_processor(self, output: str, timeout: int = 1000, **output_kwargs: Any) -> str:
        """Process outputs in a default manner with a timeout."""
        if sys.version_info < (3, 9):
            logger.warning(
                "Python version must be >= 3.9 in order to use "
                "the default output processor, which executes "
                "the Python query. Instead, we will return the "
                "raw Python instructions as a string."
            )
            return output

        local_vars = {
            "df": self.df,
            "sns": sns,
            "plt": plt,
            "np": np,
            "pd": pd,
            "train_test_split": train_test_split,
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "import_model": self.import_model,
            "import_metric": self.import_metric
        }
        global_vars = {}

        #output = parse_code_markdown(output, only_last=False)
        output=extract_python_codev4(output)
        
        # Redirect standard output to capture print statements
        old_stdout = sys.stdout
        sys.stdout = new_stdout = io.StringIO()
        

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            tree = ast.parse(output)
            module = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(module, filename="<ast>", mode="exec"), {}, local_vars)  # type: ignore
            
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore
            print("get here before fignum for loop\n")
            if module_end_str.strip("'\"") != module_end_str:
                module_end_str = eval(module_end_str, global_vars, local_vars)
            
            try:
                
                output_str = str(eval(module_end_str, global_vars, local_vars))
                
                for fig_num in plt.get_fignums():
                    plt.figure(fig_num)
                    show_plot()
                    plt.close(fig_num)
                printed_output = new_stdout.getvalue()
                if printed_output:
                    output_str = printed_output + "\n" + output_str
               
                
                return output_str

            except Exception:
                raise
        except TimeoutException:
            return "The execution timed out. Please try again with optimized code or increase the timeout limit."
        except Exception as e:
            err_string = (
                "There was an error running the output as Python code. "
                f"Error message: {e}"
            )
            traceback.print_exc()
            self.error_history.append({"error": str(e), "code": output})
            return err_string
        finally:
            sys.stdout = old_stdout  # Restore standard output
            signal.alarm(0)  # Disable the alarm


    def output_processor_comprehensive_data_analysis(self, output: str, vision_llm:any,timeout: int = 2000, **output_kwargs: Any) -> str:
        """Process outputs in a default manner with a timeout."""
        if sys.version_info < (3, 9):
            logger.warning(
                "Python version must be >= 3.9 in order to use "
                "the default output processor, which executes "
                "the Python query. Instead, we will return the "
                "raw Python instructions as a string."
            )
            return output

        local_vars = {
            "df": self.df,
            "sns": sns,
            "plt": plt,
            "np": np,
            "pd": pd,
            "train_test_split": train_test_split,
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "import_model": self.import_model,
            "import_metric": self.import_metric,
            "show_plot": show_plot

        }
        global_vars = {}

        output = extract_python_codev4(output)
        
        # Redirect standard output to capture print statements
        old_stdout = sys.stdout
        sys.stdout = new_stdout = io.StringIO()
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            tree = ast.parse(output)
            module = ast.Module(body=tree.body[:-1], type_ignores=[])

            
            exec(compile(module, filename="<ast>", mode="exec"), {}, local_vars)  # type: ignore
            
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore
            
            if module_end_str.strip("'\"") != module_end_str:
                module_end_str = eval(module_end_str, global_vars, local_vars)
            

            try:
                
                output_str = str(eval(module_end_str, global_vars, local_vars))
                descriptions=[]
                for fig_num in plt.get_fignums(): 
                    plt.figure(fig_num)
                    descriptions.append(show_plotv2(vision_llm))
                    plt.close(fig_num)
                
                printed_output = new_stdout.getvalue()
          

                if printed_output:
                    output_str = printed_output + "\n" + output_str
                
                
                return output_str,descriptions

            except Exception:
                raise
        except TimeoutException:
            return "The execution timed out. Please try again with optimized code or increase the timeout limit."
        except Exception as e:
            err_string = (
                "There was an error running the output as Python code. "
                f"Error message: {e}"
            )
            traceback.print_exc()
            self.error_history.append({"error": str(e), "code": output})
            return err_string
        finally:
            sys.stdout = old_stdout  # Restore standard output
            signal.alarm(0)  # Disable the alarm