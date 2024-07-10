import re
import io
import logging
import signal
import sys
import ast
import re
import traceback


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chainlit as cl
from enum import Enum
from typing import Any, Dict, Optional, List

from chainlit import run_sync
# from llama_index.experimental.exec_utils import safe_eval, safe_exec
from llama_index.core.output_parsers.base import ChainableOutputParser
logger = logging.getLogger(__name__)

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

def show_plot() -> str:
    try:
        # Ensure there's a plot to display
        if not plt.get_fignums():
            return Status.NO_PLOT
        
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
            display="inline", 
            content=buffer.getvalue())
        
        run_sync(cl.Message(
            content="",
            elements=[image]
        ).send())
        
        return Status.SHOW_PLOT_SUCCESS
    except Exception as e:
        return Status.SHOW_PLOT_FAILED


def timeout_handler(signum, frame):
    raise TimeoutException("Timed out!")

class InstructionParser(ChainableOutputParser):
    """instruction parser.

    This 'output parser' takes in pandas instructions (in Python code) and
    executes them to return an output.

    """

    def __init__(
        self, df: pd.DataFrame, output_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize params."""
        self.df = df
        self.output_kwargs = output_kwargs or {}

    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        return self.default_output_processor(output, **self.output_kwargs)


    def default_output_processor(self, output: str, timeout: int = 30, **output_kwargs: Any) -> str:
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
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "train_test_split": train_test_split,
            "StandardScaler": StandardScaler,
            "SVC": SVC,
            "LinearRegression": LinearRegression,
            "mean_squared_error": mean_squared_error,
            "accuracy_score": accuracy_score,
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix
        }
        global_vars = {
            "df": self.df,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "train_test_split": train_test_split,
            "StandardScaler": StandardScaler,
            "SVC": SVC,
            "LinearRegression": LinearRegression,
            "mean_squared_error": mean_squared_error,
            "accuracy_score": accuracy_score,
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix
        }

        output = parse_code_markdown(output, only_last=True)
        
        # Redirect standard output to capture print statements
        old_stdout = sys.stdout
        sys.stdout = new_stdout = io.StringIO()
        
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            tree = ast.parse(output)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(ast.unparse(module), {}, local_vars)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore
            
            if module_end_str.strip("'\"") != module_end_str:
                # if there's leading/trailing quotes, then we need to eval
                # string to get the actual expression
                module_end_str = eval(module_end_str, global_vars, local_vars)
            
            try:
                # str(pd.dataframe) will truncate output by display.max_colwidth
                # set width temporarily to extract more text
                output_str = str(eval(module_end_str, global_vars, local_vars))
                self.df = local_vars['df']      
                if show_plot() == Status.SHOW_PLOT_SUCCESS:
                    logging.info("Plot displayed successfully!")
                    output_str += "\n Plot displayed successfully!" 
                
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
            return err_string
        finally:
            sys.stdout = old_stdout  # Restore standard output
            signal.alarm(0)  # Disable the alarm
