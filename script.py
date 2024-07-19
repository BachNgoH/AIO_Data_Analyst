import io
import sys
import ast
import signal
import traceback
import logging
import re
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import chainlit as cl

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


class Status(Enum):
    NO_PLOT = "No plot"
    SHOW_PLOT_SUCCESS = "Show plot successfully!"
    SHOW_PLOT_FAILED = "Show plot failed!"


def timeout_handler(signum, frame):
    raise TimeoutException("Timed out!")


def parse_code_markdown(text: str, only_last: bool = True):
    pattern = r"```[a-zA-Z]*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    code = matches[-1] if matches and only_last else matches
    if not code:
        candidate = text.strip()
        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1]
        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1]
        if candidate.startswith("`") and candidate.endswith("`"):
            candidate = candidate[1:-1]
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate)
        if candidate.endswith("```"):
            candidate = candidate[:-3]
        code = candidate.strip()
    return code


async def show_plot():
    figures = [plt.figure(n) for n in plt.get_fignums()]
    if not figures:
        return Status.NO_PLOT

    try:
        for i, fig in enumerate(figures):
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            await cl.Image(name=f"plot_{i + 1}", content=buf.getvalue()).send()
        plt.close('all')
        return Status.SHOW_PLOT_SUCCESS
    except Exception as e:
        logger.error(f"Error displaying plot: {e}")
        return Status.SHOW_PLOT_FAILED


async def execute_code(code: str, df: pd.DataFrame, timeout: int = 30):
    local_vars = {
        "df": df,
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
    global_vars = local_vars.copy()

    old_stdout = sys.stdout
    sys.stdout = new_stdout = io.StringIO()

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        tree = ast.parse(code)
        module = ast.Module(tree.body[:-1], type_ignores=[])
        exec(ast.unparse(module), global_vars, local_vars)
        module_end = ast.Module(tree.body[-1:], type_ignores=[])
        module_end_str = ast.unparse(module_end)

        if module_end_str.strip("'\"") != module_end_str:
            module_end_str = eval(module_end_str, global_vars, local_vars)

        output_str = str(eval(module_end_str, global_vars, local_vars))
        df = local_vars['df']
        logging.info("Code executed successfully.")

        plot_status = await show_plot()
        if plot_status == Status.SHOW_PLOT_SUCCESS:
            output_str += "\nPlot displayed successfully!"
        elif plot_status == Status.SHOW_PLOT_FAILED:
            output_str += "\nFailed to display plot."

        printed_output = new_stdout.getvalue()
        if printed_output:
            output_str = printed_output + "\n" + output_str

        return output_str, df
    except TimeoutException:
        return "The execution timed out. Please try again with optimized code or increase the timeout limit.", df
    except Exception as e:
        err_string = f"There was an error running the code. Error message: {e}"
        traceback.print_exc()
        return err_string, df
    finally:
        sys.stdout = old_stdout
        signal.alarm(0)


@cl.on_message
async def main(message: cl.Message):
    # Giả sử df là DataFrame của bạn
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Giả sử đây là code Matplotlib được tạo động bởi mô hình
    code = """
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(df['A'], df['B'], 'o-')
plt.title('A vs B')

plt.subplot(122)
sns.histplot(df['A'], kde=True)
plt.title('Distribution of A')

print(df.describe())
df['C'] = df['A'] + df['B']
df
"""

    parsed_code = parse_code_markdown(code)
    result, updated_df = await execute_code(parsed_code, df)

    await cl.Message(f"Execution result:\n{result}").send()

    if not result.startswith("There was an error") and not result.startswith("The execution timed out"):
        await cl.Message("DataFrame after execution:").send()
        await cl.Message(f"```\n{updated_df}\n```").send()