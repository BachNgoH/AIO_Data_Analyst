from llama_index.core.prompts import PromptTemplate, PromptType

DEFAULT_INSTRUCTION_STR = (
    "1. Convert the query to executable Python code.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
    "6. The import of pandas as pd, numpy as np, and seaborn as sns are already made, DO NOT IMPORT AGAIN\n"
    "7. When there are pandas data frame in the result, convert the head to markdown format"
)


# **NOTE**: newer version of sql query engine
DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Instructions (optional):\n{pandas_instructions}\n\n"
    "Execution Output: {pandas_output}\n\n"
    "Response: "
)
DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL,
)

DEFAULT_PANDAS_TMPL = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

DEFAULT_PANDAS_PROMPT = PromptTemplate(
    DEFAULT_PANDAS_TMPL, prompt_type=PromptType.PANDAS
)

