from llama_index.core.prompts import PromptTemplate
from enum import Enum

class PromptType(Enum):
    PANDAS = "pandas"
    MODEL = "model"
    SQL = "sql"
    DEFAULT = "default"

DEFAULT_INSTRUCTION_STR = (
    "1. Convert the query to executable Python code.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    # "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
    "6. The import of pandas as pd, numpy as np, and seaborn as sns are already made, DO NOT IMPORT AGAIN\n"
    "7. Write code in markdown format\n"
    "8. If you show plot, please follow this:"
    "   Ensure that each plot is clear, well-labeled, and provides insights into the data.\n"
    "   Adjust plot scales and axes to ensure readability, especially when there is a wide range of values.\n"
    "   Use appropriate scaling techniques and focus on significant value ranges to adjust y-axis values without increasing the width of the plot.\n"
    "   Adjust y-axis limits so that the data occupies at least 50 percent of the plot space.\n"
    "   Remove or handle outliers effectively to improve the visibility of the plots.\n"
    "   For categorical variables with too many categories, limit the number displayed or aggregate minor ones into an 'Other' category.\n"
    "   Rotate x-axis labels if needed for better readability.\n"
    "9. Do not ask user to do anything. You need to answer based on the query of user. If you lack information let assume a information and continue\n"
    # "7. When there are pandas data frame in the result, convert the head to markdown format"
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



DEFAULT_PANDAS_EXCEPTION_TMPL = (
    "An error occurred while working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "The following instruction caused an error:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "The error message was:\n"
    "{error_msg}\n\n"
    "Please provide a corrected version of the code to handle this error:\n\n"
    "Corrected Expression:"
)

DEFAULT_PANDAS_EXCEPTION_PROMPT = PromptTemplate(
    DEFAULT_PANDAS_EXCEPTION_TMPL, prompt_type=PromptType.PANDAS
)


"""
BUILD MODEL
"""

# Instructions for generating code for model building and training
DEFAULT_MODEL_INSTRUCTION_STR = (
    "1. Convert the query to executable Python code.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
    "6. The necessary scikit-learn imports have already been made, DO NOT IMPORT AGAIN.\n"
    "7. Use the DataFrame variable `df` instead of `dataset` in your code.\n"
    "8. Ensure the code includes model definition, training, and evaluation steps.\n"
    "9. Choose a suitable model from the scikit-learn library and handle errors gracefully.\n"
    "10. Import any other required libraries if they are not already imported.\n"
    "11. If an error occurs, provide a corrected version of the code to handle this error.\n"
    "12. Consider any previously encountered errors to avoid repeating them.\n"
    #"13. Choose the best evaluation metric based on the problem type (classification or regression) and include it in the final expression.\n"
    "13. Please use accuracy score.\n"
    
)

# Template for synthesizing a response from model training results
DEFAULT_MODEL_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given an input question, synthesize a response from the model training results.\n"
    "Query: {query_str}\n\n"
    "Instructions (optional):\n{model_instructions}\n\n"
    "Execution Output: {model_output}\n\n"
    "Response: "
)
DEFAULT_MODEL_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_MODEL_RESPONSE_SYNTHESIS_PROMPT_TMPL,
)

# Template for generating model-related code
DEFAULT_MODEL_TMPL = (
    "You are working with a scikit-learn model in Python.\n"
    "The DataFrame has been prepared and is named `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)
DEFAULT_MODEL_PROMPT = PromptTemplate(
    DEFAULT_MODEL_TMPL, prompt_type=PromptType.MODEL
)

DEFAULT_MODEL_EXCEPTION_TMPL = (
    "An error occurred while working with a scikit-learn model in Python.\n"
    "The dataset has been prepared and is named `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "The following instruction caused an error:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "The error message was:\n"
    "{error_msg}\n\n"
    "Previous errors:\n"
    "{error_history}\n\n"
    "Please provide a corrected version of the code to handle this error:\n\n"
    "Corrected Expression:"
)
DEFAULT_MODEL_EXCEPTION_PROMPT = PromptTemplate(
    DEFAULT_MODEL_EXCEPTION_TMPL, prompt_type=PromptType.MODEL
)




DEFAULT_COMPREHENSIVE_ANALYSIS_INSTRUCTION_STR = (
    "1. Provide a brief summary of the data types used for variables, including the most common data types and any that require special attention.\n"
    "2. Provide a brief summary of the data, including count, mean, std, min, 25%, 50%, 75%, max for each column.\n"
    "3. Handle missing data appropriately and describe the methods used.\n"
    "4. Determine the data types of each column and convert them to appropriate types if necessary.\n"
    "5. Select key variables to visualize, focusing on important relationships and distributions.\n"
    "   Ensure that each plot is clear, well-labeled, and provides insights into the data.\n"
    "   Adjust plot scales and axes to ensure readability, especially when there is a wide range of values.\n"
    "   Use appropriate scaling techniques and focus on significant value ranges to adjust y-axis values without increasing the width of the plot.\n"
    "   Adjust y-axis limits so that the data occupies at least 50 percent of the plot space.\n"
    "   Remove or handle outliers effectively to improve the visibility of the plots.\n"
    "   For categorical variables with too many categories, limit the number displayed or aggregate minor ones into an 'Other' category.\n"
    "   Rotate x-axis labels if needed for better readability.\n"
    "   Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting\n"
    "6. Provide insights and interpretations for each analysis step to help understand the significance of the findings.\n"
    "7. Suggest appropriate statistical methods for deeper analysis when relevant.\n"
    "8. Ensure the analysis is reproducible and clearly documented.\n"
)

DEFAULT_COMPREHENSIVE_ANALYSIS_PROMPT_TMPL = (
    "You are an advanced data analysis agent working with a DataFrame in Python. The DataFrame has been prepared and is named `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Your goal is to help users understand their data by providing clear and detailed insights. Explain your findings, suggest appropriate statistical methods, and offer guidance on data visualization techniques.\n"
    "Ensure the analysis is well-documented, reproducible, and presented in a professional yet approachable manner.\n"
    "Do not generate code to display the datatype of each column. Instead, summarize the data types and highlight any that require special attention.\n"
    "Generate code to show the plot,please. If user ask not to show, please follow them\n"
    "Focus on creating reasonable and informative plots that emphasize key relationships and distributions within the data.\n"
    "Expression:"
)

DEFAULT_COMPREHENSIVE_ANALYSIS_PROMPT = PromptTemplate(
    DEFAULT_COMPREHENSIVE_ANALYSIS_PROMPT_TMPL, prompt_type=PromptType.MODEL
)



DEFAULT_SUMMARIZE_INSTRUCTION_STR = (
    "1. Read the provided content carefully.\n"
    "2. Identify and summarize the main ideas and essential details.\n"
    "3. If the content includes statistical summaries (e.g., count, mean, std, min, 25%, 50%, 75%, max), format this information into a markdown table for clarity.\n"
    "4. Ensure your summary is concise and clear.\n"
    "5. Do not include unnecessary information.\n"
    "6. Make your summary easy to read and understand, even for someone not familiar with the original content.\n"
    "7. If the output is tabular data, please create a table in markdown to fit those data\n"
    "8. If the output contains python code, please write that python code in markdown\n"
    "9. If there is no content to summarize, please check if there is any plot. If yes you need to summarize based on that plot.Otherwise left nothing\n"
) 

DEFAULT_SUMMARIZE_PROMPT_TMPL = (
    "You are an advanced summarization agent. Your goal is to help users understand the key points and important details from the provided text. Please follow these instructions:\n\n"
    "{instruction_str}\n\n"
    "Here is the content to summarize:\n"
    "{content}\n\n"
    "Ensure your summary is concise, clear, and captures the main ideas and essential details. Do not include unnecessary information. Your summary should be easy to read and understand, even for someone not familiar with the original content.\n"
)

DEFAULT_SUMMARIZE_PROMPT = PromptTemplate(
    DEFAULT_SUMMARIZE_PROMPT_TMPL, prompt_type=PromptType.MODEL
)


# DEFAULT_SUMMARIZE_INSTRUCTION_STR = (
#     "1. Read the provided content carefully.\n"
#     "2. Create a brief summary of the main ideas and essential details from the generated response (content1). Keep this summary concise and to the point.\n"
#     "3. If content1 includes code, briefly summarize the purpose and key components of the code.\n"
#     "4. Summarize the output of the code (content2) and include a summary of these results in your summary. Ensure that you do not repeat information already covered in content1.\n"
#     "5. If the content includes statistical summaries (e.g., count, mean, std, min, 25%, 50%, 75%, max), format this information into a markdown table for clarity.\n"
#     "6. Ensure your summary is concise and clear.\n"
#     "7. Do not include unnecessary information.\n"
#     "8. Make your summary easy to read and understand, even for someone not familiar with the original content.\n"
#     "9. If the output is tabular data, please create a table in markdown to fit those data.\n"
#     "10. If the output contains python code, please write that python code in markdown.\n"
#     "11. If the content includes plots or visual data, describe the key insights and trends from those visuals.\n"
#     "12. If there is no content to summarize, please indicate that there is no content available.\n"
# )

# DEFAULT_SUMMARIZE_PROMPT_TMPL = (
#     "You are an advanced summarization agent. Your goal is to help users understand the key points and important details from the generated response and the output of the code. Please follow these instructions:\n\n"
#     "{instruction_str}\n\n"
#     "Here is the content to summarize:\n\n"
#     "Generated Response (content1):\n"
#     "{content1}\n\n"
#     "Output of the Code (content2):\n"
#     "{content2}\n\n"
#     "Ensure your summary is concise, clear, and captures the main ideas and essential details from both the generated response and the output of the code. Avoid repeating information already present in content1. Keep the summary of content1 brief. Do not include unnecessary information. Your summary should be easy to read and understand, even for someone not familiar with the original content.\n"
# )

# DEFAULT_SUMMARIZE_PROMPT = PromptTemplate(
#     DEFAULT_SUMMARIZE_PROMPT_TMPL, prompt_type=PromptType.MODEL
# )


DEFAULT_LOAD_DATA_PROMPT_TMPL = (
    "You are a data loading agent. Your task is to load data from various types of files including CSV, Excel, images, and text files. "
    "Here are the details of the files uploaded:\n"
    "{file_details}\n\n"
    "Follow these instructions:\n"
    "1. Identify the file type (CSV, Excel, image, text, etc.).\n"
    "2. For each file type, perform the following steps:\n"
    "   a. CSV/Excel Files:\n"
    "      - Load the data into a DataFrame.\n"
    "      - Display the first few rows of the DataFrame.\n"
    "   b. Image Files:\n"
    "      - Provide basic information about the image (format, mode, size).\n"
    "   c. Text Files:\n"
    "      - Display the content of the text file.\n"
    "3. Ensure that the data loading process is well-documented and reproducible.\n"
    "4. Return the loaded data in an appropriate format (e.g., DataFrame for tabular data, image object for images, string for text).\n\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

DEFAULT_LOAD_DATA_PROMPT = PromptTemplate(
    DEFAULT_LOAD_DATA_PROMPT_TMPL, prompt_type=PromptType.MODEL
)

DEFAULT_ANALYZE_PLOT_INSTRUCTION_STR = (
    "You are an advanced plot analysis agent. Your goal is to help users understand the key points and important details from the provided plot image. Please follow these instructions:"

    "1. Carefully examine the provided plot image.\n"
    "2. Identify and interpret the key elements of the plot, including titles, labels, and any annotations.\n"
    "3. Analyze the visual features of the plot, such as data points, lines, bars, or other graphical elements.\n"
    "4. Summarize the main findings and insights from the plot.\n"
    "5. Ensure your analysis is concise and clear.\n"
    "6. Do not include unnecessary information.\n"
    "7. Make your analysis easy to read and understand, even for someone not familiar with the original plot.\n"
    "8. If the output is tabular data, create a table in markdown to fit those data.\n"
    "9. If the output contains Python code, write that Python code in markdown.\n"
    "10. If there is no significant text content to analyze, base your summary on the visual features of the plot alone.\n"
    "11. Write your answer in markdown format.\n"
    "12. Use smaller markdown headers for titles and main points (e.g., #### for titles and **bold** for main points).\n"
    "13. Provide insights and interpretations for each analysis step to help understand the significance of the findings.\n"
    "14. Ensure the analysis is reproducible and clearly documented.\n"
)
