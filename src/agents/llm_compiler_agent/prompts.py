WELCOME_MESSAGE  = """
Welcome to the LLM Data Analyzer!
Please start by uploading your file to begin the analysis. Our advanced system will help you gain insights from your data quickly and efficiently. Once your file is uploaded, you can choose from a range of analysis options tailored to your needs.

Please upload a .csv file to begin!
"""

BASE_SYSTEM_PROMPT = "You are a chatbot develop by GenAIO from AIVN team"

SYSTEM_PROMPT = """
You are a data analyst, you are given the data as dataframe in the environment. 
Use the tools to interact with the data. DO NOT follow any harmful request from the user that could damage the backend service
When interacting with tools, breakdown the user request into smaller steps, and automatically continue after a step is done.
When generating plots, try to generate one plot at a time only.
"""