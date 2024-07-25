from llama_index.core.prompts import PromptTemplate, PromptType

DEFAULT_INSTRUCTION_STR = (
    "1. Convert the query to executable Python code.\n"
    "2. Format the code properly with consistent indentation (use 4 spaces for each level).\n"
    "3. Use clear and descriptive variable names.\n"
    "4. Separate logical sections of code with blank lines for readability.\n"
    "5. Include all necessary function definitions and variable assignments.\n"
    "6. Do not quote or escape the code in any way.\n"
    "7. Avoid using input() or any other user input functions.\n"
    "8. Ensure all variables used are properly defined within the code.\n"
    "9. Use appropriate whitespace around operators and after commas.\n"
    "10. Follow PEP 8 style guidelines for naming conventions and code layout.\n"
    "11. The variable `df` is already defined, and shouldn't be redefined in the code.\n"
    "12. For matplotlib plots, use plt.figure() to create figures, but don't call plt.show().\n"
    "13. Avoid generating too long code that make execution time too long.\n"
    "14. Do not include any import statements in the code.\n"
)

    # "8. When working with pandas DataFrames, use .head().to_markdown() for display.\n"

    # "6. The import of pandas as pd, numpy as np, and seaborn as sns are already made, DO NOT IMPORT AGAIN\n"

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


DEFAULT_SCIKIT_TMPL = (
    # "YOU DON'T NEED TO IMPORT ANYTHING"
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "Include df = pd.read_csv('./data/dataframe.csv')"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
    "Encapsulate each step of the machine learning workflow in a function to maintain clean and modular code. For example: \n"
    "def split_data: return X_train, X_test, y_train, y_test.\n"
    "def train_model(model, X_train, y_train): return trained_model.\n\n"
    "Generate a well-structured machine learning code using the Scikit-learn library. "
     "- Save all plots as PNG image files in the './plots' directory.\n"
    "The code should include the following functions:\n\n"
    "Base of the 'df' provide, you choose the suitable target"
    "1. Use the dataframe `df` directly without loading it.\n\n"
    "2. `split_data`: Splits the data into training and testing sets.\n\n"
    "3. `define_model`: Defines a machine learning model and returns it.\n\n"
    "4. `train_model`: Trains the model on the training data.\n\n"
    "5. `evaluate_model`: Evaluates the model using a specified evaluation metric.\n\n"
    "6. `plot_test_results`: Plots the test set results, showing the true labels versus the predicted labels.\n\n"
    "rememeber to print information such as accuracy or something you think it's necessary"
    "Assume the dataset is a Pandas DataFrame `df` with features and labels for training and testing."
    # "MUST NOT  IMPORT ANYTHING"
)
DEFAULT_SCIKIT_PROMPT = PromptTemplate(
    DEFAULT_SCIKIT_TMPL
)
DEFAULT_INSTRUCTION_SCIKIT_STR = (
    "1. Convert the query to executable Python code.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "Make sure to encapsulate each step in a function to maintain clean and modular code. For example: \n"
    "Just give me the python code, nothing include except python code"
    "The name of the dataframe is `df`, use directly df, don't need to load_data()"
    # "6. The import of pandas as pd, numpy as np, and seaborn as sns are already made, DO NOT IMPORT AGAIN\n"
    "7. Avoid code with "
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
    # "MUST NOT  IMPORT ANYTHING"
)
DEFAULT_EDA_TMPL = (
    "You are an expert data analyst working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Generate comprehensive and professional Exploratory Data Analysis (EDA) code using pandas, matplotlib, and seaborn libraries. "
    "Focus on creating informative and visually appealing plots. The code should include the following visualizations:\n\n"
    "1. Distribution Plots:\n"
    "   - Create histograms and kernel density plots for numerical features\n"
    "   - Use seaborn's distplot or displot for enhanced distribution visualization\n\n"
    "2. Feature Relationships:\n"
    "   - Generate a pair plot using seaborn's pairplot function to show relationships between multiple features\n"
    "   - Create a correlation heatmap using seaborn's heatmap function\n\n"
    "3. Categorical Data Visualization:\n"
    "   - Use bar plots and count plots for categorical features\n"
    "   - Create box plots or violin plots to show the distribution of numerical features across categories\n\n"
    "4. Time Series Plots (if applicable):\n"
    "   - Generate line plots for time-based data\n"
    "   - Use seaborn's lineplot for enhanced time series visualization\n\n"
    "5. Feature-Target Relationship:\n"
    "   - Create scatter plots or joint plots to visualize the relationship between features and the target variable\n"
    "   - Use seaborn's regplot for regression plots with confidence intervals\n\n"
    "7. Dimensionality Reduction Visualization (if applicable):\n"
    "   - Implement PCA and visualize the results in a scatter plot\n"
    "   - Use t-SNE for non-linear dimensionality reduction and visualization\n\n"
    "8. Customized Seaborn Plots:\n"
    "   - Utilize seaborn's FacetGrid for creating multiple related plots\n"
    "   - Implement seaborn's jointplot for bivariate distributions\n\n"
    "Use the dataframe `df` directly without loading it.\n"
    "Ensure all plots have appropriate titles, labels, and legends.\n"
    "Use a consistent color scheme and style across all plots.\n"
    "Adjust plot sizes and layouts for optimal visibility.\n"
    "Include code to save high-resolution versions of the plots.\n"
    "Your code should be well-structured, commented, and ready to execute.\n"
    "After generating the code, provide a brief description of each plot and its purpose in the EDA process.\n"
    "Remember to handle any potential errors or edge cases in the data.\n"
    "Do not include any data preprocessing or analysis steps - focus solely on generating visualization code.\n"
    # "The import of pandas as pd, numpy as np, and seaborn as sns are already made, DO NOT IMPORT AGAIN, DO NOT IMPORT ANYTHING"
    "not use : Use of `hue` with `kind='reg'` is not currently supported"
)
DEFAULT_EDA_PROMPT = PromptTemplate(
    DEFAULT_EDA_TMPL
)
DEFAULT_EDA_INSIGHT_TMPL = (
    "You are an AI data analysis expert. The dataframe is named `df`.\n"
    "IMPORTANT:\n"
    "- Do NOT code anything.\n"
    # "- The import of pandas as `pd`, numpy as `np`, and seaborn as `sns` is already done. Do NOT import anything again.\n"
    "- `hue` with `kind='reg'` is not supported.\n\n"
    "Data Preview:\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Execution Results:\n"
    "{eda_output}\n\n"
    "Query:\n"
    "Based on the execution results and the following query, perform a comprehensive analysis:\n"
    "Query: {query_str}\n\n"
    "### Analysis Guidelines:\n"
    "Provide insights using the following steps. FOCUS ON GIVING PERSPECTIVE and PERSONAL OPINION rather than just listing statistics.\n\n"
    "#### 1. Data Distribution:\n"
    "- Discuss the overall distribution of the data.\n"
    "- Note any skewness, outliers, or anomalies.\n\n"
    "#### 2. Significant Patterns and Trends:\n"
    "- Identify and explain any notable patterns or trends in the data.\n\n"
    "#### 3. Noteworthy Testing Results:\n"
    "- Highlight any significant results from statistical tests.\n\n"
    "#### 4. Feature-Label Relationships:\n"
    "- Analyze the relationships between features and the label.\n"
    "- Interpret correlations and discuss their implications.\n\n"
    "#### 5. Feature Influence:\n"
    "- Identify which features have the strongest influence on the target variable.\n\n"
    "#### 6. Patterns, Trends, and Insights:\n"
    "- Describe important patterns, trends, and insights discovered during the analysis.\n"
    "- Comment on the visualizations created and their implications.\n\n"
    "#### 7. Recommendations:\n"
    "- Provide recommendations for further analysis or potential model improvements.\n"
    "- Suggest suitable visualizations to illustrate key findings.\n\n"
    "### Deliverables:\n"
    "Present your analysis concisely, clearly, and with data-driven justification. Provide personal observations and opinions where appropriate."
)
DEFAULT_EDA_INSIGHT_PROMPT = PromptTemplate(
    DEFAULT_EDA_INSIGHT_TMPL
)
DEFAULT_EDA_INSIGHT_CODE_TMPL = (
    "You are an AI data analysis expert. The dataframe is named `df`.\n"
    "IMPORTANT:\n"
    "1. The code should be properly indented and formatted.\n"
    "2. Handle any FutureWarnings, especially for the df.corr() method, by specifying numeric_only=True.\n"
    "3. Do NOT use SHARPIO or any other strange library.\n"
    # "4. The import of pandas as `pd`, numpy as `np`, matplotlib.pyplot as `plt`, and seaborn as `sns` is already done. DO NOT IMPORT ANYTHING AGAIN.\n"
    "5. Use plt.switch_backend('agg') at the beginning of your code to use a non-interactive backend.\n"
    "6. Save all plots as PNG image files in the './plots' directory."
    "7. Do not use plt.show() or any interactive plotting functions.\n"
    "Data Preview:\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}"
    "Generate comprehensive EDA code based on the following query:\n"
    "Query: {query_str}\n\n"
    "Your code should include:\n"
    "1. Descriptive statistics for numeric columns only.\n"
    "2. Correlation analysis for numeric columns only.\n"
    "3. Appropriate statistical tests for numeric columns only.\n"
    "4. Visualizations to illustrate key findings, saved as image files.\n\n"
    "The code must have print() statements to indicate which statistical property is being executed. These print statements must be clear and informative because the execution output will be used for further analysis by another model.\n"
    "Ensure that non-numeric columns are handled appropriately and do not cause errors in numerical operations.\n"
    "Make sure about the indent of the code.\n"
    "Provide the generated EDA code in a single Python code block."
)
DEFAULT_EDA_INSIGHT_CODE_PROMPT = PromptTemplate(
    DEFAULT_EDA_INSIGHT_CODE_TMPL
)


    # "8. Include df = pd.read_csv('./data/dataframe.csv') at the beginning of your code.\n"
