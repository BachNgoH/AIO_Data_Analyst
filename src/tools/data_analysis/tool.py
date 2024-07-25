import logging
import os
import glob
import io
import contextlib
import zipfile
import asyncio
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from llama_index.core.llms import LLM
from llama_index.core.base.response.schema import Response
from llama_index.core.tools import FunctionTool
from src.tools.data_analysis.output_parser import InstructionParser
from src.tools.data_analysis.prompts import (
    DEFAULT_PANDAS_PROMPT,
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT,
    DEFAULT_INSTRUCTION_STR,
    DEFAULT_PANDAS_EXCEPTION_PROMPT,
    DEFAULT_SCIKIT_PROMPT,
    DEFAULT_INSTRUCTION_SCIKIT_STR,
    DEFAULT_EDA_INSIGHT_PROMPT , # Add this line,
    DEFAULT_EDA_INSIGHT_CODE_PROMPT,
)
import chainlit as cl
from chainlit import run_sync
import re
from typing import List

class DataAnalysisToolSuite:

    def __init__(self, df: pd.DataFrame, llm: LLM) -> None:
        self._llm = llm
        self._eda_insight_code_prompt = DEFAULT_EDA_INSIGHT_CODE_PROMPT
        self._pandas_prompt = DEFAULT_PANDAS_PROMPT
        self._pandas_exception_prompt = DEFAULT_PANDAS_EXCEPTION_PROMPT
        self._response_synthesis_prompt = DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        self._scikit_prompt = DEFAULT_SCIKIT_PROMPT
        self._instruction_str = DEFAULT_INSTRUCTION_STR
        self._instruction_str_scikit = DEFAULT_INSTRUCTION_SCIKIT_STR
        self._verbose = True
        self._synthesize_response = False
        self._instruction_parser = InstructionParser(df)
        self._df = df
        self._head = 5
        self._eda_insight_prompt = DEFAULT_EDA_INSIGHT_PROMPT  # Add this line
        self._relative_path = '.files/generated/'
        self._eda_prompt = DEFAULT_EDA_INSIGHT_PROMPT
        self._instruction_str_eda = DEFAULT_INSTRUCTION_STR

        # Ensure the relative directory exists
        os.makedirs(self._relative_path, exist_ok=True)

    def _get_table_context(self) -> str:
        """Get table context."""
        return str(self._df.head(self._head))

    def retry_generate_code(self, prev_code: str, exception_str: str) -> Response:
        """Retry generating code with the given exception string."""
        context = self._get_table_context()
        pandas_response_str = self._llm.predict(
            self._pandas_exception_prompt,
            df_str=context,
            query_str=prev_code,
            instruction_str=self._instruction_str,
            error_msg=exception_str
        )

        if self._verbose:
            logging.info(f"> Instructions:\n{pandas_response_str}\n\n")
            pandas_response_str = "```\n" + pandas_response_str + "\n```\n" if not pandas_response_str.startswith(
                "```") else pandas_response_str
            run_sync(cl.Message(content=f"Generated Instructions:\n\n{pandas_response_str}\n").send())

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
                    query_str=prev_code,
                    pandas_instructions=pandas_response_str,
                    pandas_output=pandas_output,
                )
            )
        else:
            response_str = str(pandas_output)

        return Response(response=response_str, metadata=response_metadata)

    def generate_code(self, query_str: str) -> Response:
        """Generate code for a given query."""
        context = self._get_table_context()
        pandas_response_str = self._llm.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n{pandas_response_str}\n\n")
            pandas_response_str = "```\n" + pandas_response_str + "\n```\n" if not pandas_response_str.startswith(
                "```") else pandas_response_str
            run_sync(cl.Message(content=f"Generated Instructions:\n\n{pandas_response_str}\n").send())

        response_metadata = {
            "instruction_str": pandas_response_str,
        }

        return Response(response=pandas_response_str, metadata=response_metadata)

    def execute_code(self, code_str: str) -> Response:
        """Execute code to analyze the dataframe."""
        pandas_output = self._instruction_parser.parse(code_str)

        if self._verbose:
            logging.info(f"> Execution Output: {pandas_output}\n")
            run_sync(cl.Message(content=f"Execution Output: \n {pandas_output}\n").send())

        response_metadata = {
            "raw_pandas_output": pandas_output,
        }

        return Response(response=pandas_output, metadata=response_metadata)

    def generate_and_run_code(self, query_str: str) -> Response:
        """Generate code for a given query and execute the code to analyze the dataframe."""
        context = self._get_table_context()
        pandas_response_str = self._llm.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str,
        )

        if self._verbose:
            logging.info(f"> Instructions:\n{pandas_response_str}\n\n")
            pandas_response_str = "```python\n" + pandas_response_str + "\n```\n" if not pandas_response_str.startswith(
                "```") else pandas_response_str
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

    async def agenerate_and_run_code(self, query_str: str) -> Response:
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
            pandas_response_str = "```python\n" + pandas_response_str + "\n```\n" if not pandas_response_str.startswith("```") else pandas_response_str
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
                await self._llm.apredict(
                    self._response_synthesis_prompt,
                    query_str=query_str,
                    pandas_instructions=pandas_response_str,
                    pandas_output=pandas_output,
                )
            )
        else:
            response_str = str(response_metadata)

        return Response(response=response_str, metadata=response_metadata)
    def parse_code_markdown(self, text: str, only_last: bool = True) -> str:
        # Regular expression pattern to match code within triple-backticks with the `python` language specifier
        pattern = r"```python\n(.*?)```"

        # Find all matches of the pattern in the text
        matches = re.findall(pattern, text, re.DOTALL)

        # Return the last matched group if requested
        if matches and only_last:
            return matches[-1]
        elif matches:
            return "\n".join(matches)
        else:
            # If no matches found, return an empty string
            return ""
    def generate_and_execute_scikit_code(self, query_str: str) -> Response:
        """Generate and execute Scikit-learn code for a given query."""
        # Clear the plots folder
        self.clear_plots_folder()

        context = self._get_table_context()
        
        # Generate Scikit-learn code
        scikit_response_str = self._llm.predict(
            self._scikit_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str_scikit,
        )
        scikit_code = self.extract_first_python_code(scikit_response_str)
        
        if self._verbose:
            logging.info(f"> Scikit-learn Code Instructions:\n{scikit_code}\n")
            run_sync(cl.Message(content=f"Generated Scikit-learn Instructions:\n```python\n{scikit_code}\n```").send())

        # Parse and execute the generated code
        executable_output = self._instruction_parser.parse(scikit_code)

        # Display plots
        plots_dir = './plots'
        image_files = glob.glob(os.path.join(plots_dir, '*.png'))
        if image_files:
            for image_file in image_files:
                run_sync(cl.Message(content="", elements=[
                    cl.Image(path=image_file, name="plot", display="inline", size="large")
                ]).send())

        if self._verbose:
            logging.info(f"> Execution Output: {executable_output}\n")
            run_sync(cl.Message(content=f"Execution Output: \n{executable_output}\n").send())

        response_metadata = {
            "scikit_instruction_str": scikit_code,
            "raw_scikit_output": executable_output,
        }

        # Prepend import statements and save the code
        import_statements = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.svm import SVC\n"
            "from sklearn.linear_model import LinearRegression, LogisticRegression\n"
            "from sklearn.tree import DecisionTreeClassifier\n"
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
            "from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix\n"
            "df = pd.read_csv('./data/dataframe.csv')\n"
        )
        full_scikit_code = import_statements + scikit_code

        # Save the code to a file after execution
        file_path = self.save_code_to_file(full_scikit_code, 'model.py')

        # Zip the saved file
        zip_path = self.zip_code_file(file_path, 'model.zip')

        return Response(response=str(executable_output), metadata=response_metadata)

    async def _generate_scikit_code_async(self, query_str: str) -> str:
        context = self._get_table_context()
        scikit_response_str = await self._llm.apredict(
            self._scikit_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str_scikit,
        )

        if self._verbose:
            logging.info(f"> Scikit Code Instructions:\n{scikit_response_str}\n")
            scikit_response_str = "```\n" + scikit_response_str + "\n```\n" if not scikit_response_str.startswith(
                "```") else scikit_response_str
            await cl.Message(content=f"Generated Scikit Instructions:\n\n{scikit_response_str}\n").send()

        return scikit_response_str
    
    def _execute_scikit_code(self, code_str: str) -> str:
        """Execute the generated Scikit-learn code."""
        try:
            run_sync(cl.Message(content=f"Generated Code:\n```python\n{code_str}\n```").send())

            output_capture = io.StringIO()

            # Import necessary modules within the execution context
            exec_globals = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,  # Add this line to include seaborn
            "df": self._df
            }
            exec("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
        """, exec_globals)

            with contextlib.redirect_stdout(output_capture):
                exec(code_str, exec_globals)

            execution_output = output_capture.getvalue()
            output_capture.close()

            run_sync(cl.Message(content=f"Execution Output:\n```\n{execution_output}\n```").send())

            if plt.get_fignums():
                run_sync(cl.Message(content="Generated Plot:").send())
                plt.show()
                plt.close('all')

            return execution_output

        except Exception as e:
            error_msg = f"Error Executing Scikit Code: {str(e)}\n"
            logging.error(error_msg)
            run_sync(cl.Message(content=error_msg).send())
            return error_msg

    def save_code_to_file(self, code: str, file_name: str) -> str:
        """Save the generated code to a .py file."""
        file_path = os.path.join(self._relative_path, file_name)
        with open(file_path, 'w') as file:
            file.write(code)
        return file_path

    def zip_code_file(self, file_path: str, zip_name: str) -> str:
        """Zip the .py file."""
        zip_path = os.path.join(self._relative_path, zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(file_path, os.path.basename(file_path))
        return zip_path

    # async def display_files(self, py_file_path: str, zip_file_path: str) -> None:
    #     """Display the generated files for the user to download."""
    #     await cl.FileMessage(
    #         content="Here are the generated files:",
    #         files=[
    #             {"file_path": py_file_path, "display_name": "Generated Scikit Code.py"},
    #             {"file_path": zip_file_path, "display_name": "Generated Scikit Code.zip"},
    #         ]
    #     ).send()
    def extract_first_python_code(self, text: str) -> str:
        """Extract the first Python code block from the text."""
        pattern = r'```python\s*(.*?)\s*```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return ""
    def clear_plots_folder(self) -> None:
        """Clear the plots folder."""
        plots_dir = './plots'
        files = glob.glob(os.path.join(plots_dir, '*'))
        for f in files:
            os.remove(f)
    
    def generate_eda_insights(self, query_str: str) -> Response:
        """Generate, execute EDA code, and provide insights based on the results."""
        context = self._get_table_context()
        self.clear_plots_folder()
        # Generate EDA code
        eda_code = self._llm.predict(
            self._eda_insight_code_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str_eda,
        )
        eda_code = self.extract_first_python_code(eda_code)
        if self._verbose:
            logging.info(f"> EDA Code Instructions:\n{eda_code}\n")
            run_sync(cl.Message(content=f"Generated EDA Instructions:\n```\n{eda_code}\n```").send())

        # Parse and execute the generated code
        eda_output = self._instruction_parser.parse(eda_code)
         # Tìm tất cả các file ảnh PNG trong thư mục './plots'
        plots_dir = './plots'
        image_files = glob.glob(os.path.join(plots_dir, '*.png'))
        
        # Hiển thị các ảnh lên Chainlit, mỗi ảnh trên một dòng riêng
        if image_files:
            for image_file in image_files:
                run_sync(cl.Message(content="", elements=[
                    cl.Image(path=image_file, name="plot", display="inline", size="large")
                ]).send())
        if self._verbose:
            logging.info(f"> Execution Output: {eda_output}\n")
            run_sync(cl.Message(content=f"Execution Output: \n{eda_output}\n").send())

        response_metadata = {
            "eda_instruction_str": eda_code,
            "raw_eda_output": eda_output,
        }

        # Generate insights based on the execution results
        insights = self._llm.predict(
            self._eda_insight_prompt,
            df_str=context,
            query_str=query_str,
            eda_code=eda_code,
            eda_output=eda_output
        )
        if self._verbose:
            logging.info(f"> Eda Output: {insights}\n")
            run_sync(cl.Message(content=f"Eda Output: \n{insights}\n").send())
        response_str = str(insights)

        full_eda_code = eda_code

        # Save the code to a file after execution
        file_path = self.save_code_to_file(full_eda_code, 'eda.py')

        # Zip the saved file
        zip_path = self.zip_code_file(file_path, 'eda.zip')
        return Response(response=response_str, metadata=response_metadata)
    async def generate_eda_insights_async(self, query_str: str) -> Response:
        """Generate, execute EDA code, and provide insights based on the results asynchronously."""
        context = self._get_table_context()

        # Generate EDA code
        eda_code = await self._llm.apredict(
            self._eda_insight_code_prompt,
            df_str=context,
            query_str=query_str,
            instruction_str=self._instruction_str_eda,
        )
        eda_code = self.extract_first_python_code(eda_code)
        if self._verbose:
            logging.info(f"> EDA Code Instructions:\n{eda_code}\n")
            await cl.Message(content=f"Generated EDA Instructions:\n```python\n{eda_code}\n```").send()

        # Parse and execute the generated code
        eda_output = await cl.make_async(self._instruction_parser.parse)(eda_code)

        # Display plots
        plots_dir = './plots'
        image_files = glob.glob(os.path.join(plots_dir, '*.png'))
        if image_files:
            for image_file in image_files:
                await cl.Message(content="", elements=[
                    cl.Image(path=image_file, name="plot", display="inline", size="large")
                ]).send()

        if self._verbose:
            logging.info(f"> Execution Output: {eda_output}\n")
            await cl.Message(content=f"Execution Output: \n{eda_output}\n").send()

        response_metadata = {
            "eda_instruction_str": eda_code,
            "raw_eda_output": eda_output,
        }

        # Generate insights based on the execution results
        insights = await self._llm.apredict(
            self._eda_insight_prompt,
            df_str=context,
            query_str=query_str,
            eda_code=eda_code,
            eda_output=eda_output
        )
        if self._verbose:
            logging.info(f"> Eda Output: {insights}\n")
            await cl.Message(content=f"Eda Output: \n{insights}\n").send()
        response_str = str(insights)

        # Prepend import statements and save the code
        import_statements = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
        )
        full_eda_code = import_statements + eda_code
        file_path = await cl.make_async(self.save_code_to_file)(full_eda_code, 'eda.py')
        zip_path = await cl.make_async(self.zip_code_file)(file_path, 'eda.zip')

        return Response(response=response_str, metadata=response_metadata)
    
    def full_analysis(self, query_str: str) -> Response:
        """Perform full analysis including EDA insights and Scikit-learn model execution."""
        
        # Step 1: Generate EDA insights
        eda_response = self.generate_eda_insights(query_str)
        
        # Step 2: Generate and execute Scikit-learn code
        scikit_response = self.generate_and_execute_scikit_code(query_str)
        
        # Combine the responses
        full_response = f"## EDA Insights\n\n{eda_response.response}\n\n"
        full_response += f"## Scikit-learn Model Analysis\n\n{scikit_response.response}"
        
        # Combine metadata
        combined_metadata = {
            "eda_metadata": eda_response.metadata,
            "scikit_metadata": scikit_response.metadata
        }
        
        # Display plots from EDA
        plots_dir = './plots'
        image_files = glob.glob(os.path.join(plots_dir, '*.png'))
        if image_files:
            for image_file in image_files:
                cl.Message(content="", elements=[
                    cl.Image(path=image_file, name="plot", display="inline", size="large")
                ]).send()
        
        # Send the formatted response
        cl.Message(content=full_response).send()
        
        return Response(response=full_response, metadata=combined_metadata)

    def get_tools(self):
        """Get tools."""
        return [
            FunctionTool.from_defaults(fn=self.generate_and_run_code, async_fn=self.agenerate_and_run_code),
            # FunctionTool.from_defaults(fn=self.agenerate_and_run_code),
            # FunctionTool.from_defaults(fn=self.execute_code),
            FunctionTool.from_defaults(fn=self.generate_code),
            FunctionTool.from_defaults(fn=self.retry_generate_code),
            # FunctionTool.from_defaults(fn=self.generate_and_execute_scikit_code),
            # FunctionTool.from_defaults(fn=self.generate_eda_insights,description="generate insight, generate eda"),
            # FunctionTool.from_defaults(fn=self.generate_eda_insights_async, description="generate insight, generate eda (async)"),
            # FunctionTool.from_defaults(fn=self.full_analysis, description="perform full analysis including EDA and machine learning modeling"),
        ]