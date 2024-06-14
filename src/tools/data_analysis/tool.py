from typing import Optional
import logging
from llama_index.core.llms import LLM 

class DataAnalysisToolSuite:
        
    def __init__(self, llm : LLM) -> None:
        self._llm = llm
        self._pandas_prompt = "Generate pandas code for the query."
        self._response_synthesis_prompt = "Synthesize a response based on the query and the pandas output."
        self._instruction_str = "Generate pandas code for the query."
        self._verbose = True
        self._synthesize_response = True


    def retry_generate_code(self, code: str, exception: Exception):
        correction_input = ErrorCorrectionPipelineInput(code, exception)
        return self.code_exec_error_pipeline.run(correction_input)


    def run_generate_code(self, query_str) -> dict:
        """
        Generate code for a given query and execute the code.

        Args:
            query_str (str): The query string to generate code for.

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
            logging.info(f"> Pandas Instructions:\n" f"```\n{pandas_response_str}\n```\n")
        pandas_output = self._instruction_parser.parse(pandas_response_str)
        if self._verbose:
            logging.info(f"> Execution Output: {pandas_output}\n")

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
