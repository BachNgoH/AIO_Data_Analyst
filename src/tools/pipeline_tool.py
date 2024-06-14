from typing import Optional

from pandasai.helpers.query_exec_tracker import QueryExecTracker
from pandasai.pipelines.chat.chat_pipeline_input import (
    ChatPipelineInput,
)
from pandasai.pipelines.chat.code_execution_pipeline_input import (
    CodeExecutionPipelineInput,
)
from pandasai.pipelines.chat.error_correction_pipeline.error_correction_pipeline import (
    ErrorCorrectionPipeline,
)
from pandasai.pipelines.chat.error_correction_pipeline.error_correction_pipeline_input import (
    ErrorCorrectionPipelineInput,
)
from pandasai.pipelines.chat.validate_pipeline_input import (
    ValidatePipelineInput,
)


def on_code_execution_failure(self, code: str, errors: Exception) -> str:
    """
    Executes on code execution failure
    Args:
        code (str): code that is ran
        exception (Exception): exception that is raised during code execution

    Returns:
        str: returns the updated code with the fixes
    """
    # Add information about the code failure in the query tracker for debug
    self.query_exec_tracker.add_step(
        {
            "type": "CodeExecution",
            "success": False,
            "message": "Failed to execute code",
            "execution_time": None,
            "data": {
                "content_type": "code",
                "value": code,
                "exception": errors,
            },
        }
    )


def on_code_retry(self, code: str, exception: Exception):
    correction_input = ErrorCorrectionPipelineInput(code, exception)
    return self.code_exec_error_pipeline.run(correction_input)


def run_generate_code(self, input: ChatPipelineInput) -> dict:
    """
    Executes the code generation pipeline with user input and return the result
    Args:
        input (ChatPipelineInput): _description_

    Returns:
        The `output` dictionary is expected to have the following keys:
        - 'type': The type of the output.
        - 'value': The value of the output.
    """
    self._logger.log(f"Executing Pipeline: {self.__class__.__name__}")

    # Reset intermediate values
    self.context.reset_intermediate_values()

    # Start New Tracking for Query
    self.query_exec_tracker.start_new_track(input)

    self.query_exec_tracker.add_skills(self.context)

    self.query_exec_tracker.add_dataframes(self.context.dfs)

    # Add Query to memory
    self.context.memory.add(input.query, True)

    self.context.add_many(
        {
            "output_type": input.output_type,
            "last_prompt_id": input.prompt_id,
        }
    )
    try:
        output = self.code_generation_pipeline.run(input)

        self.query_exec_tracker.success = True

        self.query_exec_tracker.publish()

        return output

    except Exception as e:
        # Show the full traceback
        import traceback

        traceback.print_exc()

        self.last_error = str(e)
        self.query_exec_tracker.success = False
        self.query_exec_tracker.publish()

        return (
            "Unfortunately, I was not able to answer your question, "
            "because of the following error:\n"
            f"\n{e}\n"
        )

def run_execute_code(self, input: CodeExecutionPipelineInput) -> dict:
    """
    Executes the chat pipeline with user input and return the result
    Args:
        input (CodeExecutionPipelineInput): _description_

    Returns:
        The `output` dictionary is expected to have the following keys:
        - 'type': The type of the output.
        - 'value': The value of the output.
    """
    self._logger.log(f"Executing Pipeline: {self.__class__.__name__}")

    # Reset intermediate values
    self.context.reset_intermediate_values()

    # Start New Tracking for Query
    self.query_exec_tracker.start_new_track(input)

    self.query_exec_tracker.add_skills(self.context)

    self.query_exec_tracker.add_dataframes(self.context.dfs)

    # Add Query to memory
    self.context.memory.add(input.code, True)

    self.context.add_many(
        {
            "output_type": input.output_type,
            "last_prompt_id": input.prompt_id,
        }
    )
    try:
        output = self.code_execution_pipeline.run(input.code)

        self.query_exec_tracker.success = True

        self.query_exec_tracker.publish()

        return output

    except Exception as e:
        # Show the full traceback
        import traceback

        traceback.print_exc()

        self.last_error = str(e)
        self.query_exec_tracker.success = False
        self.query_exec_tracker.publish()

        return (
            "Unfortunately, I was not able to answer your question, "
            "because of the following error:\n"
            f"\n{e}\n"
        )