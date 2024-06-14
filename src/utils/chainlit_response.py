from typing import Any
import chainlit as cl
from chainlit.sync import run_sync
import pandasai.pandas as pd
from pandasai.responses.response_parser import ResponseParser
from pandasai.skills import skill


class ChainlitResponse(ResponseParser):
    def __init__(self, context):
        super().__init__(context)

    def format_plot(self, result) -> None:
        """
        Display plot against a user query in Streamlit
        Args:
            result (dict): result contains type and value
        """
        image_path = result["value"]
        image = cl.Image(path=image_path, name="plot", display="inline")
        run_sync(cl.Message(
            content="Here is the generated plot:", 
            elements=[image]
        ).send())
        
        # return image_path
    
    
    def format_dataframe(self, result: dict) -> pd.DataFrame:
        """
        Format dataframe generate against a user query
        Args:
            result (dict): result contains type and value
        Returns:
            Any: Returns depending on the user input
        """
        return result["value"].to_markdown()

    def format_other(self, result) -> Any:
        """
        Format other results
        Args:
            result (dict): result contains type and value
        Returns:
            Any: Returns depending on the user input
        """
        return result["value"]


