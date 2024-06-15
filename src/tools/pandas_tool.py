import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.groq import Groq
from src.utils.llm_utils import load_model
from src.tools.data_analysis.tool import DataAnalysisToolSuite

def load_pandas_tool(df_path="./data/train.csv"):
    llm = load_model()
    data_df = pd.read_csv(df_path)
    # query_engine = PandasQueryEngine(df=data_df, verbose=True, llm=llm)
    
    # return QueryEngineTool(
    #     query_engine=query_engine,
    #     metadata=ToolMetadata(
    #         name="pandas_engine",
    #         description=(
    #             "Provides pandas code and execution on the data frame based on the query. "
    #         )))
    
    tool_suite = DataAnalysisToolSuite(df=data_df, llm=llm)
    return tool_suite.get_tools()