import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.groq import Groq

def load_pandas_tool():
    llm = Groq(model="llama3-70b-8192")
    data_df = pd.read_csv("../data/train.csv")
    query_engine = PandasQueryEngine(df=data_df, verbose=True, llm=llm)
    
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="pandas_engine",
            description=(
                "Provides pandas code and execution on the data frame based on the query. "
            )))