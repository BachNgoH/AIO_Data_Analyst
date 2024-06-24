import os
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from src.const import LLM_PROVIDER, MODEL_ID, TEMPERATURE
from dotenv import load_dotenv
import logging

load_dotenv(override=True)

def load_model():
    """
    Select a model for text generation using multiple services.
    Args:
        LLM_PROVIDER (str): Service name indicating the type of model to load.
        MODEL_ID (str): Identifier of the model to load from HuggingFace's model hub.
    Returns:
        LLM: llama-index LLM for text generation
    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {MODEL_ID}")
    logging.info("This action can take a few minutes!")

    if LLM_PROVIDER == "ollama":
        logging.info(f"Loading Ollama Model: {MODEL_ID}")
        return Ollama(model=MODEL_ID, temperature=TEMPERATURE)
    elif LLM_PROVIDER == "openai":
        logging.info(f"Loading OpenAI Model: {MODEL_ID}")
        return OpenAI(model=MODEL_ID, temperature=TEMPERATURE, api_key=os.getenv("OPENAI_API_KEY"))
    elif LLM_PROVIDER == "groq":
        logging.info(f"Loading Groq Model: {MODEL_ID}")    
        return Groq(model=MODEL_ID, temperature=TEMPERATURE, api_key=os.getenv("GROQ_API_KEY"))
    elif LLM_PROVIDER == "gemini":
        logging.info(f"Loading Gemini Model: {MODEL_ID}")
        return Gemini(model=MODEL_ID, temperature=TEMPERATURE, api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        raise NotImplementedError("The implementation for other types of LLMs are not ready yet!")