import yaml
from pydantic import BaseModel
from typing import Optional
import os
from pathlib import Path
import openai
from dotenv import load_dotenv

class LLMConfig(BaseModel):
    """
    Configuration of the LLM prompt parameters 
    """
    temperature: float
    model: str
    max_tokens: int

class SimpleNodeParser(BaseModel):
    """
    Configuration of the simple Document to Node parser
    """
    chunk_size: int
    chunk_overlap: int 

class PromptHelper(BaseModel):
    """
    This is a Prompt Helper config which configures actual constraints of the prompt context size
    and overlap between chunks 
    """
    context_window: int
    chunk_overlap_ratio: float
    chunk_size_limit: Optional[int]

class QueryEngineConfig(BaseModel):
    """
    Configuration of the query engine
    """
    similarity_top_k: int

class ClusteringConfig(BaseModel):
    """
    Configuration of the SentenceTransformer Embedding model for cluster analysis
    """
    SentenceTransformerModel: str

class EscoSkillApiConfig(BaseModel):
    index_name: str
    top_k: int

class SkillsToListConfig(BaseModel):
    llm_model_name: str
    temperature: float
    prompt_template: str

class AppConfig(BaseModel):
    """
    This is a basic config object for the Streamlit Application
    """
    LLMConfig: LLMConfig
    PromptHelper: PromptHelper
    SimpleNodeParser: SimpleNodeParser
    QueryEngineConfig: QueryEngineConfig
    ClusteringConfig: ClusteringConfig
    EscoSkillApiConfig: EscoSkillApiConfig
    SkillsToListConfig: SkillsToListConfig

def load_app_config() -> AppConfig:
    """
    This method loads the AppConfig object

    :return: Initialized AppConfig object
    """    
    dirname = Path(os.path.dirname(__file__))
    with (dirname / "app_config.yaml").open("r", encoding="utf-8") as f:    
        app_config_dict = yaml.safe_load(f)

    app_config_obj = AppConfig(**app_config_dict)

    return app_config_obj

def set_global_api_key():
    """
    This method sets the API key globally
    """
    load_dotenv()

    openai.api_key = os.environ["OPENAI_API_KEY"]
