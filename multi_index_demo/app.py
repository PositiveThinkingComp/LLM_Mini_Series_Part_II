import streamlit as st
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from streamlit_utils import local_css, remote_css, load_pdf_files
from config import load_app_config, set_global_api_key
from indexing_utils import ServiceContextLoader, create_multi_index
from query_executers import QueryExecuter


def main():
    """
    This is the main method of the streamlit application 
    """
    # Set the OpenAI Api key as an environment variable 
    set_global_api_key()

    dirname = Path(os.path.dirname(__file__))
    local_css((dirname /"style.css").as_posix())
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

    # Load a Configuration object for the application
    app_config = load_app_config()
    
    # Initialize a ServiceContext for the QueryEngine 
    service_context = ServiceContextLoader(app_config=app_config).load()

    # Initialize a simple SentenceTransformer model for clustering the final responses 
    sbert_model = SentenceTransformer(app_config.ClusteringConfig.SentenceTransformerModel)

    st.title("Parallel Multi-Document Question Answering")

    # Provide a file_uploader with drag and drop functionality 
    multiple_files = st.file_uploader(
        "Drop multiple files:", accept_multiple_files=True
    )

    if multiple_files is None:
        st.text("No upload")
    else:
        files = [file for file in multiple_files if str(file.name).endswith(".pdf")] 

    # Load the pdf files based on the file objects 
    file_content_list = load_pdf_files(files=files)

    if file_content_list:
        top_k = app_config.QueryEngineConfig.similarity_top_k

        # Create a multi-index query engine based on the pdf file content 
        multi_index_query_engine = create_multi_index(file_content_list=file_content_list,
                                                    _service_context=service_context,
                                                    top_k=top_k)
        
        # Execute the query and display the results in the streamlit app
        query_executer = QueryExecuter(query_engine=multi_index_query_engine, 
                                    sbert_model=sbert_model,
                                    config=app_config)
        query_executer.run()

if __name__ == "__main__":
    main()
