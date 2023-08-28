import streamlit as st
import fitz
from pathlib import Path
import re
from typing import List, Dict

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

@st.cache_data
def load_pdf_files(files: List) -> List[Dict]:
    """
    This method loads and caches the content of PDF Files based on a list of file objects 

    :param files: File objects 

    :return: List of pdf content
    """
    content_list = [] 
    if len(files) > 0:               
        for ind, file in enumerate(files):
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                page_texts = []
                for page in doc:
                    page_text = page.get_text()
                    page_texts.append(page_text)
                
                page_text = "\n".join(page_texts)

            cv_id = ind + 1
            title = re.sub("\.pdf", "", file.name)            
            title = f"{title} Data Scientist {cv_id}"
            engine_name = re.sub(" ", "_", title)
            tmp_engine_name = f"{engine_name}_query_engine"
            tmp_index_name = f"{engine_name}_index"  

            content_list.append({
                "engine_name": tmp_engine_name,
                "index_name": tmp_index_name,
                "text": page_text,
                "title": title
            })
                        
    return content_list
