from llama_index.query_engine import SubQuestionQueryEngine
import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer

from streamlit_utils import icon
from esco_skill_graph.esco_skill_extractor import SkillsToList
from esco_skill_graph.esco_skill_mapping import EscoSkillApi
from esco_skill_graph.esco_skill_graph import SkillGraph
from config import AppConfig
from response_clustering import ResponseClustering
from llama_index.response.schema import Response


class QueryExecuter:
    """
    This class executes queries against the multi-index and summarizes the results
    """
    def __init__(self, query_engine: SubQuestionQueryEngine, 
                sbert_model: SentenceTransformer,
                config: AppConfig):
        self.query_engine = query_engine
        self.sbert_model = sbert_model
        self.config = config

    def run(self):
        """
        This QueryExecuter runs a query based on the provided input text and 
        visualizes the query results 
        """
        query_text = st.text_input("", "Search...")
        icon("search")
        button_clicked = st.button("OK")

        if button_clicked:
            response = self.query_engine.query(str(query_text))
            st.title("Raw Search results: ")
            st.write(f"**Query: {query_text}**")

            response_df = self.response_nodes_2_df(response=response)
            
            st.markdown("""**Raw Response**""")
            st.write(f"""{response.response}""")
            st.write(response_df)

            resp_clustering = ResponseClustering(sbert_model=self.sbert_model)
            resp_clustering.compute_response_clusters(response_df=response_df)

            if "skill" in query_text.lower():
                st.title("Network analysis of skills")
                self.visualize_skill_graph(response_df=response_df)

    def response_nodes_2_df(self, response: Response) -> pd.DataFrame:
        """
        This method returns the response node content as a DataFrame 

        :param: Response object to be formatted as a pandas DataFrame

        :return: Response object as a pandas DataFrame 
        """
        data_list = []
        for ind, node in enumerate(response.__dict__["source_nodes"]):
            split_node_text = node.node.text.split("\nResponse: \n")
            subquery = split_node_text[0]
            sub_response = split_node_text[-1]
            data_list.append({"id": f"Data Scientist {ind + 1}", "response": sub_response, "subquery": subquery, })

        df = pd.DataFrame(data_list)

        return df
    
    def visualize_skill_graph(self, response_df: pd.DataFrame):
        """
        This method visualizes the SKill Graph 

        :param response_df: DataFrame containing the Query Responses per PDF index 
        """
        skills_2_list = SkillsToList(model_name=self.config.SkillsToListConfig.llm_model_name, 
                                    temperature=self.config.SkillsToListConfig.temperature, 
                                    prompt_template_path=self.config.SkillsToListConfig.prompt_template)
        
        skill_lists = skills_2_list(skill_descriptions=response_df.response.tolist())

        esco_api = EscoSkillApi(sbert_model=self.sbert_model, 
                        index_name=self.config.EscoSkillApiConfig.index_name,
                        top_k=self.config.EscoSkillApiConfig.top_k)

        skill_lists_2_graph = []
        for ind, skill_list in enumerate(skill_lists):
            normalized_skills = esco_api.run_queries(queries=skill_list)
            skill_lists_2_graph.append({
            "id": f"Data Scientist {ind + 1}",
            "skills": normalized_skills
            })

        skill_graph = SkillGraph(graph_type="spring")
        skill_graph.plot_skill_graph(skill_lists=skill_lists_2_graph)
