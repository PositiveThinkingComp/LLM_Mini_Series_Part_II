import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Union
import logging
import os
from paths import REPO_DIR_PATH

logging.basicConfig(level=logging.DEBUG)

class EscoSkillApi:
    """
    The EscoSkillApi object 
    """
    def __init__(self,  
                sbert_model: SentenceTransformer, 
                index_name: str,
                top_k: int):   
        
        self.esco_skills = self.load_esco_dataset(filepath=os.environ["ESCO_NER_SEARCHTERMS"])
        self.sbert_model = sbert_model
        self.index_name = index_name
        try:
            self.index = faiss.read_index((REPO_DIR_PATH / "esco_skill_graph"/ self.index_name).as_posix())
        except: 
            self.create_index()
        self.top_k = top_k

    def load_esco_dataset(self, filepath: str) -> List[str]:
        """
        This method loads a Dataset with the European Skills and Competencies 

        :param filepath: Filepath to the ESCO dataset

        :return: List of ESCO skills 
        """
        skill_search_df = pd.read_csv(filepath)
        esco_skills = list(sorted(set(skill_search_df.skill.astype(str).tolist())))

        return esco_skills

    def create_index(self):
        
        encoded_data = self.sbert_model.encode(self.esco_skills)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        self.index.add_with_ids(encoded_data, np.array(range(0, len(self.esco_skills))))
        faiss.write_index(self.index, 'esco_skill_index')

    def run_query(self, query: str) -> Union[List, str]:

        query_vector = self.sbert_model.encode([query])
        top_k = self.index.search(query_vector, self.top_k)

        results = [self.esco_skills[_id] for _id in top_k[1].tolist()[0]]

        if top_k == 1:
            return results[0]
        else:
            return results
    
    def run_queries(self, queries: List[str]):
        query_results = []
        for query in queries:
            res = self.run_query(query=query)
            query_results.append(res[0])
        return query_results
