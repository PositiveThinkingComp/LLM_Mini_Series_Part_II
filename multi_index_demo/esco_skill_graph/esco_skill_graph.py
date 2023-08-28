import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from typing import List, Dict, Union, Set
from matplotlib.pyplot import figure, text

class SkillGraph:
    """
    This is a Graph for visualizing the overlap of skills and competencies of different Candidates 
    """
    def __init__(self, graph_type: str="spring"):
        assert graph_type in ["spring", "shell"], "graph type must be 'spring' or 'shell'"
        self.graph_type = graph_type 

    def get_edge_df(self, skill_lists: List[Dict], unique_skills: Set) -> pd.DataFrame:
        """
        This method returns a DataFrame with the Edges of the Graph
        """
        relationship_list = []
        for skill_dict in skill_lists:
            candidate_id = skill_dict.get("id")
            intersections = unique_skills.intersection(set(skill_dict.get("skills")))
            for skill in list(set(intersections)):
                relationship_list.append({"from": candidate_id, "to": skill})

        relationships = pd.DataFrame(relationship_list)
        return relationships
        
    def get_node_df(self, unique_skills: Set, skill_lists: List[Dict]) -> pd.DataFrame:
        """
        This method returns a DataFrame with the nodes of the Graph
        """
        candidate_nodes = [{"ID": skill_dict.get("id"), "type": "candidate"} for skill_dict in skill_lists]
        skill_nodes =  [{"ID": skill, "type": "skill"} for skill in unique_skills]
        nodes = candidate_nodes + skill_nodes 
        carac = pd.DataFrame(nodes)
        return carac

    def plot_skill_graph(self, skill_lists: List[Dict]):
        """
        This method plots the actual skill graph based on a list of provided skills per candidate 

        :param skill_list: Skills per candidate 
        """
        unique_skills = set([skill for skills in skill_lists for skill in skills.get("skills")])

        relationships = self.get_edge_df(skill_lists=skill_lists, unique_skills=unique_skills)
        carac = self.get_node_df(skill_lists=skill_lists, unique_skills=unique_skills)

        # Set overall figure size
        fig, ax = plt.subplots()
        fig.tight_layout()

        # Create graph object
        G = nx.from_pandas_edgelist(relationships, 'from', 'to', create_using=nx.Graph())

        # Make types into categories
        carac= carac.set_index('ID')
        carac=carac.reindex(G.nodes())

        carac['type']=pd.Categorical(carac['type'])
        carac['type'].cat.codes

        # Set node colors
        cmap = matplotlib.colors.ListedColormap(['dodgerblue', 'lightgray']) #, 'darkorange'])

        # Set node sizes
        node_sizes = [1000 if entry == 'candidate' else 250 for entry in carac.type]

        if self.graph_type == "spring":

            pos = nx.spring_layout(G)
            # Create Layouts
            nx.draw(G, pos=pos, with_labels=False, node_color=carac['type'].cat.codes, cmap=cmap, 
                    node_size = node_sizes, edgecolors='gray')
            
        elif self.graph_type == "shell":
            pos = nx.shell_layout(G)
            nx.draw_shell(G, pos=pos, with_labels=False, node_color=carac['type'].cat.codes, cmap=cmap, 
            node_size = node_sizes, edgecolors='gray')

        for node, (x, y) in pos.items():
            text(x, y, node, fontsize=8, ha='center', va='center')

        plt.title('European Skills, Competences, Qualifications and Occupations (ESCO) skill network', fontsize=14)

        st.pyplot(fig)
