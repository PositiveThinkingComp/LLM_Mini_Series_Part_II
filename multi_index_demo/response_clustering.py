from scipy.cluster.hierarchy import linkage, dendrogram
import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns

@dataclass
class ClusterResult:
    similarity_df: pd.DataFrame
    cluster_df: pd.DataFrame

class ResponseClustering:
    """
    This class performs clustering of the Query Responses 
    """
    def __init__(self, sbert_model: SentenceTransformer):
        self.sbert_model = sbert_model

    def compute_response_clusters(self, response_df: pd.DataFrame) -> ClusterResult:
        """
        This method encodes the responses via SBERT, computes the cosine similarity of the 

        :param response_df: This is a DataFrame containing the query responses

        :return: ClusterResult object with the cosine similarity DataFrame and the Cluster Result DataFrame
        """
        embeddings = self.sbert_model.encode(response_df.response.tolist())

        self.plot_cluster_dendrogram(encodings=embeddings)

        # Compute the cosine similarity of the embeddings and plot the heatmap 
        similarity_mat = cosine_similarity(embeddings)
        cosine_sim_df = pd.DataFrame(similarity_mat, columns=response_df.id, index=response_df.id)
        self.plot_heatmap(similarity_df=cosine_sim_df)
        
        # Perform agglomerative clustering and plot the dendrogram 
        clustering = AgglomerativeClustering().fit(embeddings)
        response_df["cluster_labels"] = clustering.labels_
        response_df = response_df.sort_values("cluster_labels")
        cluster_result = ClusterResult(similarity_df=cosine_sim_df, cluster_df=response_df)

        return cluster_result

    def plot_cluster_dendrogram(self, encodings: np.ndarray):
        # Calculate the linkage: mergings
        mergings = linkage(encodings, method='ward')

        fig, ax = plt.subplots()
        # Plot the dendrogram, using varieties as labels
        fig.tight_layout()

        dendrogram(mergings,
                labels=[f"skills Data Scientist {i + 1}" for i in range(encodings.shape[0])],
                leaf_rotation=90,
                leaf_font_size=6,
                )

        plt.title("Clustering Dendrogram of the Data Scientist query-response embeddings")
        plt.xticks(rotation = 0)
        # plt.show()
        st.pyplot(fig)

    def plot_heatmap(self, similarity_df: pd.DataFrame):
        fig1, ax1 = plt.subplots()
        # Plot the dendrogram, using varieties as labels
        fig1.tight_layout()
        plt.title("Cosine-similarity heatmap of the Data Scientist query-response embeddings")
        sns.heatmap(similarity_df, cmap="viridis")
        st.pyplot(fig1)
