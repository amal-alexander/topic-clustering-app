import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

def preprocess_text(text):
    """Basic text cleaning"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def embed_topics(topics):
    """Generate sentence embeddings using SBERT"""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & fast
    return model.encode(topics)

def cluster_topics(topics, similarity_threshold=0.75):
    processed_topics = [preprocess_text(t) for t in topics]
    embeddings = embed_topics(processed_topics)
    similarity_matrix = cosine_similarity(embeddings)

    # Convert similarity to distance for clustering
    distance_matrix = 1 - similarity_matrix

    # Clustering with dynamic threshold
    clustering = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        distance_threshold=1 - similarity_threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(distance_matrix)

    # Group topics
    cluster_map = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_map[label].append(topics[idx])

    # Analyze similarity within clusters
    final_clusters = []
    for cluster_id, cluster_topics in cluster_map.items():
        if len(cluster_topics) == 1:
            final_clusters.append({
                'topics': cluster_topics,
                'type': 'unique',
                'action': 'keep',
                'avg_similarity': 1.0
            })
        else:
            cluster_indices = [i for i, t in enumerate(topics) if t in cluster_topics]
            sub_matrix = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
            avg_sim = np.mean(sub_matrix[np.triu_indices(len(sub_matrix), 1)])

            if avg_sim >= 0.85:
                cluster_type = 'full_duplicate'
                action = 'merge'
            elif avg_sim >= 0.7:
                cluster_type = 'partial_duplicate'
                action = 'rewrite or merge'
            else:
                cluster_type = 'related'
                action = 'review for siloing'

            final_clusters.append({
                'topics': cluster_topics,
                'type': cluster_type,
                'action': action,
                'avg_similarity': avg_sim
            })

    return final_clusters

def main():
    st.title("🔍 Semantic Topic Clustering with SBERT")
    st.markdown("Group topics using **semantic similarity** for better deduplication and siloing suggestions.")

    input_text = st.text_area("Enter one topic per line:", height=300)
    if not input_text.strip():
        st.warning("Please enter at least 2 topics.")
        return

    topics = [line.strip() for line in input_text.split('\n') if line.strip()]
    if len(topics) < 2:
        st.warning("Need at least 2 topics to cluster.")
        return

    sim_threshold = st.sidebar.slider("Similarity Threshold", 0.5, 0.95, 0.75, 0.05)

    if st.button("Run Analysis"):
        with st.spinner("Clustering topics..."):
            clusters = cluster_topics(topics, similarity_threshold=sim_threshold)

        st.success(f"Done! Found {len(clusters)} groups.")

        dup_summary = {
            'full_duplicate': 0,
            'partial_duplicate': 0,
            'related': 0,
            'unique': 0
        }
        for cluster in clusters:
            dup_summary[cluster['type']] += 1

        cols = st.columns(4)
        cols[0].metric("Full Duplicates", dup_summary['full_duplicate'])
        cols[1].metric("Partial Duplicates", dup_summary['partial_duplicate'])
        cols[2].metric("Related", dup_summary['related'])
        cols[3].metric("Unique", dup_summary['unique'])

        st.subheader("📌 Topic Clusters")
        for i, cluster in enumerate(clusters, 1):
            with st.expander(f"Group {i}: {cluster['type'].upper()} ({len(cluster['topics'])} topics) | Avg Sim: {cluster['avg_similarity']:.0%}"):
                for topic in cluster['topics']:
                    st.markdown(f"- {topic}")
                st.markdown(f"**Suggested Action:** `{cluster['action'].upper()}`")

if __name__ == "__main__":
    main()
