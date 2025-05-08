import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from docx import Document
import io

def preprocess_text(text):
    """Basic text cleaning"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def embed_topics(topics):
    """Generate sentence embeddings using SBERT"""
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/tmp/huggingface')
    return model.encode(topics)

def cluster_topics(topics, similarity_threshold=0.75):
    """Cluster topics based on semantic similarity"""
    if not topics:
        return []
    processed_topics = [preprocess_text(t) for t in topics]
    embeddings = embed_topics(processed_topics)
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=1 - similarity_threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(distance_matrix)
    cluster_map = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_map[label].append(topics[idx])
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
            avg_sim = np.mean(sub_matrix[np.triu_indices(len(sub_matrix), 1)]) if sub_matrix.size > 1 else 1.0
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

def extract_titles_from_docx(uploaded_files):
    """Extract headings from uploaded DOCX files"""
    titles = []
    for file in uploaded_files:
        try:
            doc = Document(file)
            for para in doc.paragraphs:
                if para.style.name.startswith('Heading'):
                    titles.append(para.text.strip())
                    break
            else:
                titles.append(f"Recommended Link ({file.name})")
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    return titles

def create_excel_file(clusters):
    """Create an Excel file from clustering results"""
    data = []
    for i, cluster in enumerate(clusters, 1):
        for topic in cluster['topics']:
            data.append({
                'Group': f"Group {i}",
                'Topic': topic,
                'Type': cluster['type'],
                'Action': cluster['action'],
                'Average Similarity': cluster['avg_similarity']
            })
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Topic Clusters')
    output.seek(0)
    return output

def main():
    st.title("🔍 Semantic Topic Clustering with SBERT")
    st.markdown("Group topics using **semantic similarity** for better deduplication and siloing suggestions.")

    st.sidebar.header("Input Options")
    uploaded_files = st.sidebar.file_uploader("Upload DOCX files", type="docx", accept_multiple_files=True)

    # Initialize session state for topics if not set
    if 'topics_input' not in st.session_state:
        st.session_state.topics_input = ""

    # Update session state with uploaded file titles
    if uploaded_files and st.button("Load Titles from Files"):
        titles = extract_titles_from_docx(uploaded_files)
        st.session_state.topics_input = "\n".join(titles)

    # Editable text area for topics
    input_text = st.text_area(
        "Enter or edit topics (one per line, e.g., Gold Loan Interest Rates):",
        value=st.session_state.topics_input,
        height=300,
        key="topics_input"
    )

    topics = [line.strip() for line in input_text.split('\n') if line.strip()]
    if not topics:
        st.warning("Please enter or upload at least one topic.")
        return

    sim_threshold = st.sidebar.slider("Similarity Threshold", 0.5, 0.95, 0.75, 0.05)

    if st.button("Run Analysis"):
        with st.spinner("Clustering topics..."):
            progress = st.progress(0)
            clusters = cluster_topics(topics, similarity_threshold=sim_threshold)
            progress.progress(1.0)

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

        # Generate and offer Excel download
        excel_file = create_excel_file(clusters)
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_file,
            file_name="topic_clusters.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.subheader("📌 Topic Clusters")
        for i, cluster in enumerate(clusters, 1):
            with st.expander(f"Group {i}: {cluster['type'].upper()} ({len(cluster['topics'])} topics) | Avg Sim: {cluster['avg_similarity']:.0%}"):
                for topic in cluster['topics']:
                    st.markdown(f"- {topic}")
                st.markdown(f"**Suggested Action:** `{cluster['action'].upper()}`")

if __name__ == "__main__":
    main()
