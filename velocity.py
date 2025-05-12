import os
os.environ["STREAMLIT_WATCHER_IGNORE_MODULES"] = "torch"
import streamlit as st
import pandas as pd
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(layout="wide")

def extract_titles_from_files(uploaded_files):
    """Extract titles from DOCX, TXT, CSV, XLSX files"""
    titles = []
    for file in uploaded_files:
        try:
            filename = file.name.lower()

            if filename.endswith(".docx"):
                doc = Document(file)
                for para in doc.paragraphs:
                    if para.style.name.startswith('Heading') and para.text.strip():
                        titles.append(para.text.strip())
                        break
                else:
                    titles.append(f"Recommended Link ({file.name})")

            elif filename.endswith(".txt"):
                lines = file.read().decode("utf-8").splitlines()
                titles.extend([line.strip() for line in lines if line.strip()])

            elif filename.endswith(".csv"):
                df = pd.read_csv(file)
                col = st.sidebar.selectbox(f"Select column from {file.name}", df.columns)
                titles.extend(df[col].dropna().astype(str).tolist())

            elif filename.endswith(".xlsx"):
                df = pd.read_excel(file)
                col = st.sidebar.selectbox(f"Select column from {file.name}", df.columns)
                titles.extend(df[col].dropna().astype(str).tolist())

            else:
                st.warning(f"Unsupported file type: {file.name}")
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    return titles

def cluster_titles(titles, num_clusters):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(titles)
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(X)
    labels = model.labels_
    return labels

def display_clusters(titles, labels, num_clusters):
    clusters = [[] for _ in range(num_clusters)]
    for title, label in zip(titles, labels):
        clusters[label].append(title)

    for i, cluster in enumerate(clusters):
        st.subheader(f"Cluster {i + 1}")
        for title in cluster:
            st.write(f"- {title}")

def generate_wordcloud(titles):
    text = " ".join(titles)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.image(wordcloud.to_array(), use_column_width=True)

def main():
    st.title("🔍 Topic Clustering Tool")

    st.sidebar.header("📂 Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Files (DOCX, TXT, CSV, XLSX)", 
        type=["docx", "txt", "csv", "xlsx"], 
        accept_multiple_files=True
    )

    st.sidebar.header("⚙️ Clustering Options")
    num_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)

    if uploaded_files:
        titles = extract_titles_from_files(uploaded_files)
        if not titles:
            st.warning("No titles found in uploaded files.")
            return

        st.header("📄 Extracted Titles")
        st.write(titles)

        labels = cluster_titles(titles, num_clusters)
        st.header("📊 Clusters")
        display_clusters(titles, labels, num_clusters)

        st.header("☁️ Word Cloud of Topics")
        generate_wordcloud(titles)
    else:
        st.info("Upload one or more files from the sidebar to begin.")

    st.markdown("---")
    st.subheader("📝 How to Use This Tool:")
    st.markdown("""
    1. **Upload one or more files** (`.docx`, `.txt`, `.csv`, or `.xlsx`) via the sidebar.
    2. **For CSV/XLSX files**, choose the column containing topic titles.
    3. **Set the number of clusters** you want using the slider.
    4. View **extracted topics**, **clusters**, and a **word cloud** of the keywords.
    """)

if __name__ == "__main__":
    main()
