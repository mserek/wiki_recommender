import pickle
from urllib.parse import unquote

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import streamlit as st

from src.recommender import Recommender, RocchioConfig

plt.style.use("ggplot")


def setup():
    with open("assets/saved_recommender.pickle", "rb") as f:
        r = pickle.load(f)
    st.session_state["recommender"] = r
    st.session_state["last_was_feedback"] = False
    st.session_state["num_articles_feedback"] = 5


if "recommender" not in st.session_state:
    setup()
else:
    recommender_ = st.session_state["recommender"]


def app():
    st.set_page_config(
        page_title="Wiki recommender",
        layout="wide",
    )
    st.header("Wikipedia recommendations")
    tab_recomms, tab_plots = st.tabs(["Recommendations", "Dataset statistics"])
    with tab_recomms:
        st.sidebar.subheader("Recommendations:")
        with st.sidebar.form("List of articles:"):
            links = st.text_area("Wiki links:", height=3, max_chars=None).split()[:5]
            num_articles = st.number_input(
                "How many articles to get?", value=5, min_value=1, max_value=20, step=1
            )
            submit = st.form_submit_button(
                "Get recommendations!", on_click=callback_submit_links
            )

        recomms = pd.Series([], dtype="str")

        if st.session_state["last_was_feedback"]:
            recomms = recommender_.recommend_with_feedback(
                relevant=st.session_state["feedbacks"]["Relevant"],
                irrelevant=st.session_state["feedbacks"]["Irrelevant"],
                num_of_articles=st.session_state["num_articles_feedback"],
            )
        elif submit and links:
            recomms = recommender_.recommend(links, num_articles)

        left, right = st.columns(2)
        columns = {1: left, 0: right}
        for i, (recommendation_index, article) in enumerate(recomms.items(), start=1):
            col = columns[i % 2]
            col.subheader(f"{i}. {format_link(article)}")
            col.pyplot(recommender_.breakdown_prediction(recommendation_index))

        if st.session_state["last_was_feedback"] or submit:
            st.sidebar.subheader("Relevance feedback:")
            with st.sidebar.form("Feedback:"):
                for i, (recommendation_index, article) in enumerate(
                    recomms.items(), start=1
                ):
                    st.write(f"{i}.{format_link(article)}")
                    st.radio(
                        "Was this relevant?",
                        ["No decision", "Relevant", "Irrelevant"],
                        key=f"{recommendation_index}radio",
                    )
                num_articles_feedback = st.number_input(
                    "How many articles to get?",
                    value=10,
                    min_value=1,
                    max_value=20,
                    step=1,
                    key="feedback_count",
                )
                st.subheader("Hyperparameters:")
                alpha = st.number_input(
                    "Alpha",
                    value=1.0,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.05,
                    key="alpha",
                )

                beta = st.number_input(
                    "Beta",
                    value=0.75,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.05,
                    key="beta",
                )

                gamma = st.number_input(
                    "Gamma",
                    value=0.15,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.05,
                    key="gamma",
                )
                rocchio = alpha, beta, gamma
                st.form_submit_button(
                    "Submit feedback!",
                    on_click=callback_feedback,
                    args=(recomms.index, num_articles_feedback, rocchio),
                )
        with st.expander(
            "Guide:", expanded=not (submit or st.session_state["last_was_feedback"])
        ):
            st.write(
                """
            Submit 1-5 links for wikipedia articles in the sidebar to get recommendations of similar atricles.
            If there is more than one link, separate them with whitespace. They can be in the following forms: 
        
        https://en.wikipedia.org/wiki/Joseph_Fourier
        en.wikipedia.org/wiki/Joseph_Fourier
        /wiki/Joseph_Fourier
        Joseph_Fourier
             
    When you submit a query, the system retrieves a set of documents from Wikipedia that might be relevant to your search.
    
    If some of the documents are relevant to what you're looking for, you can give the system feedback by marking
     those documents as 'relevant.' Similarly, if some documents aren't relevant, you can mark them as 'irrelevant.'
    
    The system then uses the Rocchio algorithm, also known as SMART, to adjust your query and get better search
     results. It does this by moving your query closer to the center of the relevant documents and farther away
      from the center of the non-relevant documents.
    
    To find the right balance between your original query and the feedback you gave, the system uses weights called
     alpha, beta, and gamma. Alpha is the weight of your original query, beta is the weight of the relevant
      documents you marked, and gamma is the weight of the non-relevant documents you marked.
    
    Usually, the weight of your original query is the highest and the weight of the non-relevant documents is the
     lowest. However, if you give the system a lot of feedback, it may be a good idea to increase beta and gamma."
            """
            )
    with tab_plots:
        wc1, wc2 = st.columns(2)
        with wc1:
            st.image("assets/word_counts.png")
            st.image("assets/similarity_distribution.png")
            st.image("assets/Le_Malade_imaginaire.png")
        with wc2:
            st.image("assets/word_counts_nostopwords.png")
            st.image("assets/max_similarity.png")
            st.image("assets/France.png")


def callback_feedback(keys, num_articles, rocchio):
    st.session_state["last_was_feedback"] = True
    feedbacks = {"Relevant": [], "Irrelevant": [], "No decision": []}
    for index in keys:
        opinion = st.session_state[f"{index}radio"]
        feedbacks[opinion].append(index)
    st.session_state["feedbacks"] = feedbacks
    st.session_state["num_articles_feedback"] = num_articles
    st.session_state["recommender"].rocchio = RocchioConfig(*rocchio)


def callback_submit_links():
    st.session_state["last_was_feedback"] = False
    st.session_state["feedbacks"] = {}


def format_link(link: str):
    return f"[{unquote(link.split('/')[-1])}]({link})"


if __name__ == "__main__":
    app()
