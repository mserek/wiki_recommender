"""
Module providing a content-based recommender system.

The `Recommender` class implements a recommendation engine that takes as input a corpus
of articles and allows the user to query it with a set of URLs from Wikipedia.
Additionally, implements relevance feedback using Rocchio algorithm.

The `RocchioConfig` defines a configuration object with parameters for the Rocchio algorithm.
"""

from dataclasses import dataclass
from typing import Sequence, List

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import scipy.sparse  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from wiki import text_prep


@dataclass
class RocchioConfig:
    """
    A dataclass that stores the configuration parameters for the Rocchio algorithm.

    Attributes:
    -----------
    alpha : float
        The weight given to the original query vector.
    beta : float
        The weight given to the centroid of the relevant documents.
    gamma : float
        The weight given to the centroid of the irrelevant documents.
    """

    alpha: float
    beta: float
    gamma: float


_article_transformers = {
    "lemmatized": text_prep.lemmatize_text,
    "stemmed": text_prep.stem_text,
}


class Recommender:
    """
    Recommender system based on TF-IDF.

    Attributes
    ----------
    rocchio : RocchioConfig
        A configuration object for Rocchio's algorithm.
    mode : str
        A string indicating the text processing mode. One of 'lemmatized' or 'stemmed'.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.8,
        gamma: float = 0.1,
        mode: str = "lemmatized",
    ) -> None:
        self.rocchio = RocchioConfig(alpha, beta, gamma)
        self.mode = mode
        self._queried_articles: List[str] = []
        self._query_tfidf_centroid: scipy.sparse.csr_matrix
        self._tfidf_matrix: scipy.sparse.csr_matrix
        self._articles: pd.DataFrame
        self._vectorizer: TfidfVectorizer

    def fit(self, articles: pd.DataFrame) -> None:
        """
        Fit the recommender to the corpus of articles.

        Parameters
        ----------
        articles : pd.DataFrame
            A pandas DataFrame containing the corpus of articles.
            Has to contain "articles" column with article names and
            a column with a name equivalent to the mode
            specified during the instantiation of the Recommender object.
        """
        self._articles = articles
        self._vectorizer = TfidfVectorizer()
        self._vectorizer.fit(self._articles[self.mode])
        self._tfidf_matrix = self._vectorizer.transform(self._articles[self.mode])

    def recommend(self, urls: Sequence, num_of_articles: int = 5) -> pd.Series:
        """
        Recommend articles based on the provided URLs.

        Parameters
        ----------
        urls : Sequence
            A sequence of URLs to use as the basis for the recommendation.
        num_of_articles : int, optional
            The number of articles to recommend, by default 5.

        Returns
        -------
        pd.Series
            A Series containing the recommended article titles indexed by internal id.

        Notes
        -----
        Assumes that the `fit` method has already been called before.
        """
        self._query_tfidf_centroid = None
        self._queried_articles = []
        self._calculate_query_centroid(urls)
        return self._get_best_match(num_of_articles)

    def recommend_with_feedback(self, relevant=(), irrelevant=(), num_of_articles=5):
        """
        Recommend articles based on the provided feedback.

        Uses Rocchio SMART algorithm.

        Parameters
        ----------
        relevant : Sequence
            A sequence of indices of relevant articles to be used for Rocchio algorithm.
        irrelevant : Sequence
            A sequence of indices of irrelevant articles to be used for Rocchio algorithm.
        num_of_articles : int
            The number of articles to recommend. Defaults to 5.

        Returns
        -------
        pd.Series
            A Series containing the recommended article titles indexed by internal id.

        Notes
        -----
        Assumes that the `recommend` method has already been called before.
        """
        self._update_query_centroid(relevant, irrelevant)
        return self._get_best_match(num_of_articles)

    def _get_best_match(self, num_of_articles) -> pd.Series:
        cosine_similarities = cosine_similarity(
            self._tfidf_matrix, self._query_tfidf_centroid
        )
        best_matches = cosine_similarities.reshape(
            cosine_similarities.shape[0]
        ).argsort()[::-1]
        return (
            self._articles.iloc[best_matches]
            .loc[lambda df_: ~df_["article"].isin(self._queried_articles)]
            .iloc[:num_of_articles]
            .article.apply(lambda x: f"https://en.wikipedia.org/wiki/{x}")
        )

    def _calculate_query_centroid(self, urls) -> None:
        for url in urls:
            text = self._query_wiki(url)
            if text is not None:
                tfidf_query = self._vectorizer.transform([text])
                if self._query_tfidf_centroid is None:
                    self._query_tfidf_centroid = tfidf_query
                else:
                    self._query_tfidf_centroid += tfidf_query
            self._query_tfidf_centroid /= len(urls)

    def _query_wiki(self, url: str) -> str:
        text = ""
        title = text_prep.get_article_title(url)
        soup = text_prep.get_soup(title)
        if soup is not None:
            self._queried_articles.append(title)
            text = text_prep.get_raw_text(soup)
            text = text_prep.clean_text(text)
            text = text_prep.remove_stopwords(text)
            text = _article_transformers[self.mode](text)
        return text

    def _update_query_centroid(self, relevant, irrelevant):
        self._query_tfidf_centroid = self.rocchio.alpha * self._query_tfidf_centroid
        if relevant:
            relevant_centroid = self._tfidf_matrix[relevant].mean(axis=0)
            self._query_tfidf_centroid += self.rocchio.beta * relevant_centroid
        if irrelevant:
            irrelevant_centroid = self._tfidf_matrix[irrelevant].mean(axis=0)
            self._query_tfidf_centroid -= self.rocchio.gamma * irrelevant_centroid
        self._query_tfidf_centroid = scipy.sparse.csr_matrix(self._query_tfidf_centroid)

    def breakdown_prediction(self, index: int, length: int = 30):
        """
        Return a bar plot displaying the breakdown of the predicted score for an article.

        Parameters
        ----------
        index : int
            The index of the article to break down the prediction for.
        length : int, optional
            The number of terms to display in the plot, by default 30.

        Returns
        -------
        matplotlib.figure.Figure
            The bar plot.
        """
        fig = plt.figure()
        res = (
            self._tfidf_matrix[index].toarray()[0]
            * self._query_tfidf_centroid.toarray()[0]
        )
        (
            pd.Series(res)
            .rename({v: k for k, v in self._vectorizer.vocabulary_.items()})
            .sort_values(ascending=False)
            .div(res.sum())
            .mul(100)
            .iloc[:length]
            .plot.bar()
        )
        fig.patch.set_alpha(0.0)
        return fig
