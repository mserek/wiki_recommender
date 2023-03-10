"""
This module contains utilities to scrape and preprocess text data from Wikipedia articles.

Functions
---------
get_wiki_texts(links: List[str], limit: int = 50, sleep_time: float = 0.5) -> List[Tuple[str, str]]:
    Scrape the text content from a list of Wikipedia article links.

preprocess_texts(text_df: pd.DataFrame) -> pd.DataFrame:
    Preprocess the raw text paragraphs in a DataFrame by applying stemming and lemmatization,
    and removing English stopwords.

Command-line interface
----------------------
When this module is executed as a script, it provides a command-line interface to scrape
Wikipedia articles and save the preprocessed data to a pickled pandas Dataframe.

positional arguments:
  links                 one or more Wikipedia article links in the format
                        "https://en.wikipedia.org/wiki/TITLE",
                        "en.wikipedia.org/wiki/TITLE", or
                        "/wiki/TITLE"
                        that the search starts from

optional arguments:
  -h, --help            show this help message and exit
  -s SLEEP, --sleep SLEEP
                        sleep time between requests to wikipedia (in seconds)
  -n NUM_ARTICLES, --num_articles NUM_ARTICLES
                        number of articles to scrape
  -o OUTPUT, --output OUTPUT
                        output file path

Example:
    python text_prep.py "https://en.wikipedia.org/wiki/Joseph_Fourier" -n 100 -s 1 -o results/
"""

import argparse
import re
from collections import deque
from pathlib import Path
from time import sleep
from typing import List, Tuple

import nltk  # type: ignore
import pandas as pd
import requests
from bs4 import BeautifulSoup


def _validate_url(url: str) -> bool:
    return ":" not in url and "#" not in url and url.startswith("/wiki/")


def scrape_wiki_texts(
    links: List[str], limit: int = 50, sleep_time: float = 0.5
) -> List[Tuple[str, str]]:
    """
    Scrape text content of Wikipedia pages.

    Traverses wikipedia pages using BFS and retrieves articles' text.

    Parameters
    ----------
    links : list of str
        A list of URLs used as starting points in the BFS.
    limit : int, optional
        The maximum number of pages to be visited. Defaults to 50.
    sleep_time : float, optional
        The minimal amount of time to wait (in seconds) between sending requests. Defaults to 0.5.

    Returns
    -------
    list of tuple of str, str
        A list of tuples where the first element is the URL (with the prefix removed),
        the second element is the raw text content of the Wikipedia page.
    """

    queue: deque = deque()
    visited = set()
    currently_visited = 0
    out = []

    for url in links:
        url_no_prefix = url.removeprefix("https://").removeprefix("en.wikipedia.org")
        if url_no_prefix not in visited:
            queue.append(url_no_prefix)

    while queue and currently_visited < limit:
        url = queue.popleft()
        if _validate_url(url) and url not in visited:
            currently_visited += 1
            visited.add(url)
            sleep(sleep_time)
            response = requests.get(f"https://en.wikipedia.org{url}", timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                page = soup.find_all("p")
                raw_text = "".join((paragraph.text for paragraph in page))
                out.append((url[6:], raw_text))
                children = soup.find_all("a", attrs={"href": re.compile(r"^/wiki")})
                for child_link in children:
                    url_no_prefix = (
                        child_link["href"]
                        .removeprefix("https://")
                        .removeprefix("en.wikipedia.org")
                    )
                    queue.append(url_no_prefix)
    return out


def preprocess_articles(text_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Preprocess text data in a DataFrame.

    This function applies several text preprocessing steps to a DataFrame containing raw text data.
    The steps include removing duplicate paragraphs, filtering non-alphanumeric characters,
    removing english stopwords, stemming, lemmatization.

    Parameters
    ----------
    text_df : pandas.DataFrame
        A DataFrame containing raw text data in the 'raw_paragraph' column.
    text_column : str
        Name of the column that contains text to perfrom processing on.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the following additional columns:
        - 'cleaned_text': text with non-alphanumeric characters removed
        - 'stopwords_filtered': cleaned text data with English stopwords removed.
        - 'stemmed': cleaned text data after stemming and stopword removal.
        - 'lemmatized': cleaned text data after lemmatization and stopword removal.
    """

    stopwords_english = nltk.corpus.stopwords.words("english")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    return text_df.drop_duplicates(subset=text_column, ignore_index=True).assign(
        cleaned_text=text_df[text_column].apply(
            lambda text: "".join(
                (
                    letter.lower() if not letter.isspace() else " "
                    for letter in text
                    if letter.isalpha() or letter.isspace()
                )
            )
        ),
        stopwords_filtered=lambda df_: df_["cleaned_text"].apply(
            lambda text: " ".join(
                (word for word in text.split() if word not in stopwords_english)
            )
        ),
        stemmed=lambda df_: df_["stopwords_filtered"].apply(
            lambda text: " ".join((stemmer.stem(word) for word in text.split()))
        ),
        lemmatized=lambda df_: df_["stopwords_filtered"].apply(
            lambda text: " ".join((lemmatizer.lemmatize(word) for word in text.split()))
        ),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Scrape text data from Wikipedia articles."
    )
    parser.add_argument(
        "links",
        nargs="+",
        help="""
             accepts one or more articles in forms like 
             "https://en.wikipedia.org/wiki/Joseph_Fourier", 
             "en.wikipedia.org/wiki/Joseph_Fourier" or 
             "/wiki/Joseph_Fourier"
             """,
    )
    parser.add_argument(
        "-s",
        "--sleep",
        type=float,
        default=0.5,
        help="sleep time between requests (in seconds)",
    )
    parser.add_argument(
        "-n",
        "--num_articles",
        type=int,
        default=50,
        help="number of articles to scrape",
    )
    parser.add_argument("-o", "--output", type=str, default="", help="output file path")
    args = parser.parse_args()

    raw_texts = scrape_wiki_texts(
        args.links, limit=args.num_articles, sleep_time=args.sleep
    )
    article, texts = [list(values) for values in zip(*raw_texts)]
    df = pd.DataFrame().assign(article=article, raw_paragraph=texts)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    preprocess_articles(df, text_column="raw_paragraph").to_pickle(
        out_path / Path("results.pickle")
    )
