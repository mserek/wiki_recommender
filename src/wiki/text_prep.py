"""
Module containing utilities to scrape and preprocess text data from Wikipedia articles.

Command-line interface
----------------------
When executed as a script, it provides a command-line interface to scrape
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


def get_article_title(url: str) -> str:
    """
    Return wikipedia article title given a Wikipedia URL.

    Parameters
    ----------
    url : str
        The Wikipedia URL.

    Returns
    -------
    str
        The article title extracted from the URL.

    Examples
    --------
    >>> get_article_title('https://en.wikipedia.org/wiki/Lemmatisation')
    'Lemmatisation'

    >>> get_article_title('en.wikipedia.org/wiki/Lemmatisation')
    'Lemmatisation'

    >>> get_article_title('/wiki/Lemmatisation')
    'Lemmatisation'
    """
    return (
        url.removeprefix("https://")
        .removeprefix("en.wikipedia.org")
        .removeprefix("/wiki/")
    )


def _validate_title(title: str) -> bool:
    return ":" not in title and "#" not in title


def get_soup(title: str) -> BeautifulSoup | None:
    """
    Fetch the HTML content of a given Wikipedia page.

    Parameters
    ----------
    title : str
        The title of the Wikipedia page to fetch i.e. last part of URL's path component

    Returns
    -------
    BeautifulSoup or None
        A BeautifulSoup object representing the HTML content of the page, if the request
        was successful. Returns None if the request failed.

    Examples
    --------
    >>> soup = get_soup("Joseph_Fourier")
    >>> if soup is not None:
    ...     print("Page scraped successfully!")
    ... else:
    ...     print("Unable to scrape page.")

    """
    response = requests.get(f"https://en.wikipedia.org/wiki/{title}", timeout=5)
    if response.status_code == 200:
        return BeautifulSoup(response.text, "html.parser")
    return None


def get_raw_text(soup: BeautifulSoup) -> str:
    """
    Extract the raw text content of a given Wikipedia page paragraphs.

    Parameters
    ----------
    soup : BeautifulSoup
        A BeautifulSoup object representing the HTML content of a Wikipedia page.

    Returns
    -------
    str
        The raw text content of the page's paragraphs.
    """
    paragraphs = soup.find_all("p")
    return "".join((paragraph.text for paragraph in paragraphs))


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
    total_visited = 0
    out = []
    for title in links:
        title = get_article_title(title)
        if title not in visited:
            queue.append(title)

    while queue and total_visited < limit:
        title = queue.popleft()
        if _validate_title(title) and title not in visited:
            total_visited += 1
            visited.add(title)
            sleep(sleep_time)
            soup = get_soup(title)
            if soup is not None:
                raw_text = get_raw_text(soup)
                out.append((title, raw_text))

                children = soup.find_all("a", attrs={"href": re.compile(r"^/wiki")})
                for child_link in children:
                    title = get_article_title(child_link["href"])
                    queue.append(title)
    return out

  
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
_stemmer = nltk.stem.PorterStemmer()
_lemmatizer = nltk.stem.WordNetLemmatizer()
_stopwords_english = nltk.corpus.stopwords.words("english")


def clean_text(text: str) -> str:
    """
    Clean a given text.

    Casefolds all letters ande and replaces all whitespace characters with spaces.
    Drops non-alphanumeric and non-whitespace characters.

    Parameters
    ----------
    text : str
        The text to be cleaned.

    Returns
    -------
    str
        The cleaned text.
    """
    return "".join(
        (
            letter.casefold() if not letter.isspace() else " "
            for letter in text
            if letter.isalpha() or letter.isspace()
        )
    )


def remove_stopwords(text: str) -> str:
    """
    Remove all stopwords from a given text.

    Parameters
    ----------
    text : str
        The text from which to remove stopwords.

    Returns
    -------
    str
        The input text with all stopwords removed.
    """
    return " ".join((word for word in text.split() if word not in _stopwords_english))


def lemmatize_text(text: str) -> str:
    """
    Lemmatizes a given text by reducing all words to their base form.

    Parameters
    ----------
    text : str
        The text to be lemmatized.

    Returns
    -------
    str
        The lemmatized text.
    """
    return " ".join((_lemmatizer.lemmatize(word) for word in text.split()))


def stem_text(text: str) -> str:
    """
    Stems a given text by reducing all words to their stem.

    Parameters
    ----------
    text : str
        The text to be stemmed.

    Returns
    -------
    str
        The stemmed text.
    """
    return " ".join((_stemmer.stem(word) for word in text.split()))


def preprocess_articles(text_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Preprocess text data in a DataFrame.

    Applies several text preprocessing steps to a DataFrame containing raw text data.
    The steps include removing duplicates, filtering non-alphanumeric characters,
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
        A new DataFrame with new index and the following additional columns:
        - 'cleaned_text': text with non-alphanumeric characters removed
        - 'stopwords_filtered': cleaned text data with English stopwords removed.
        - 'stemmed': cleaned text data after stemming and stopword removal.
        - 'lemmatized': cleaned text data after lemmatization and stopword removal.
    """
    return (
        text_df.drop_duplicates(subset=text_column, ignore_index=True)
        .assign(
            cleaned_text=text_df[text_column].apply(clean_text),
            stopwords_filtered=lambda df_: df_["cleaned_text"].apply(remove_stopwords),
            stemmed=lambda df_: df_["stopwords_filtered"].apply(stem_text),
            lemmatized=lambda df_: df_["stopwords_filtered"].apply(lemmatize_text),
        )
        .reset_index(drop=True)
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
