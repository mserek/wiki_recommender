import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wiki.recommender import Recommender
import matplotlib.pyplot as plt

plt.style.use("ggplot")
if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    data = df.loc[:, ["article", "lemmatized"]].assign(
        lemmatized=lambda df_: df_["lemmatized"]
        .str.split()
        .str[:500]
        .apply(lambda l: " ".join(l))
    )
    r = Recommender(mode="lemmatized")
    r.fit(data)
    r._articles = r._articles.drop(columns="lemmatized")

    with open("assets/saved_recommender.pickle", "wb") as f:
        pickle.dump(r, f)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["lemmatized"])
    tfidf_matrix = vectorizer.transform(df["lemmatized"])
    sims = cosine_similarity(tfidf_matrix)
    sims[sims > 0.99] = 0  # Removing similarities of articles to itself
    fig = plt.figure()
    pd.Series(sims.max(axis=1)).plot.bar(
        ylabel="Cosine similarity",
        title="Max similarity for articles",
        xticks=[],
        ylim=(0, 1),
        width=1,
    )
    fig.patch.set_alpha(0.0)
    plt.savefig("assets/max_similarity.png")
    plt.clf()
    fig = plt.figure()
    pd.Series(sims.max(axis=1)).plot.kde(
        xlim=(0, 1), title="Distribution of max similarities"
    )
    plt.xlabel("Cosine similarities")
    fig.patch.set_alpha(0.0)
    plt.savefig("assets/similarity_distribution.png")
    plt.clf()

    def plot_single_article(n, title):
        fig = plt.figure()
        pd.Series(sims[n]).plot.kde(
            xlim=(0, 1), title=f"Distribution of similarities for {title} article"
        )
        plt.axvline(x=sims[n].max(), color="green")
        plt.xlabel("Green line = similarity of the best match")
        fig.patch.set_alpha(0.0)
        plt.savefig(f"assets/{title}.png")
        plt.clf()

    n = sims.max(axis=1).argmin()
    name = df.iloc[n].loc["article"]
    plot_single_article(n, name)
    plt.clf()

    n = sims.max(axis=1).argmax()
    n_best_match = sims[n].argmax()
    best_match = df.iloc[n_best_match].loc["article"]

    name = df.iloc[n].loc["article"]
    plot_single_article(n, name)
    plt.clf()

    n = sims.mean(axis=1).argmax()
    name = df.iloc[n].loc["article"]
    plot_single_article(n, name)
    plt.clf()
    all_words = (
        pd.Series(df["cleaned_text"].sum().split()).astype("category").value_counts()
    )
    fig = plt.figure()
    all_words.iloc[30::-1].plot.barh(title="Word counts")
    fig.patch.set_alpha(0.0)
    plt.savefig("assets/word_counts.png")
    plt.clf()

    fig = plt.figure()
    all_words = (
        pd.Series(df["stopwords_filtered"].sum().split())
        .astype("category")
        .value_counts()
    )
    all_words.iloc[30::-1].plot.barh(title="Word counts - no stopwords")
    fig.patch.set_alpha(0.0)
    plt.savefig("assets/word_counts_nostopwords.png")
    plt.clf()
    fig = plt.figure()
    (
        all_words.loc[lambda x: x > 10]
        .div(all_words.sum())
        .reset_index(drop=True)
        .plot.line(
            loglog=True, title="Zipf's law", ylabel="Frequency", xlabel="Word Rank"
        )
    )
    fig.patch.set_alpha(0.0)
    plt.savefig("assets/zipf.png")
