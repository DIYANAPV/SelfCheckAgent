import gensim.downloader as api

def download_word2vec():
    print("Downloading Word2Vec model...")
    api.load("word2vec-google-news-300")  # This automatically caches the model in ~/.gensim-data/
    print("Word2Vec model downloaded and cached.")

if __name__ == "__main__":
    download_word2vec()
