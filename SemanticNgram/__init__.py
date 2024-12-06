# Import necessary modules
import os

# Importing the main classes from Semantic_model.py to make them available at the package level.
from .Semantic_model import SemanticLanguageModel, SemanticUnigramModel, SemanticNgramModel

# Import necessary functions from download_word2vec
from .download_word2vec.download_word2vec import download_word2vec

# Define the path for the downloaded model
model_path = "model/GoogleNews-vectors-negative300.bin"

# Only download if the model doesn't already exist
if not os.path.exists(model_path):
    print("Word2Vec model not found. Downloading...")
    download_word2vec(destination_folder="model")
else:
    print("Word2Vec model already exists.")


