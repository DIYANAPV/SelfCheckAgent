import spacy
import numpy as np
from nltk.util import ngrams
from typing import Dict, List, Set, Tuple, Union
from scipy.spatial.distance import cosine
import gensim.downloader as api  # Use Gensim's downloader

class SemanticLanguageModel:
    def __init__(self, lowercase: bool = True, similarity_threshold: float = 0.9) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.token_count = 0
        self.counts = {'<unk>': 0}
        self.lowercase = lowercase
        self.similarity_threshold = similarity_threshold

        # Automatically download and load the Word2Vec model using Gensim's API
        print("Downloading Word2Vec model via Gensim...")
        self.word2vec = api.load("word2vec-google-news-300")  # Download pre-trained Word2Vec
        print("Word2Vec model loaded successfully.")
        self.token_vectors = {}

    def _get_vector(self, token: str) -> np.ndarray:
        if token not in self.token_vectors:
            if self.word2vec and token in self.word2vec:
                self.token_vectors[token] = self.word2vec[token]
            else:
                self.token_vectors[token] = np.zeros(300)  # Default to zero vector
        return self.token_vectors[token]
    
    # (Keep all other methods as is)

    def _are_similar(self, token1: str, token2: str) -> bool:
        vec1 = self._get_vector(token1)
        vec2 = self._get_vector(token2)
        similarity = 1 - cosine(vec1, vec2)
        return similarity >= self.similarity_threshold

    def _get_similar_tokens(self, token: str) -> Set[str]:
        similar_tokens = {token}
        for other_token in list(self.token_vectors.keys()):
            if self._are_similar(token, other_token):
                similar_tokens.add(other_token)
        return similar_tokens

    def train(self, k: int = 1) -> None:
        self.probs = {}
        for item, item_count in self.counts.items():
            prob_nom = item_count + k
            prob_denom = self.token_count + k * len(self.counts)
            self.probs[item] = prob_nom / prob_denom

    def evaluate(self, sentences: List[str]) -> Dict[str, Dict[str, Union[List[float], float]]]:
        avg_neg_logprob = []
        max_neg_logprob = []
        logprob_doc = []
        for sentence in sentences:
            logprob_sent = []
            tokens = [token.text for token in self.nlp(sentence)]
            if self.lowercase:
                tokens = [token.lower() for token in tokens]
            for token in tokens:
                similar_tokens = self._get_similar_tokens(token)
                prob = sum(self.probs.get(similar_token, 0) for similar_token in similar_tokens)
                if prob == 0:
                    prob = self.probs['<unk>']
                logprob = np.log(prob)
                logprob_sent.append(logprob)
                logprob_doc.append(logprob)
            avg_neg_logprob.append(-1.0 * np.mean(logprob_sent))
            max_neg_logprob.append(-1.0 * np.min(logprob_sent))
        avg_neg_logprob_doc = -1.0 * np.mean(logprob_doc)
        avg_max_neg_logprob_doc = np.mean(max_neg_logprob)
        return {
            'sent_level': {'avg_neg_logprob': avg_neg_logprob, 'max_neg_logprob': max_neg_logprob},
            'doc_level': {'avg_neg_logprob': avg_neg_logprob_doc, 'avg_max_neg_logprob': avg_max_neg_logprob_doc},
        }


class SemanticUnigramModel(SemanticLanguageModel):
    def add(self, text: str) -> None:
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        for sentence in sentences:
            tokens = [token.text for token in self.nlp(sentence)]
            if self.lowercase:
                tokens = [token.lower() for token in tokens]
            self.token_count += len(tokens)
            for token in tokens:
                similar_tokens = self._get_similar_tokens(token)
                for similar_token in similar_tokens:
                    if similar_token not in self.counts:
                        self.counts[similar_token] = 1
                    else:
                        self.counts[similar_token] += 1


class SemanticNgramModel(SemanticLanguageModel):
    def __init__(self, n: int, lowercase: bool = True, left_pad_symbol: str = '<s>', similarity_threshold: float = 0.9) -> None:
        super().__init__(lowercase, similarity_threshold)
        self.n = n
        self.left_pad_symbol = left_pad_symbol

    def add(self, text: str) -> None:
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        for sentence in sentences:
            tokens = [token.text for token in self.nlp(sentence)]
            if self.lowercase:
                tokens = [token.lower() for token in tokens]
            ngs = list(ngrams(tokens, n=self.n, pad_left=True, left_pad_symbol=self.left_pad_symbol))
            self.token_count += len(ngs)
            for ng in ngs:
                similar_ngs = self._get_similar_ngrams(ng)
                for similar_ng in similar_ngs:
                    if similar_ng not in self.counts:
                        self.counts[similar_ng] = 1
                    else:
                        self.counts[similar_ng] += 1

    def _get_similar_ngrams(self, ng: Tuple[str]) -> Set[Tuple[str]]:
        similar_ngs = {ng}
        for i, token in enumerate(ng):
            similar_tokens = self._get_similar_tokens(token)
            for similar_token in similar_tokens:
                similar_ng = ng[:i] + (similar_token,) + ng[i+1:]
                similar_ngs.add(similar_ng)
        return similar_ngs

def semantic_model_predict(passage: str, sampled_passages: List[str], n: int, word2vec_model_path: str = None) -> Dict[str, Dict[str, Union[List[float], float]]]:
    """
    Predict using either unigram or n-gram model based on the value of n
    :param passage: The text passage to evaluate
    :param sampled_passages: A list of sample passages for training
    :param n: The n-gram size; 1 for unigram, >1 for n-gram
    :param word2vec_model_path: Optional path to a pre-trained Word2Vec model
    :return: Evaluation metrics for the passage
    """
    if n == 1:
        model = SemanticUnigramModel(word2vec_model_path=word2vec_model_path)
    else:
        model = SemanticNgramModel(n=n, word2vec_model_path=word2vec_model_path)

    # Add sampled passages and original passage to the model
    for sample in sampled_passages + [passage]:
        model.add(sample)

    # Train the model
    model.train()

    # Evaluate the passage
    sentences = [sent.text.strip() for sent in model.nlp(passage).sents]
    results = model.evaluate(sentences)
    return results

'''def semantic_model_predict(passage: str, sampled_passages: List[str], n: int) -> float:
    """
    Predicts a single normalized score (0 to 1) for a given passage compared to sampled passages.

    :param passage: The main passage to evaluate.
    :param sampled_passages: A list of sampled passages to build the semantic model.
    :param n: The n-gram size for the model (1 for unigram, >1 for n-gram).
    :return: A normalized score between 0 and 1 representing passage-level semantic coherence.
    """
    # Initialize the appropriate semantic model
    if n == 1:
        model = SemanticUnigramModel()
    else:
        model = SemanticNgramModel(n=n)

    # Add sampled passages and the main passage to the model
    for sample in sampled_passages + [passage]:
        model.add(sample)

    # Train the model
    model.train()

    # Evaluate the main passage
    sentences = [sent.text.strip() for sent in model.nlp(passage).sents]
    evaluation_results = model.evaluate(sentences)

    # Extract document-level metrics
    avg_neg_logprob = evaluation_results['doc_level']['avg_neg_logprob']  # Mean of log probabilities
    avg_max_neg_logprob = evaluation_results['doc_level']['avg_max_neg_logprob']  # Mean of max log probabilities

    # Normalize the scores to the range [0, 1]
    # Using the formula: normalized_score = e^(-avg_neg_logprob), where high probabilities lead to scores close to 1
    #normalized_score = np.exp(-avg_neg_logprob)  # Transform into a coherence score
    # Inverse log scaling
    #normalized_score = 1 / (1 + avg_neg_logprob)


    # Optionally include avg_max_neg_logprob in a weighted average for a composite score
    # normalized_score = (normalized_score + np.exp(-avg_max_neg_logprob)) / 2

    return normalized_score'''


