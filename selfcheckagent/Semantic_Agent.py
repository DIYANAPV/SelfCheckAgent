import spacy
import numpy as np
from nltk.util import ngrams
from typing import Dict, List, Set, Tuple, Union
from scipy.spatial.distance import cosine
import gensim.downloader as api  # Use Gensim's downloader


class SemanticLanguageModel:
    word2vec = None  # Class-level attribute

    def __init__(self, lowercase: bool = True, similarity_threshold: float = 0.9) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.token_count = 0
        self.counts = {'<unk>': 0}
        self.lowercase = lowercase
        self.similarity_threshold = similarity_threshold

        # Load the Word2Vec model only if it's not already loaded
        if SemanticLanguageModel.word2vec is None:
            print("Downloading Word2Vec model via Gensim...")
            SemanticLanguageModel.word2vec = api.load("word2vec-google-news-300")
            print("Word2Vec model loaded successfully.")
        self.word2vec = SemanticLanguageModel.word2vec
        self.token_vectors = {}

    def _get_vector(self, token: str) -> np.ndarray:
        if token not in self.token_vectors:
            if self.word2vec and token in self.word2vec:
                self.token_vectors[token] = self.word2vec[token]
            else:
                self.token_vectors[token] = np.zeros(300)  # Default to zero vector
        return self.token_vectors[token]



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
        min_neg_logprob = []  # Add this for minimum negative log probabilities
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
        
            # Add calculations for min and max negative log probabilities at sentence level
            avg_neg_logprob.append(-1.0 * np.mean(logprob_sent))
            max_neg_logprob.append(-1.0 * np.min(logprob_sent))
            min_neg_logprob.append(-1.0 * np.max(logprob_sent))  # Opposite of max for log probabilities

        avg_neg_logprob_doc = -1.0 * np.mean(logprob_doc)
        avg_max_neg_logprob_doc = np.mean(max_neg_logprob)
        avg_min_neg_logprob_doc = np.mean(min_neg_logprob)  # Document-level minimum negative log probability

        return {
            'sent_level': {
                'avg_neg_logprob': avg_neg_logprob,
                'max_neg_logprob': max_neg_logprob,
                'min_neg_logprob': min_neg_logprob,  # Include in results
            },
            'doc_level': {
                'avg_neg_logprob': avg_neg_logprob_doc,
                'avg_max_neg_logprob': avg_max_neg_logprob_doc,
                'avg_min_neg_logprob': avg_min_neg_logprob_doc,  # Include in results
            },
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
        

    
def semantic_model_predict(passage: str, sampled_passages: List[str], n: int) -> float:
    if n == 1:
        model = SemanticUnigramModel()
    else:
        model = SemanticNgramModel(n=n)

    for sample in sampled_passages + [passage]:
        model.add(sample)

    model.train()
    sentences = [sent.text.strip() for sent in model.nlp(passage).sents]
    results = model.evaluate(sentences)

    # Get sentence-level min and max values for normalization
    all_avg_neg_logprobs = results['sent_level']['avg_neg_logprob']
    all_max_neg_logprobs = results['sent_level']['max_neg_logprob']
    all_min_neg_logprobs = results['sent_level']['min_neg_logprob']

    neg_logprob_min = min(all_min_neg_logprobs)
    neg_logprob_max = max(all_max_neg_logprobs)

    def normalize(value):
        return (value - neg_logprob_min) / (neg_logprob_max - neg_logprob_min)

    # Normalize sentence-level scores
    normalized_avg_scores = [normalize(score) for score in all_avg_neg_logprobs]

    # Compute document-level average hallucination score
    doc_avg_hallucination_score = sum(normalized_avg_scores) / len(normalized_avg_scores)

    # Return only the document-level average hallucination score
    return doc_avg_hallucination_score



"""def semantic_model_predict(passage: str, sampled_passages: List[str], n: int) -> Dict[str, Dict[str, Union[List[float], float]]]:
    if n == 1:
        model = SemanticUnigramModel()
    else:
        model = SemanticNgramModel(n=n)

    for sample in sampled_passages + [passage]:
        model.add(sample)

    model.train()
    sentences = [sent.text.strip() for sent in model.nlp(passage).sents]
    results = model.evaluate(sentences)
    return results"""

