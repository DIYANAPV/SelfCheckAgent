import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List




class SelfCheckNLI:
    """
    SelfCheckNLI: Supports multiple NLI models hosted on Hugging Face.
    Allows dynamic switching between models and customization of input settings.
    """

    # Predefined model options
    MODEL_OPTIONS = [
        "diyanamuhammed/mistral-mnli",
        "diyanamuhammed/gemma-mnli",
        "diyanamuhammed/Phi-3-MNLI",
        "diyanamuhammed/llama-mnli"
    ]

    def __init__(self, model_name: str = "diyanamuhammed/Phi-3-MNLI", device: str = None, max_length: int = 512):
        """
        Initialize the SelfCheckNLI with a selected Hugging Face NLI model.

        :param model_name: str -- Hugging Face model name (default: 'diyanamuhammed/Phi-3-MNLI').
        :param device: str -- Device to load the model on ('cuda' or 'cpu').
        :param max_length: int -- Maximum sequence length for tokenization (default: 512).
        """
        if model_name not in self.MODEL_OPTIONS:
            raise ValueError(f"Invalid model name '{model_name}'. Choose from {self.MODEL_OPTIONS}.")

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Load tokenizer and model from Hugging Face
        print(f"Loading model '{model_name}' from Hugging Face...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3, ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"SelfCheckNLI initialized with model '{model_name}' on device '{self.device}'.")

    @torch.no_grad()
    def predict(self, sentences: List[str], sampled_passages: List[str]):
        """
        Compare sentences against sampled passages using the loaded model.

        :param sentences: List[str] -- Sentences to evaluate.
        :param sampled_passages: List[str] -- Passages for comparison.
        :return: Average contradiction scores per sentence and detailed scores.
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))

        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(sampled_passages):
                # Prepare input using tokenizer
                inputs = self.tokenizer.encode_plus(
                    sample, sentence,
                    add_special_tokens=True,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get model predictions (logits)
                logits = self.model(**inputs).logits
                logits_entailment = logits[0][0].item()  # Entailment logit
                logits_contradiction = logits[0][2].item()  # Contradiction logit

                # Compute probability of contradiction
                prob_contradiction = (
                    torch.exp(torch.tensor(logits_contradiction)) /
                    (torch.exp(torch.tensor(logits_contradiction)) + torch.exp(torch.tensor(logits_entailment)))
                ).item()

                scores[sent_i, sample_i] = prob_contradiction

        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence
