import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

class SelfCheckNLI:
    """
    SelfCheckGPT (NLI variant): Supports multiple fine-tuned models for contextual consistency.
    """
    def __init__(self, model_name: str = "phi3_nli", device: torch.device = None):
        """
        Initialize the SelfCheckNLI with the selected fine-tuned model.
        
        :param model_name: Name of the fine-tuned model directory (e.g., 'phi3_nli' or 'other_model').
        :param device: Device on which the model should run.
        """
        # Path to the model directory
        model_path = f"./selfcheckagent/models/{model_name}"
        
        # Load tokenizer and model dynamically
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3, ignore_mismatched_sizes=True
        )
        self.model.eval()

        # Select device (CPU or GPU)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)

        print(f"SelfCheck-NLI initialized with model: {model_name} on device: {device}")

    @torch.no_grad()
    def predict(self, sentences: List[str], sampled_passages: List[str]):
        """
        Compare sentences against sampled passages using the loaded model.
        
        :param sentences: List of sentences to evaluate.
        :param sampled_passages: List of sampled passages for comparison.
        :return: Average contradiction scores per sentence and detailed scores.
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)

        scores = np.zeros((num_sentences, num_samples))

        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(sampled_passages):
                inputs = self.tokenizer.encode_plus(
                    sample, sentence, add_special_tokens=True, padding="max_length",
                    truncation=True, max_length=2048, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                logits = self.model(**inputs).logits
                logits_entailment = logits[0][0].item()  # Entailment logit
                logits_contradiction = logits[0][2].item()  # Contradiction logit

                prob_contradiction = (
                    torch.exp(torch.tensor(logits_contradiction)) /
                    (torch.exp(torch.tensor(logits_contradiction)) + torch.exp(torch.tensor(logits_entailment)))
                ).item()

                scores[sent_i, sample_i] = prob_contradiction

        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence, scores


