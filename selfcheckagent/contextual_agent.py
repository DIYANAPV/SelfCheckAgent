import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

class SelfCheckNLI:
    """
    SelfCheckGPT (NLI variant): Checking LLM's text against its own sampled texts via Phi-3 fine-tuned to Multi-NLI
    """
    def __init__(self, nli_model: str = "./fine_tuned_phi3_mnli", device: torch.device = None):
        """
        Initialize the SelfCheckNLI class with the fine-tuned Phi-3 model and tokenizer.
        
        :param nli_model: Path to the fine-tuned Phi-3 model.
        :param device: The device on which the model should run (CPU or GPU). If None, it auto-selects.
        """
        # Load the fine-tuned Phi-3 model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            nli_model,
            num_labels=3,  # The number of labels for Multi-NLI (entailment, neutral, contradiction)
            ignore_mismatched_sizes=True,  # For any size mismatches between the model and the pre-trained weights
        )
        self.model.eval()

        # Set the device (use GPU if available, otherwise fallback to CPU)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)

        print(f"SelfCheck-NLI initialized to device: {device}")

    @torch.no_grad()
    def predict(self, sentences: List[str], sampled_passages: List[str]):
        """
        Compares each sentence (hypothesis) with every sampled passage (premise).
        Calculates the probability of contradiction for each pair.
        Returns the mean contradiction probability (sentence score) for each sentence.
        
        :param sentences: List of sentences to be evaluated.
        :param sampled_passages: List of sampled passages (evidence) for comparison.
        :return: 
            - scores_per_sentence: Mean contradiction probability for each sentence.
            - scores: Individual contradiction probabilities for each hypothesis-sampled passage pair.
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)

        # Store scores for each hypothesis-sampled passage pair
        scores = np.zeros((num_sentences, num_samples))

        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(sampled_passages):
                # Tokenize the input pairs (sampled passage as premise, sentence as hypothesis)
                inputs = self.tokenizer.encode_plus(
                    sample,  # Premise: sampled passage
                    sentence,  # Hypothesis: sentence
                    add_special_tokens=True,
                    padding="max_length",
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get logits from the model (output for entailment, neutral, contradiction)
                logits = self.model(**inputs).logits
                logits_entailment = logits[0][0].item()  # Logit for entailment
                logits_contradiction = logits[0][2].item()  # Logit for contradiction

                # Calculate probability of contradiction
                prob_contradiction = (
                    torch.exp(torch.tensor(logits_contradiction)) /
                    (torch.exp(torch.tensor(logits_contradiction)) + torch.exp(torch.tensor(logits_entailment)))
                ).item()

                # Store the probability of contradiction for this pair
                scores[sent_i, sample_i] = prob_contradiction

        # Average contradiction probability per sentence
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence, scores

