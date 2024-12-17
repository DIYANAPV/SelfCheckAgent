import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class ContextualAgent:
    """
    ContextualAgent (LLM Prompt): Checking LLM's text against its own sampled texts via open-source LLM prompting
    """
    def __init__(
        self,
        model: str = None,
        device=None
    ):
        model = model if model is not None else LLMPromptConfig.model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        self.prompt_template = (
            "Context: {context}\n\nSentence: {sentence}\n\n"
            "Is the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        )
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"ContextualAgent ({model}) initialized to device {device}")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence) and returns sent-level scores.

        :param sentences: list[str] -- Sentences to be evaluated, e.g., GPT text responses split by Spacy.
        :param sampled_passages: list[str] -- Stochastically generated responses (without sentence splitting).
        :param verbose: bool -- If True, tqdm progress bar will be shown.
        :return sent_scores: Sentence-level scores.
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                # This improves performance when using the simple prompt template
                sample = sample.replace("\n", " ")

                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    do_sample=False,  # HF's default for Llama2 is True
                )
                output_text = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                generate_text = output_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(self, text):
        """
        To map from generated text to score:
        Yes -> 0.0
        No  -> 1.0
        Everything else -> 0.5
        """
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]

