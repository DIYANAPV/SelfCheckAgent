import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class ContextualAgent:
    """
    ContextualAgent: Allows switching between LLM models such as Llama 2, Llama 3, Mistral, and GPT-based models.
    """
    MODEL_OPTIONS = {
        "llama2_7b": "meta-llama/Llama-2-7b-hf",
        "llama2_13b": "meta-llama/Llama-2-13b-hf",
        "llama3_8b": "meta-llama/Llama-3.1-8b-hf",
        "mistral_7b": "mistralai/Mistral-7B-v0.1",
        "gpt_3.5": "gpt-3.5-turbo",  # Requires OpenAI API
        "gpt_4": "gpt-4",            # Requires OpenAI API
    }

    def __init__(self, model_name: str = "llama2_7b", device=None):
        """
        Initialize the ContextualAgent with a selected model.
        
        :param model_name: str -- Name of the model to use. Options: llama2_7b, llama2_13b, llama3_8b, mistral_7b, gpt_3.5, gpt_4
        :param device: str -- Device to load the model on, e.g., "cuda" or "cpu"
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.api_mode = model_name in ["gpt_3.5", "gpt_4"]  # Check if OpenAI API is needed
        if not self.api_mode:
            self._load_model()
        else:
            print(f"Using OpenAI API for model: {model_name}")
        self.prompt_template = (
            "Context: {context}\n\nSentence: {sentence}\n\n"
            "Is the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        )
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"ContextualAgent ({model_name}) initialized to device {self.device}")

    def _load_model(self):
        """
        Load the tokenizer and model dynamically based on the chosen model name.
        """
        if self.model_name not in self.MODEL_OPTIONS:
            raise ValueError(f"Model '{self.model_name}' not supported. Choose from {list(self.MODEL_OPTIONS.keys())}")
        
        model_path = self.MODEL_OPTIONS[self.model_name]
        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=self.device)
        self.model.eval()

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(self, sentences: List[str], sampled_passages: List[str], verbose: bool = False):
        """
        Predict sentence-level scores based on model outputs.
        
        :param sentences: list[str] -- Sentences to evaluate.
        :param sampled_passages: list[str] -- Sampled passages as context.
        :param verbose: bool -- Whether to display the tqdm progress bar.
        """
        if self.api_mode:
            return self._predict_with_openai(sentences, sampled_passages, verbose)
        
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                sample = sample.replace("\n", " ")
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False)
                output_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
                generate_text = output_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        return scores.mean(axis=-1)

    def _predict_with_openai(self, sentences: List[str], sampled_passages: List[str], verbose: bool = False):
        """
        Handle prediction using OpenAI API (for GPT-3.5, GPT-4).
        """
        import openai
        scores = []
        for sentence in tqdm(sentences, disable=not verbose):
            sentence_scores = []
            for sample in sampled_passages:
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                response = openai.ChatCompletion.create(
                    model=self.MODEL_OPTIONS[self.model_name],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5
                )
                output_text = response['choices'][0]['message']['content']
                score_ = self.text_postprocessing(output_text)
                sentence_scores.append(score_)
            scores.append(np.mean(sentence_scores))
        return scores

    def text_postprocessing(self, text):
        """
        Map model output to a score.
        """
        text = text.lower().strip()
        if text[:3] == 'yes':
            return self.text_mapping['yes']
        elif text[:2] == 'no':
            return self.text_mapping['no']
        else:
            if text not in self.not_defined_text:
                print(f"Warning: {text} not defined")
                self.not_defined_text.add(text)
            return self.text_mapping['n/a']
