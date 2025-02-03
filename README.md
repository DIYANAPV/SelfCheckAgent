# SelfCheckAgent


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-GPT--4-blue)](https://platform.openai.com/)

SelfCheckAgent is a tool to detect hallucination of LLM outputs with zero external resource, by leveraging consistency based approach.



## Installation

To get started, clone the repository and install the dependencies:

git clone https://github.com/DIYANAPV/SelfCheckAgent.git
cd SelfCheckAgent
!pip install -r requirements.txt

```plaintext
SelfCheckAgent/
│
├── selfcheckagent/               
│   ├── __init__.py                # Initialize the package
|   ├── symbolic_agent.py
|   ├── specialized_agent.py
│   ├── contextual_agent.py        
├── usage.ipynb                 # Example notebook for usage
├── README.md                     # Documentation
├── requirements.txt              # Dependencies
└── setup.py                      # Packaging script


If you use OpenAI's GPT models (e.g., GPT-4), set your API key:


import openai
openai.api_key = "your_openai_api_key"
```
  


### Experimental Results
Results on the **wiki_bio_gpt3_hallucination** dataset.

| Method                                          | NonFact (AUC-PR) | Factual (AUC-PR) | Ranking (PCC) |
|------------------------------------------------|:----------------:|:----------------:|:-------------:|
| SelfCheck-Unigram (baseline)   | 85.63            | 58.47            | 64.71         |
| Semantic Agent-Unigram         | 86.97            | 59.02            | 65.88         |
| Phi-3-mini-4k-instruct Pre-train Model (ZS)    | 89.27            | 56.26            | 61.88         |
| Llama 3.1 8b Pre-train Model (ZS)              | 78.90            | 37.12            | 34.25         |
| Mistral 7B Pre-train Model (ZS)               | 86.13            | 58.60            | 41.15         |
| Gemma-7b-aps-it Pre-train Model (ZS)           | 75.41            | 30.87            | 4.80          |
| T5 Pre-train Model (ZS)                       | 70.04            | 25.55            | -9.78         |
| gpt2 Pre-train Model (ZS)                     | 71.67            | 29.26            | 0.68          |
| Roberta-large (baseline) Pre-train Model (ZS)  | 76.44            | 31.20            | 6.07          |
| Phi-3-mini-4k-instruct Pre-train Model (CoT)         | 75.08            | 29.67            | 13.79         |
| Llama 3.1 8b Pre-train Model (CoT)                  | 63.88            | 21.75            | -17.11        |
| Mistral 7B Pre-train Model (CoT)                    | 81.91            | 46.10            | 43.49         |
| Gemma-7b-aps-it Pre-train Model (CoT)              | 70.97            | 25.99            | -7.62         |
| T5 Pre-train Model (CoT)                             | 69.97            | 24.79            | -10.95        |
| gpt2 Pre-train Model (CoT)                          | 72.01            | 30.59            | 3.55          |
| Roberta-large (baseline) Pre-train Model (CoT)     | 73.86            | 31.47            | 13.02         |
| Phi-3-mini-4k-instruct fine-tuned Specialized Detection Agent       | 92.87            | 65.25            | 73.54         |
| Llama 3.1 8b fine-tuned Specialized Detection Agent                | 76.85            | 29.71            | 8.91          |
| Mistral 7B fine-tuned Specialized Detection Agent                  | 92.68            | 67.10            | 75.63         |
| Gemma 7B fine-tuned Specialized Detection Agent                    | 83.47            | 43.12            | 50.98         |
| SelfCheck-NLI (baseline)       | 92.50            | 66.08            | 74.14         |
| Mistral 7B (CoT) Contextual Consistency Agent                        | 91.74            | 64.01            | 75.40         |
| Llama2-7B-chat (CoT) Contextual Consistency Agent                   | 81.92            | 33.37            | 12.62         |
| Llama3.1-8B-instruct (CoT) Contextual Consistency Agent              | 93.64            | 70.26            | 78.48         |
| Llama2-13B-chat (CoT) Contextual Consistency Agent                   | 87.85            | 50.09            | 53.24         |
| gpt-3.5-turbo (CoT) Contextual Consistency Agent                    | 90.59            | 62.11            | 72.17         |
| gpt-4o-mini (CoT) Contextual Consistency Agent                       | 94.14            | 74.95            | 76.33         |
| gpt-4o-mini (ZS) Contextual Consistency Agent                | 94.00            | 74.11            | 77.48         |
| Mistral 7B - Baseline (ZS) Contextual Consistency Agent         | 91.31            | 62.76            | 74.46         |
| Llama2-7B-chat - Baseline (ZS) Contextual Consistency Agent    | 89.05            | 63.06            | 61.52         |
| Llama2-13B-chat- Baseline (ZS) Contextual Consistency Agent    | 91.91            | 64.34            | 75.44         |
| Llama3-8B-chat (ZS) Contextual Consistency Agent            | 92.85            | 70.73            | 76.54         |
| gpt-3.5-turbo- Baseline (ZS) Contextual Consistency Agent    | 93.42            | 67.09            | 78.32         |




**On AIME dataset**

| Method                                          | NonFact (AUC-PR) | Factual (AUC-PR) | Ranking (PCC) |
|------------------------------------------------|:----------------:|:----------------:|:-------------:|
| SelfCheck-Unigram (baseline)                   | 85.69            | 14.91            | -1.25         |
| Semantic Agent-Unigram             | 87.24            | 14.62            | 11.94         |
| Phi-3-mini-4k-instruct Pre-train Model (ZS)    | 90.34            | 19.16            | 12.75         |
| Llama 3.1 8b Pre-train Model (ZS)             | 85.33            | 13.13            | -4.07         |
| Mistral 7B Pre-train Model (ZS)                | 86.42            | 14.82            | -0.91         |
| Gemma-7b-aps-it Pre-train Model (ZS)          | 85.72            | 12.85            | -2.74         |
| gpt2 Pre-train Model (ZS)                     | 85.92            | 13.19            | -2.22         |
| Phi-3-mini-4k-instruct Pre-train Model (CoT)        | 87.76            | 21.10            | 3.16          |
| Llama 3.1 8b Pre-train Model (CoT)                 | 85.33            | 13.13            | -4.07         |
| Mistral 7B Pre-train Model (CoT)                     | 91.33            | 17.11            | 2.57          |
| Gemma-7b-aps-it Pre-train Model (CoT)              | 90.07            | 13.88            | 0.50          |
| gpt2 Pre-train Model (CoT)                           | 86.58            | 14.25            | 1.65          |
| Phi-3-mini-4k-instruct Specialized Detection Agent       | 93.38            | 20.38            | 21.76         |
| Llama 3.1 8b Specialized Detection Agent                | 79.63            | 9.74             | -15.44        |
| Mistral 7B Specialized Detection Agent                    | 92.91            | 17.37            | 20.05         |
| Gemma-7b-aps-it Specialized Detection Agent              | 82.58            | 10.70            | -16.77        |
| SelfCheck-NLI (baseline)      | 87.64            | 17.45            | 5.32          |
| Mistral 7B (CoT) Contextual Consistency Agent   | 93.63            | 37.98            | 22.51         |
| Llama2-7B-chat (CoT) Contextual Consistency Agent      | 93.15            | 56.88            | 1.71          |
| Llama3-8B-chat (CoT) Contextual Consistency Agent | 91.77            | 24.89            | 14.23         |
| Llama2-13B-chat (CoT) Contextual Consistency Agent | 92.40            | 56.09            | 10.80         |
| gpt-3.5-turbo (CoT) Contextual Consistency Agent  | 95.28            | 57.62            | 25.42         |
| gpt-4o-mini (CoT) Contextual Consistency Agent        | 94.89            | 30.58            | 30.68         |
| gpt-4o-mini (ZS) Contextual Consistency Agent                 | 93.93            | 24.26            | 21.13         |
| Mistral 7B - Baseline (ZS) Contextual Consistency Agent         | 92.15            | 45.31            | 17.19         |
| Llama2-7B-chat - Baseline (ZS) Contextual Consistency Agent    | 92.99            | 49.39            | 18.51         |
| Llama3.1-8B-chat (ZS) Contextual Consistency Agent             | 94.79            | 37.75            | 29.67         |
| Llama2-13B-chat- Baseline (ZS) Contextual Consistency Agent    | 93.18            | 56.90            | 2.89          |
| gpt-3.5-turbo- Baseline (ZS) Contextual Consistency Agent      | 95.66            | 56.95            | 30.08         |


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for major changes.

## License

This project is licensed under the MIT License.
