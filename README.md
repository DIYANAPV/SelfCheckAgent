# SelfCheckAgent


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-GPT--4-blue)](https://platform.openai.com/)

SelfCheckAgent is a tool to detect hallucination of LLM outputs with zero external resource, by leveraging consistency based approach.

## Installation

To get started, clone the repository and install the dependencies:

```bash
git clone https://github.com/DIYANAPV/SelfCheckAgent.git
cd SelfCheckAgent
pip install .

```



```plaintext
SelfCheckAgent/
│
├── selfcheckagent/               
│   ├── __init__.py                # Initialize the package
|   ├── symbolic_agent.py
|   ├── specialized_agent.py
│   ├── contextual_agent.py        
├── example.ipynb                 # Example notebook for usage
├── README.md                     # Documentation
├── requirements.txt              # Dependencies
└── setup.py                      # Packaging script


If you use OpenAI's GPT models (e.g., GPT-4), set your API key:


import openai
openai.api_key = "your_openai_api_key"
```



```


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for major changes.

## License

This project is licensed under the MIT License.

## 📄 Citation

```bash

@article{rabby2024mc,
  title={-},
  author={Rabby, Gollam and Keya, Farhana and Zamil, Parvez and Auer, S{\"o}ren},
  journal={-},
  year={2024}
}

```

## References







Happy coding! 🚀
