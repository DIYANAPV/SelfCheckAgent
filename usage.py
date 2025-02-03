
#Specialized Detection Agent

'''
Model
Switch between different models using the model_name parameter:


nli_agent = SelfCheckNLI(model_name="hiddennode/mistral-mnli")

MODEL_OPTIONS = [
"hiddennode/mistral-mnli",

"hiddennode/Phi-3-MNLI",

"hiddennode/gemma-mnli",

"hiddennode/llama-mnli"

]



Device
Force the model to use a specific device:

nli_agent = SelfCheckNLI(device="cpu")  # Run on CPU


Max Length
Adjust the maximum tokenization length to optimize for speed or context size:


nli_agent = SelfCheckNLI(max_length=128)
'''


@By default
from selfcheckagent.specialized_agent import SelfCheckNLI

# Initialize with default parameters
nli_agent = SelfCheckNLI()

# Define test sentences and passages
sentences = ["The cat is on the mat."]
sampled_passages = ["The cat sleeps on the mat."]

# Get predictions
scores_per_sentence = nli_agent.predict(sentences, sampled_passages)

# Output the results
print("Average Contradiction Scores per Sentence:", scores_per_sentence)



from selfcheckagent.specialized_agent import SelfCheckNLI

# Define test sentences and passages
sentences = ["The cat is on the mat.", "The dog is barking loudly."]
sampled_passages = [
    "The cat sleeps on the mat.",
    "A dog barks in the distance."
]

# Initialize the SelfCheckNLI with a specified model
nli_agent = SelfCheckNLI(
    model_name="-",  # Choose the model
    device="cuda",                          # Use GPU if available, fallback to CPU
    max_length=256                          # Specify maximum sequence length
)

# Get predictions
scores_per_sentence = nli_agent.predict(sentences, sampled_passages)

# Output the results
print("Average Contradiction Scores per Sentence:", scores_per_sentence)




@Contextual Consistency Agent

# you can choose any one of the model among this options. 
'''MODEL_OPTIONS = {
    "llama2_7b": "meta-llama/Llama-2-7b-hf",
    "llama2_13b": "meta-llama/Llama-2-13b-hf",
    "llama3_8b": "meta-llama/Llama-3.1-8b-hf",
    "mistral_7b": "mistralai/Mistral-7B-v0.1",
    "gpt_3.5": "gpt-3.5-turbo",
    "gpt_4": "gpt-4",
}'''


from selfcheckagent.contextual_agent import ContextualAgent

# Set up the OpenAI API key
import openai
openai.api_key = "your_openai_api_key"  # Replace with your OpenAI API key

# Initialize the ContextualAgent with GPT-4
agent = ContextualAgent(model_name="gpt_4")

# Define context and sentences
sampled_passages = ["The cat is sleeping on the mat."]
sentences = ["The cat is on the mat."]

# Make predictions
scores = agent.predict(sentences, sampled_passages, verbose=True)

# Output the results
print("Hallucination score:", scores)
