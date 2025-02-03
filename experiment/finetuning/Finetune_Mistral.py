
import logging
import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
from transformers.data.data_collator import DataCollatorWithPadding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set configurations
torch_dtype = torch.float16
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
output_dir = "./fine_tuned_mistral7B_mnli"
device_map = "auto"

# QLoRA Config for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


# Set padding token for the tokenizer if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use `eos_token` as `pad_token`
logger.info(f"Tokenizer padding token set to: {tokenizer.pad_token}")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    num_labels=3,
    device_map=device_map,
)

# Explicitly set the pad_token_id in the model configuration
model.config.pad_token_id = tokenizer.pad_token_id

# Resize token embeddings if necessary
if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))


logger.info("Base model loaded successfully.")

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj", "v_proj"],  # Adapted for Mistral
)
model = get_peft_model(model, peft_config)
logger.info("LoRA adapters added successfully.")

# Dataset loading and preprocessing
logger.info("Loading MNLI dataset...")
dataset = load_dataset("nyu-mll/multi_nli")
dataset = dataset.filter(lambda x: x['label'] != -1)

# Tokenization and preprocessing
logger.info("Tokenizing dataset...")

label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

def preprocess_function(examples):
    tokenized_examples = tokenizer(
        examples["premise"], examples["hypothesis"], 
        truncation=True, max_length=256, padding="max_length"  # Ensure proper padding
    )
    tokenized_examples["label"] = examples["label"]  # Keep numerical labels for training
    tokenized_examples["label_text"] = [label_map[label] for label in examples["label"]]  # Add class names for context
    return tokenized_examples

encoded_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)

# Training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=100,
    report_to="tensorboard",
    bf16=torch.cuda.is_available(),
    group_by_length=True,
    optim="paged_adamw_8bit",
)

# Initialize SFTTrainer
logger.info("Initializing Trainer...")
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation_matched"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start Training
logger.info("Starting model training...")
model.config.use_cache = False  # Disable caching during training
trainer.train()

# Save the model and tokenizer
logger.info("Saving fine-tuned model...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Clean up resources
logger.info("Cleaning up resources...")
model.to('cpu')
del model
del trainer
gc.collect()
torch.cuda.empty_cache()
logger.info("Training complete and resources cleared.")

