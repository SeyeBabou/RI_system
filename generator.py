import torch
from transformers import pipeline

# Model ID
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Initialize the pipeline with the model
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Define the retrieved context and question
retrieved_context = (
    "The capital of France is Paris. "
    "Paris is known for its iconic landmarks like the Eiffel Tower."
)
question = "Where is the Eiffel Tower?"

# Combine the context and question in the messages
messages = [
    {"role": "system", "content": f"Context: {retrieved_context}"},
    {"role": "user", "content": question},
]

# Generate a response
outputs = pipe(
    messages,
    max_new_tokens=256,  # Maximum tokens for the response
    num_beams=5,  # Beam search for coherent output
    no_repeat_ngram_size=2,  # Avoid repetitions
    early_stopping=True,  # Stop generation when complete
)

# Print the generated text
print(outputs[0]["generated_text"])
