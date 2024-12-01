import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import json

# Load pre-trained model and tokenizer


path_tokenizer= './models/gpt2'
path_model= './mymodels/toytrans'
tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)
tokens = json.load(open(os.path.join(path_model, "tokens.json"), "r"))
special_tokens_dict = {'additional_special_tokens': tokens}
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens(special_tokens_dict)

model = GPT2LMHeadModel.from_pretrained(path_model)

# Function to generate a response from the model
def generate_conversation(prompt, max_length=100, num_return_sequences=1):
    # Encode the input prompt
    #input_ids = tokenizer.encode(prompt, return_tensors='pt')

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # tensor([[31373,   703,   389,   345,    30]], device='mps:0')
    # tensor([[31373,   703,   389,   345,    30]])
    print(input_ids)
    #tensor([[4304, 7600, 5125, 2124, 1875]])

    #tensor([[4304, 7600, 5125, 2124, 1875, 220],
    #        [4761, 2808, 10190, 2124, 1875, 220]])
    #tensor([[4304, 7600, 5125, 2124, 1875, 220],
    #        [4761, 2808, 10190, 2124, 1875, 220]])
    # Generate text continuation
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            top_p=0.95,
            temperature=0.7
        )

    # Decode and return the generated text
    return [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in output]

# Example usage
prompt = "76 85 49 x >"
responses = generate_conversation(prompt)

for idx, response in enumerate(responses):
    print(f"Conversation {idx + 1}:\n{response}\n")
