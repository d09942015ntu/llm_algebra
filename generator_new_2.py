import torch
import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from mymodels.toytrans.modeling_toytrans import ToyTransLMHeadModel
from mymodels.toytrans.configuration_toytrans import ToyTransConfig
from transformers import PreTrainedTokenizerFast
# Load pre-trained model and tokenizer
#model_name = 'gpt2'  # You can choose 'gpt2-medium', 'gpt2-large', 'gpt2-xl' if needed

#checkpoint_path = "/Users/markchang/code/RLSTaR/checkpoints/checkpoint-1000"


path_tokenizer= './models/gpt2'
token_dir= './data/ide_41_5_1'
tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)
tokens = json.load(open(os.path.join(token_dir, "tokens.json"), "r"))
special_tokens_dict = {'additional_special_tokens': tokens}
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens(special_tokens_dict)



model_savepath = "./results_backup/20241203_084140-ide_41_5_9/checkpoints/checkpoint-39000/"
model = ToyTransLMHeadModel.from_pretrained(model_savepath)
model.resize_token_embeddings_by_tokenizer(tokenizer,reinitialize=False)
model.debug = True
#model.save_pretrained("./mymodels/toytrains_out")


# Function to generate a response from the model
def generate_conversation(prompt, max_length=9, num_return_sequences=1):
    # Encode the input prompt
    #input_ids = tokenizer.encode(prompt, return_tensors='pt')

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    input_ids = input_ids.to(torch.int32)
    # tensor([[31373,   703,   389,   345,    30]], device='mps:0')
    # tensor([[31373,   703,   389,   345,    30]])
    print(f"prompt:{prompt}")
    print(f"input_ids:{input_ids}")
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
    print(f"output:{output}")

    # Decode and return the generated text
    print(f"result:{[tokenizer.decode(output_seq) for output_seq in output]}")

# Example usage
generate_conversation("[1][+][1][+][1][+][1][=]")
generate_conversation("[1][+][1][+][1][+][2][=]")
generate_conversation("[1][+][1][+][2][+][2][=]")
generate_conversation("[1][+][2][+][2][+][2][=]")
generate_conversation("[2][+][2][+][2][+][1][=]")
generate_conversation("[4][+][4][+][1][+][1][=]")
generate_conversation("[4][+][4][=]")
generate_conversation("[4][+][4][=]")

