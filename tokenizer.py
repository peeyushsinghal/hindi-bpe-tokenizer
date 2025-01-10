import yaml
import regex as re
from tqdm import tqdm
import gc
import json

        
def load_config(config_file_path: str = "config.yml"):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_input_text(config: dict) -> str:
    with open(config["input_file_info"]["file_path"], 'r', encoding='utf-8') as _f:
        hi_text = [line.strip() for line in _f.readlines()]

    hi_text_abridged = hi_text[:int(config["input_file_info"]["input_file_limit"])]
    hi_text_abridged = '\n'.join(hi_text_abridged)

    if config["input_file_info"]["print_text"]:
        print(" Sample text: ", hi_text_abridged[:10])

    return hi_text_abridged

def get_stats(ids, counts= None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def stoi(text: str, config: dict) -> list:
    # tokenize the text
    if config["regex_string"] and len(config["regex_string"]) > 0:
        print("Using regex string: ", config["regex_string"])
        tokens = re.findall(config["regex_string"], text)
        # Convert tokens to bytes and then to integers
        return [b for token in tokens for b in token.encode('utf-8')]
    else:
        print("Using default tokenizer")
        # Instead of splitting, we'll preserve spaces by encoding them directly
        return [b for ch in text for b in ch.encode('utf-8')]


def encode(text, merges, config: dict):
    """
    Encode text into tokens using the learned merges
    """
    ids = stoi(text, config)
    
    sorted_merges = sorted(merges.items(), key=lambda x: x[1])
    for (p1, p2), idx in sorted_merges:
        ids = merge(ids, (p1, p2), idx)
    
    return ids

def decode(ids, merges, config: dict):
    """
    Decode tokens back to text using the learned merges
    """
    # Create reverse mapping from token to pair
    reverse_merges = {idx: pair for pair, idx in merges.items()}
    
    # Expand all tokens recursively
    def expand_token(token):
        if token < 256:  # Base case: token is a byte
            return bytes([token])
        
        # Recursive case: expand the token into its constituent pair
        pair = reverse_merges[token]
        return expand_token(pair[0]) + expand_token(pair[1])
    
    # Expand all tokens and concatenate
    bytes_list = [expand_token(id) for id in ids]
    bytes_data = b''.join(bytes_list)
    
    # Convert bytes back to text
    try:
        return bytes_data.decode('utf-8')
    except UnicodeDecodeError:
        return "[DECODE_ERROR]"
    
class Tokenizer:
    def __init__(self, merges = None, config: dict = None):
        self.merges = merges or {}
        self.config = config
    
    def save(self, file_path):
        # Convert tuple keys to strings for JSON serialization
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_merges, f)
    
    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            serialized_merges = json.load(f)
            # Convert string keys back to tuples
            merges = {tuple(map(int, k.split(','))): v 
                          for k, v in serialized_merges.items()}
        
        return cls(merges)
    
    def encode(self, text):
        return encode(text, self.merges, self.config)
    
    def decode(self, ids):
        return decode(ids, self.merges, self.config)

def train_tokenizer(config: dict) -> None:    
    # get input text
    hi_text = get_input_text(config)

    # convert string to tokens   
    tokens = stoi(hi_text, config)
    initial_len = len(tokens)
    print("Tokens length (initial): ", initial_len, " tokens unique: ", len(set(tokens)))
    print("Example tokens: ", ord('क'), chr(2325), ord("।"), chr(2404))

    print("Training tokenizer....")
    num_merges = config["vocab_size"] - 256
    original_token = tokens
    
    merges ={}
    pbar = tqdm(range(num_merges), desc="Training tokenizer")
    output_file = config["output_file_info"]["file_path"]

    for i in pbar:
        # Get statistics of the tokens
        stats = get_stats(tokens)
        # Get the most frequent pair
        pair = max (stats, key=stats.get)
        # Get the index of the new token
        idx = 256 + i

        # Merge the pair
        tokens = merge(tokens, pair, idx)
        merges[pair] = idx


        # Show progress
        if (i + 1) % 100 == 0:
            current_ratio = initial_len / len(tokens)
            pbar.write(f"Iteration {i+1}: compression ratio: {current_ratio:.2f}X")
        
        # Garbage collection periodically
        if (i + 1) % 1000 == 0:
            gc.collect()
        
        # Save intermediate merges
        if (i + 1) % 1000 == 0:
            temp_tokenizer = Tokenizer(merges)
            temp_tokenizer.save(f"{output_file}.checkpoint")

    print("Training tokenizer completed")
    final_tokenizer = Tokenizer(merges)
    final_tokenizer.save(f"{output_file}")    

    print("\n=== Final Statistics ===")
    print(f"Vocabulary size: {config['vocab_size']}")
    print(f"Initial tokens: {initial_len:,}")
    print(f"Final tokens: {len(tokens):,}")
    print(f"Initial bytes: {initial_len * 4:,}")
    print(f"Final bytes: {len(tokens) * 4:,}")
    print(f"Token compression ratio: {initial_len / len(tokens):.2f}X")
    print(f"Byte compression ratio: {initial_len * 4 / len(tokens) * 4:.2f}X")
    print(f"Saved tokenizer to: {output_file}")

    return merges

def load_tokenizer(config: dict) -> Tokenizer:
    "load the tokenizer from the json file"
    with open(config["output_file_info"]["file_path"], 'r', encoding='utf-8') as f:
        serialized_merges = json.load(f)
    
    merges = {tuple(map(int, k.split(','))): v 
                          for k, v in serialized_merges.items()}
    
    return Tokenizer(merges, config)

if __name__ == "__main__":

    # TRAIN TOKENIZER
    config = load_config()
    merges = train_tokenizer(config)
    print("Merges: ", merges)

    # USE TOKENIZER
    # tokenizer = load_tokenizer(config)
    # test_text = config["test_text"]

    # print("Test text: ", test_text)
    # print("Encoded text: ", tokenizer.encode(test_text))
    # decoded = tokenizer.decode(tokenizer.encode(test_text))
    # print("Decoded text: ", decoded)

    # print(f"Successful roundtrip: {test_text == decoded}")

    

    
