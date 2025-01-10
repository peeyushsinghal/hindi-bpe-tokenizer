# Hindi BPE Tokenizer

This project implements a Byte Pair Encoding (BPE) tokenizer for Hindi text. The tokenizer is designed to efficiently encode and decode Hindi text while preserving spaces and providing detailed statistics about the tokenization process.

## Online Demo

You can also try the tokenizer online via [Hugging Face Spaces](https://huggingface.co/spaces/peeyushsinghal/hindi-tokenizer-bpe).

## Features

- Tokenizes Hindi text using a customizable regex pattern.
- Encodes and decodes text while preserving whitespace.
- Provides visual feedback on tokenization with highlighted tokens and tooltips showing token IDs.
- Displays statistics about the tokenization process, including total tokens, unique tokens, character count, and compression ratio.

## Data
hi.txt was used from https://github.com/AI4Bharat/indicnlp_corpus.

First 10000 lines were used

## Regex
```
regex_string: r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{N}+| ?(?:[\u0904-\u0939\u093d-\u093d\u0950-\u0950\u0958-\u0961\u0970-\u097f\ua8f2-\ua8fe\U00011b00-\U00011b09\u1cd3-\u1cd3\u1ce9-\u1cec\u1cee-\u1cf3\u1cf5-\u1cf6\u1cfa-\u1cfa][\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\ua8e0-\ua8f1\ua8ff-\ua8ff\u1cd0-\u1cd2\u1cd4-\u1ce8\u1ced-\u1ced\u1cf4-\u1cf4\u1cf7-\u1cf9]*)+| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```


## Requirements

- Python 3.6 or higher
- Gradio
- PyYAML
- regex
- tqdm

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/hindi-bpe-tokenizer.git
   cd hindi-bpe-tokenizer
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before running the application, ensure that the `config.yml` and `config_app.yml` files are correctly set up. The `config.yml` file should contain the path to your input text file and other configuration settings.

### Example `config.yml`
```yaml
input_file_info:
    file_path: "data/hi.txt"
    url: "https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt"
    language: "Hindi"
    input_file_limit: 10000 # number of lines to read
    print_text: true
regex_string: r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{N}+| ?(?:[\u0904-\u0939\u093d-\u093d\u0950-\u0950\u0958-\u0961\u0970-\u097f\ua8f2-\ua8fe\U00011b00-\U00011b09\u1cd3-\u1cd3\u1ce9-\u1cec\u1cee-\u1cf3\u1cf5-\u1cf6\u1cfa-\u1cfa][\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\ua8e0-\ua8f1\ua8ff-\ua8ff\u1cd0-\u1cd2\u1cd4-\u1ce8\u1ced-\u1ced\u1cf4-\u1cf4\u1cf7-\u1cf9])+| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
output_file_info:
    file_path: "model/hi_tokenizer_regex.json"
vocab_size: 5001
test_text: "यहां वर्तमान में 20 हजार पुस्तकें थी जो अभी रैन बसेरा परिसर के कक्ष में रखी हुई है।"
```

## Usage

1. To train the tokenizer, run the following command:

   ```bash
   python tokenizer.py
   ```

2. To launch the Gradio application, run:

   ```bash
   python app.py
   ```

3. Open the provided link in your web browser to access the tokenizer interface.

## Example Input

You can test the tokenizer with the following Hindi sentences:

- "यहां वर्तमान में 20 हजार पुस्तकें थी जो अभी रैन बसेरा परिसर के कक्ष में रखी हुई है।"
- "भारत एक विशाल देश है।"
- "मैं हिंदी में बात कर रहा हूं।"
- "नमस्ते, आप कैसे हैं?"
- "दिल्ली भारत की राजधानी है।"

## Training Logs and Statistics

During the training of the tokenizer, the following statistics were recorded:

- **Initial tokens**: 3,480,023
- **Final tokens**: 431,205
- **Initial bytes**: 13,920,092
- **Final bytes**: 1,724,820
- **Token compression ratio**: 8.07X
- **Byte compression ratio**: 129.13X

The tokenizer was saved to: `model/hi_tokenizer_regex.json`



