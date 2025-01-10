import gradio as gr
from tokenizer import Tokenizer, load_config
import json
import html

# Load the tokenizer
config = load_config("config_app.yml")
tokenizer = Tokenizer.load(config["tokenizer_file_path"])
tokenizer.config = config

def highlight_tokens(text: str, encoded_tokens: list) -> str:
    """
    Create HTML with highlighted tokens in the text.

    Args:
        text (str): The original input text to be tokenized.
        encoded_tokens (list): A list of encoded token IDs.

    Returns:
        str: HTML string with highlighted tokens and tooltips showing token IDs.
    """
    decoded_tokens = []
    current_pos = 0
    html_text = ""
    
    # Decode each token and create spans with different colors
    for i, token in enumerate(encoded_tokens):
        token_bytes = tokenizer.decode([token])
        decoded_tokens.append(token_bytes)
        
        # Find the token in the original text
        token_pos = text.find(token_bytes, current_pos)
        if token_pos != -1:
            # Add any skipped text
            if token_pos > current_pos:
                html_text += html.escape(text[current_pos:token_pos])
            
            # Add the highlighted token with improved tooltip
            color = f"hsl({(i * 60) % 360}, 80%, 85%)"
            html_text += f'''
                <span 
                    style="background-color: {color}; 
                           border-radius: 3px; 
                           padding: 0 3px; 
                           margin: 0 1px;
                           position: relative;
                           cursor: help;"
                    onmouseover="this.querySelector('.tooltip').style.display='block'"
                    onmouseout="this.querySelector('.tooltip').style.display='none'">
                    {html.escape(token_bytes)}
                    <span class="tooltip" 
                          style="display: none;
                                 position: absolute;
                                 bottom: 100%;
                                 left: 50%;
                                 transform: translateX(-50%);
                                 background-color: #333;
                                 color: white;
                                 padding: 4px 8px;
                                 border-radius: 4px;
                                 font-size: 12px;
                                 white-space: nowrap;
                                 z-index: 1000;">
                        Token ID: {token}
                    </span>
                </span>'''
            current_pos = token_pos + len(token_bytes)
    
    # Add any remaining text
    if current_pos < len(text):
        html_text += html.escape(text[current_pos:])
    
    return html_text

def process_text(text: str) -> tuple:
    """
    Process input text through the tokenizer and return results.

    Args:
        text (str): The input text to be processed.

    Returns:
        tuple: A tuple containing:
            - HTML string of highlighted tokens.
            - HTML string of token statistics.
            - String of token IDs.
    """
    try:
        # Encode the text
        encoded = tokenizer.encode(text)
        
        # Decode back to text
        decoded = tokenizer.decode(encoded)
        
        # Create token visualization
        highlighted_text = highlight_tokens(text, encoded)
        
        # Token statistics
        stats = {
            "Total Tokens": len(encoded),
            "Unique Tokens": len(set(encoded)),
            "Characters": len(text),
            "Bytes": len(text.encode('utf-8')),
            "Compression Ratio": f"{len(text.encode('utf-8')) / (len(encoded) * 4):.2f}x"
        }
        
        # Format statistics
        stats_html = "<div style='margin-top: 20px;'>"
        for key, value in stats.items():
            stats_html += f"<div style='margin: 5px 0;'><b>{key}:</b> {value}</div>"
        stats_html += "</div>"
        
        return (
            gr.HTML(highlighted_text),
            gr.HTML(stats_html),
            f"Token IDs: {encoded}"
        )
    except Exception as e:
        return (
            gr.HTML(f"<span style='color: red'>Error: {str(e)}</span>"),
            "",
            ""
        )

# Define example inputs
examples = [
    ["यहां वर्तमान में 20 हजार पुस्तकें थी जो अभी रैन बसेरा परिसर के कक्ष में रखी हुई है।"],
    ["भारत एक विशाल देश है।"],
    ["मैं हिंदी में बात कर रहा हूं।"],
    ["नमस्ते, आप कैसे हैं?"],
    ["दिल्ली भारत की राजधानी है।"]
]

# Custom CSS
custom_css = """
.container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}
.token-viz {
    font-family: monospace;
    line-height: 1.6;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 5px;
    background: white;
    margin: 10px 0;
    position: relative;
}
.stats {
    background: #f7f7f7;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.token-ids {
    font-family: monospace;
    padding: 15px;
    background: #f0f0f0;
    border-radius: 5px;
    overflow-wrap: break-word;
}
.tooltip {
    pointer-events: none;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
"""

# Create the Gradio interface
iface = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(
            label="Input Text",
            placeholder="Enter Hindi text here...",
            lines=3
        )
    ],
    outputs=[
        gr.HTML(label="Tokenized Text", elem_classes="token-viz"),
        gr.HTML(label="Statistics", elem_classes="stats"),
        gr.Textbox(label="Token IDs", elem_classes="token-ids")
    ],
    title="Hindi BPE Tokenizer Visualization",
    description="""
    This demo shows how the Hindi BPE tokenizer processes text. Each token is highlighted with a different color.
    Hover over the highlighted tokens to see their token IDs.
    """,
    examples=examples,
    theme=gr.themes.Soft(),
    css=custom_css,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=True) 