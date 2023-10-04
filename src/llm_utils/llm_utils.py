import tiktoken

# OpenAI specific.
def count_tokens(model: str, string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# OpenAI specific.
def calculate_cost(
    num_input_tokens: int, num_output_tokens: int, model_type: str, context_size: str
):
    """
    Calculate the cost of processing a request based on model type and context size.

    Args:
        num_input_tokens (int): Number of input tokens.
        num_output_tokens (int): Number of output tokens.
        model_type (str): The type of GPT model used.
        context_size (str): Context size (e.g., 8K, 32K, 4K, 16K).

    Returns:
        The cost of processing the request, in USD.
    """
    # Latest pricing info from OpenAI (https://openai.com/pricing), as of Oct 3 2023.
    PRICING_PER_1000 = {
        "gpt-4": {
            "8K": {"input": 0.03, "output": 0.06},
            "32K": {"input": 0.06, "output": 0.12},
        },
        "gpt-3.5-turbo": {
            "4K": {"input": 0.0015, "output": 0.002},
            "16K": {"input": 0.003, "output": 0.004},
        },
    }

    # Ensure model_type and context_size are valid.
    if (
        model_type not in PRICING_PER_1000
        or context_size not in PRICING_PER_1000[model_type]
    ):
        raise ValueError(
            f"Invalid model_type or context_size. Choose from {', '.join(PRICING_PER_1000.keys())} and respective context sizes."
        )

    # Calculate total cost per token and total tokens.
    input_cost_per_token = PRICING_PER_1000[model_type][context_size]["input"] / 1000
    output_cost_per_token = PRICING_PER_1000[model_type][context_size]["output"] / 1000
    total_tokens = num_input_tokens + num_output_tokens

    # Calculate cost for input and output separately.
    input_cost = num_input_tokens * input_cost_per_token
    output_cost = num_output_tokens * output_cost_per_token

    return input_cost + output_cost


def word_wrap_except_code_blocks(text: str) -> str:
    """
    Wraps text except for code blocks for nice terminal formatting.

    Splits the text into paragraphs and wraps each paragraph,
    except for paragraphs that are inside of code blocks denoted
    by ` ``` `. Returns the updated text.

    Args:
        text: The text to wrap.

    Returns:
        The wrapped text.
    """
    # Split text into paragraphs.
    paragraphs = text.split("\n\n")
    wrapped_paragraphs = []
    # Check if currently in a code block.
    in_code_block = False
    # Loop through each paragraph and apply appropriate wrapping.
    for paragraph in paragraphs:
        # If this paragraph starts and ends with a code block, add it as is.
        if paragraph.startswith("```") and paragraph.endswith("```"):
            wrapped_paragraphs.append(paragraph)
            continue
        # If this is the beginning of a code block add it as is.
        if paragraph.startswith("```"):
            in_code_block = True
            wrapped_paragraphs.append(paragraph)
            continue
        # If this is the end of a code block stop skipping text.
        if paragraph.endswith("```"):
            in_code_block = False
            wrapped_paragraphs.append(paragraph)
            continue
        # If we are currently in a code block add the paragraph as is.
        if in_code_block:
            wrapped_paragraphs.append(paragraph)
        else:
            # Otherwise, apply text wrapping to the paragraph.
            wrapped_paragraph = textwrap.fill(paragraph)
            wrapped_paragraphs.append(wrapped_paragraph)
    # Join all paragraphs into a single string.
    wrapped_text = "\n\n".join(wrapped_paragraphs)
    return wrapped_text
