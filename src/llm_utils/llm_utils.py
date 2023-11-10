import textwrap
import tiktoken


# OpenAI specific.
def count_tokens(model: str, string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# OpenAI specific.
def calculate_cost(num_input_tokens: int, num_output_tokens: int, model_type: str) -> float:
    """
    Calculate the cost of processing a request based on model type.

    Args:
        num_input_tokens (int): Number of input tokens.
        num_output_tokens (int): Number of output tokens.
        model_type (str): The type of GPT model used (model name).

    Returns:
        The cost of processing the request, in USD.
    """
    # Latest pricing info from OpenAI (https://openai.com/pricing and
    # https://platform.openai.com/docs/deprecations/), as of November 9, 2023.
    PRICING_PER_1000 = {
        "gpt-3.5-turbo-1106":       {"input": 0.001,  "output": 0.002},
        "gpt-3.5-turbo":            {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-0613":       {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-0301":       {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k":        {"input": 0.003,  "output": 0.004},
        "gpt-3.5-turbo-16k-0613":   {"input": 0.003,  "output": 0.004},
        "gpt-4-1106-preview":       {"input": 0.01,   "output": 0.03},
        "gpt-4":                    {"input": 0.03,   "output": 0.06},
        "gpt-4-0314":               {"input": 0.03,   "output": 0.06},
        "gpt-4-32k":                {"input": 0.06,   "output": 0.12},
        "gpt-4-32k-0314":           {"input": 0.06,   "output": 0.12},
    }

    if not (price_per_1000 := PRICING_PER_1000.get(model_type)):
        raise ValueError(
            f'Unknown model "{model_type}". Choose from: {", ".join(m for m in PRICING_PER_1000)}.'
        )

    return num_input_tokens / 1000 * price_per_1000['input'] + \
           num_output_tokens / 1000 * price_per_1000['output']



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


def read_lines(file_path: str, start_line: int, end_line: int) -> tuple[str, int]:
    """
    Read lines from a file.

    Args:
        file_path (str): The path of the file to read.
        start_line (int): The line number of the first line to include (1-indexed). Will be bounded below by 1.
        end_line (int): The line number of the last line to include (1-indexed). Will be bounded above by file's line count.

    Returns:
        The lines read as an array and the number of the first line included.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    # Prevent pathological case where lines are REALLY long.
    max_chars_per_line = 128

    def truncate(s, l):
        """
        Truncate the string to at most the given length, adding ellipses if truncated.
        """
        if len(s) < l:
            return s
        else:
            return s[:l] + "..."

    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [truncate(line.rstrip(), max_chars_per_line) for line in lines]

    # Ensure indices are in range.
    start_line = max(1, start_line)
    end_line = min(len(lines), end_line)

    return (lines[start_line - 1 : end_line], start_line)
