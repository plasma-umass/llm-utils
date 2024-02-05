import re
import json
import textwrap
from typing import Any

import tiktoken


# OpenAI specific.
def count_tokens(model: str, string: str) -> int:
    """Returns the number of tokens in a text string."""

    def extract_after_slash(s: str) -> str:
        # Split the string by '/' and return the part after it if '/' is found, else return the whole string
        parts = s.split("/", 1)  # The '1' ensures we split at the first '/' only
        return parts[1] if len(parts) > 1 else s

    try:
        encoding = tiktoken.encoding_for_model(extract_after_slash(model))
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except KeyError:
        return 0


# OpenAI specific.
def calculate_cost(
    num_input_tokens: int, num_output_tokens: int, model_type: str
) -> float:
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
        "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-0613": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-0301": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "gpt-3.5-turbo-16k-0613": {"input": 0.003, "output": 0.004},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-0314": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-4-32k-0314": {"input": 0.06, "output": 0.12},
    }

    if not (price_per_1000 := PRICING_PER_1000.get(model_type)):
        raise ValueError(
            f'Unknown model "{model_type}". Choose from: {", ".join(m for m in PRICING_PER_1000)}.'
        )

    return (
        num_input_tokens / 1000 * price_per_1000["input"]
        + num_output_tokens / 1000 * price_per_1000["output"]
    )


def word_wrap_except_code_blocks(text: str, width: int = 80) -> str:
    """
    Wraps text except for code blocks for nice terminal formatting.

    Splits the text into paragraphs and wraps each paragraph,
    except for paragraphs that are inside of code blocks denoted
    by ` ``` `. Returns the updated text.

    Args:
        text (str): The text to wrap.
        width (int): The width of the lines to wrap at, passed to `textwrap.fill`.

    Returns:
        The wrapped text.
    """
    # Find code blocks.
    blocks = []
    is_code_block = []
    block = []
    in_code_block = False
    for line in text.split("\n"):
        if line.startswith("```"):
            if in_code_block:
                block.append(line)
                blocks.append(block)
                block = []
            else:
                blocks.append(block)
                block = [line]
            is_code_block.append(in_code_block)
            in_code_block = not in_code_block
        else:
            block.append(line)
    blocks.append(block)
    is_code_block.append(in_code_block)

    # Split text (non-code) paragraphs.
    copied_blocks = []
    copied_is_code_block = []
    for i in range(len(blocks)):
        if is_code_block[i]:
            copied_blocks.append(blocks[i])
            copied_is_code_block.append(True)
        else:
            block = []
            for line in blocks[i]:
                if not line:
                    if block:
                        copied_blocks.append(block)
                        copied_is_code_block.append(False)
                        block = []
                else:
                    block.append(line)
            if block:
                copied_blocks.append(block)
                copied_is_code_block.append(False)
                block = []
    blocks = copied_blocks
    is_code_block = copied_is_code_block

    # Word wrap text blocks.
    for i in range(len(blocks)):
        if not is_code_block[i]:
            blocks[i] = [textwrap.fill(l, width) for l in blocks[i]]

    # Join lines with a single newline, blocks with two newlines.
    return "\n\n".join(["\n".join(b) for b in blocks])


def read_lines(file_path: str, start_line: int, end_line: int) -> tuple[list[str], int]:
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

    def truncate(s: str, l: int) -> str:
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


def number_group_of_lines(group: list[str], first: int, strip: bool = True) -> str:
    """
    Add line numbers for each line in the input list.

    Args:
        group (list[str]): The lines to number.
        first (int): The number for the first line.
        strip (bool): Whether to strip leading and trailing blank lines.

    Returns:
        A string concatenation of the numbered lines.
    """
    if strip:
        while group and not group[0].strip():
            group = group[1:]
            first += 1
        while group and not group[-1].strip():
            group = group[:-1]

    last = first + len(group) - 1
    max_line_number_length = len(str(last))
    result = "\n".join(
        [
            "{0:>{1}} {2}".format(first + i, max_line_number_length, line)
            for i, line in enumerate(group)
        ]
    )
    return result


def contains_valid_json(my_string: str) -> Any:
    """
    Parses JSON if valid, replacing any triple quotes with single quotes.
    Returns the parsed JSON blob if successful, None if not.
    """

    # Find start and end position of possible JSON string
    start_pos = my_string.find("{")
    end_pos = my_string.rfind("}") + 1

    # There is no JSON object if start or end position is -1
    if start_pos == -1 or end_pos == 0:
        return None

    json_string = my_string[start_pos:end_pos]
    pattern = r'"""(.*?)"""'
    processed_string = re.sub(
        pattern, lambda m: json.dumps(m.group(1)), json_string, flags=re.DOTALL
    )
    # processed_string = json_string.sub(r'"""', r'\"\"\"')
    try:
        return json.loads(processed_string, strict=False)
    except json.JSONDecodeError as e:
        # print(e)
        # print("IN DECODING\n")
        # print(processed_string)
        return None


def extract_code_blocks(text: str) -> list[str]:
    pattern = r"```(python)?(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    return [block[1].strip() for block in blocks]


def parse_chatlog(log: str) -> list[dict[str, str]]:
    entries = log.split("\n\n")  # split entries by empty lines
    result = []
    for entry in entries:
        if not entry.strip():  # ignore empty entries
            continue
        role, *content = entry.split(": ", 1)
        content_str = content[0] if content else ""
        result.append({"role": role, "content": content_str})
    return result
