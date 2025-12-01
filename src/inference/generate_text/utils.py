import re


def parse_generated_text_output(response: dict):
    
    response_before_parsing = response['text']
    if "the text is:" in response_before_parsing.lower():
        fixed_text = re.findall(r"the text is:(.*)", response_before_parsing, flags=re.IGNORECASE)[0].strip()
        return fixed_text

    raise ValueError(f"Could not parse response: {response_before_parsing}")
