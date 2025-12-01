import json
import logging
from typing import List

from src.inference.utils import OutputValidationException
from src.inference.generate_json_object.utils import replace_json_key_with_valid_json_end

logger = logging.getLogger(__name__)


def remote_generate_json_object(inference_wrapper, messages: List[dict]):
    response_before_parsing = call_remote__generate_json_object(inference_wrapper, messages)

    generated_json_object = parse_json_output(response_before_parsing)
    
    return generated_json_object, [response_before_parsing]


def call_remote__generate_json_object(inference_wrapper, messages: List[dict], generation_config):
    logger.debug("generating json object")
    
    stop_words = get_stop_words(generation_config)
    
    completion = inference_wrapper.client.chat.completions.create(
        model=inference_wrapper.model,
        messages=messages,
        stop=stop_words,
        max_tokens=1024
    )
    
    finish_reason = completion.choices[0].finish_reason
    if not isinstance(finish_reason, str):
        finish_reason = finish_reason.value

    return {
        "text": completion.choices[0].message.content,
        "finish_reason_value": finish_reason
    }
    

def get_stop_words(generation_config):
    """
    Unlike the local implementation where we can simply add a stopping criteria,
    here we need to provide a list of tokens that the model should stop at.
    For example, " doesn't work, but ": (with colon :) does work
    This is not robust and requires adding various tokens that the model should stop at.
    """

    json_keys_with_quotations = [f'"{json_key}"' for json_key in generation_config.get('stopping_json_keys', [])]
    
    stop_words = []
    
    for json_key_with_quotation in json_keys_with_quotations:
        stop_words.append(f"{json_key_with_quotation}:")
        stop_words.append(f"{json_key_with_quotation}")
    
    if generation_config['remote_client'] == 'openai':
        stop_words.append("}")
    else:
        stop_words.extend(["}", "\n}", "}\n", "\n}\n", "\"}", "}\"", "}\n\n", "}`"])
    
    return stop_words


def parse_json_output(response: dict):
    """
    The remote implementation is based on regexes and not vocab tokens like the local implementation
    """
    
    try:
        response_before_parsing = response['text']
        finish_reason_value = response['finish_reason_value']
        
        # TODO: handle differently finish_reason_value == 'length' (the model reached the max tokens)
        
        outputs = remove_text_before_json_start_token(response_before_parsing)

        outputs = replace_json_key_with_valid_json_end(outputs)

    except Exception as e:
        raise OutputValidationException(
            model_output=response_before_parsing,
            feedback="Your last response did not include a valid JSON object.",
            validation_output="Failed to fix json object from decoded text",
        )

    try:
        generated_json_object = json.loads(outputs)
    except Exception as e:
        raise OutputValidationException(
            model_output=response_before_parsing,
            feedback="Your last response did not include a valid JSON object.",
            validation_output={
                "outputs": outputs
            }
        )

    return generated_json_object


def remove_text_before_json_start_token(response_before_parsing: str) -> str:
    """
    Input: '://www.wikinews.org/wiki/Template:Highlighter {"id": 4, "parent_ids": [0], "text" ... }
    Output: '{"id": 4, "parent_ids": [0], "text" ... }
    """
    
    json_start_index = response_before_parsing.rindex('{')  # use rindex in case the text has {, we want to find the last
    outputs = response_before_parsing[json_start_index:]
    return outputs

