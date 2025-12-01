
from typing import List
import logging


from litellm import completion

logger = logging.getLogger(__name__)


def remote_generate_text(model: str, messages: List[dict]):
    """
    The prompt should include an instruction to the model to generate "the text is:"
    """
    
    logger.debug("generating text")
    
    response = completion(
        model=model,
        messages=messages,
        max_tokens=1024    
    )
    
    finish_reason_value = response.choices[0].finish_reason
    if not isinstance(finish_reason_value, str):
        finish_reason_value = finish_reason_value.value

    return {
        "text": response.choices[0].message.content,
        "input_len": len(''.join(message['content'] for message in messages)),
        "output_len": len(response.choices[0].message.content),
        "finish_reason_value": finish_reason_value,
        "created": response.created,
        "id": response.id
    }
    