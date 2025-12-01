import logging
import random


def generate_random_seed():
    return random.randint(0, 10000)


class OutputValidationException(Exception):
    """
    Exception raised when the model output fails validation.
    """
    
    def __init__(self, model_output: str, feedback: str, validation_output: str):
        super().__init__()
        self.model_output = model_output
        self.feedback = feedback
        self.validation_output = validation_output


def retry_wrapper(func):
    def retry_wrapper_inner(*args, **kwargs):
        num_retries = 0
        while num_retries < 5:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.exception(f"Error, retrying")
                num_retries += 1
        
        raise ValueError("Failed after 5 retries")  
    return retry_wrapper_inner
