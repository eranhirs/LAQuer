from typing import List, Optional
import litellm
from src.inference.generate_json_object.remote_generate_json_object import remote_generate_json_object
from src.inference.generate_text.remote_generate_text import remote_generate_text

class RemoteInferenceWrapper:
    def __init__(self, args) -> None:
        super().__init__()
        
        self.args = args

    def generate_json_object(self, messages: List[dict], generation_config: Optional[dict] = None):
        return remote_generate_json_object(self, messages, generation_config)
    
    def generate_text(self, messages: List[dict]):
        return remote_generate_text(self.args.model, messages)
        
