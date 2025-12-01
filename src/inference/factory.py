

from src.consts import *
from src.inference.remote_inference_wrapper import RemoteInferenceWrapper
from src.inference.trueteacher_entailment_model import TrueTeacherEntailmentModel


class Factory:
    """
    Generates necessary inference models/wrappers based on the provided config.
    """
    
    def __init__(self, args) -> None:
        self.args = args
        self.cache = {}


    def inference_wrapper(self):
        if 'inference_wrapper' in self.cache:
            return self.cache['inference_wrapper']
        
        inference_wrapper = RemoteInferenceWrapper(self.args)
            
        self.cache['inference_wrapper'] = inference_wrapper
        
        return inference_wrapper
    
    def entailment_model(self):
        if 'entailment_model' in self.cache:
            return self.cache['entailment_model']
        
        if self.args.entailment_model == TRUE_TEACHER_ENTAILMENT_MODEL_IDENTIFIER:
            entailment_model = TrueTeacherEntailmentModel(device='auto')  # run on multiple gpus to avoid OOM
        elif self.args.entailment_model == 'llm_prompt':
            entailment_model = LLMPromptEntailmentModel(self.inference_wrapper())
        else:
            raise ValueError(f"Unknown entailment model {self.config['entailment_model']}")
        
        self.cache['entailment_model'] = entailment_model
        
        return entailment_model