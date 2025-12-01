import logging
from typing import List, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TrueTeacherEntailmentModel:
    def __init__(self, model_id="google/t5_11b_trueteacher_and_anli", device='cuda:7') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=device, torch_dtype='auto')
    
    def generate_entailment_decision(self, premise_text: Union[str, List[str]], hypothesis_text: Union[str, List[str]]):
        logging.debug('Start generating entailment decision')
        prompt = self.get_prompt(premise_text, hypothesis_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            max_length=5,
            pad_token_id=self.tokenizer.eos_token_id
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        assert result in ["0", "1"]
        entailment_result = result == "1"
            
        logging.debug('Finish generating entailment decision')
                
        return entailment_result, [{
            "premise": premise_text,
            "hypothesis": hypothesis_text,
            "result": result
        }]

    def get_prompt(self, premise, hypothesis):
        """
        Based on https://huggingface.co/google/t5_11b_trueteacher_and_anli
        """
        return f"""
Premise: {premise}
Hypothesis: {hypothesis}
"""
