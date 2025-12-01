import json
import logging
import os
import random
import shutil
from time import time
import pandas as pd
from tqdm import tqdm
import transformers
from collections import defaultdict
from src.consts import *
from src.decontextualize_facts import get_decontextualized_path


def get_highlight_obj_source_id(highlight_obj):
    if highlight_obj.get('docSentText') is None:
        return f"{highlight_obj['documentFile']}"
    else:
        # docSentCharIdx instead of the spans, because we print the sentence and want to avoid printing the same sentence multiple times
        # in attr-first original files we had one per docSpanOffsets, so it made sense to use the offsets as key
        return f"{highlight_obj['documentFile']}__{highlight_obj['docSentCharIdx']}"



    
def extract_attribution(datapoint, alignment_model):
    is_aligned = datapoint['source_spans'] != {}
    if not is_aligned:
        return {
            'results': pd.DataFrame([{
                "topic": datapoint['topic'],
                "scuSentence": datapoint['sentence'],
                "complete_scuSentence": datapoint['complete_scuSentence'],
                "scuSpanOffsets": datapoint['scuSpanOffsets'],
                "is_sampled": datapoint['is_sampled'],
                "documentFile": None,
                "docSpanText": None,
                "docSpanOffsets": None,
                "fact_idx": datapoint['fact_idx']
            }])
        }
    
    start = time()
    result = alignment_model.extract_attribution(datapoint)
    result["fact_idx"] = datapoint['fact_idx']  # necessary to keep track for later use in the autoais with the original
    end = time()
    result['time_start'] = start
    result['time_end'] = end
    return result


def get_laquer_method(laquer_method_name, task, args):
    if laquer_method_name == LLM_LAQUER_METHOD:
        from src.laquer_methods.llm_method import LLMBasedAlignment
        laquer_model = LLMBasedAlignment(task=task, args=args)
    else:
        raise ValueError(f"Invalid alignment technique: {laquer_method_name}")

    return laquer_model
    

def get_laquer_method_results_path(split, task, technique, laquer_method_name):
    return f'results/{split}/{task}/{technique}/{laquer_method_name}_results.csv'

def main(task: str, split: str, results, args, laquer_method_name: str):
    """
    Runs LLM-based LAQuer method.
    
    """
    
    logging.info(f"Running LLM-based LAQuer method")
    

    for technique, technique_obj in results.items():
        logging.info(f"Technique: {technique}")

        laquer_model = get_laquer_method(laquer_method_name, task, args)
                    

        # fix formats (can be deleted if no more old_results_output_file_path exist)
        results_output_file_path = get_laquer_method_results_path(split, task, technique, laquer_method_name)
        responses_output_file_path = f'results/{split}/{task}/{technique}/{laquer_method_name}_responses.csv'
        
        def rows_to_input_obj(rows, instance_unique_id, documents):
            any_row = rows.iloc[0]
            sentence = any_row['scuSentence']
            source_spans = {}
            source_metadata = {}
            rows['source_unique_id'] = rows.apply(get_highlight_obj_source_id, axis=1)
            source_granularity = 'sentence'
            for group_idx, rows_by_source in rows.groupby('source_unique_id'):
                any_row_by_source = rows_by_source.iloc[0]
                
                is_aligned = any_row_by_source.get('documentFile') is not None
                if not is_aligned:
                    continue
                    
                if any_row_by_source.get('docSpanText') is not None:
                    source_text = ' '.join([x.replace(HIGHLIGHT_SEP, ' ') for x in rows_by_source['docSpanText'].tolist()])
                else:
                    source_text = documents[any_row_by_source['documentFile']]
                    source_granularity = 'document'
                    
                source_unique_id = group_idx
                source_spans[source_unique_id] = source_text
                source_metadata[source_unique_id] = rows_by_source.to_dict('records')

            
            input_obj = {
                "topic": instance_unique_id,
                "unique_id": instance_unique_id,
                "source_spans": source_spans,
                "source_metadata": source_metadata,
                "sentence": sentence,
                "scuSpanOffsets": any_row['scuSpanOffsets'],
                "complete_scuSentence": any_row['complete_scuSentence'],
                "is_sampled": any_row['is_sampled'],
                "source_granularity": source_granularity,
                "fact_idx": int(any_row['fact_idx'])
            }
            
            if 'question' in any_row:
                input_obj['question'] = any_row['question']
            elif 'query' in any_row:
                input_obj['question'] = any_row['query']
                
            return input_obj

        
        def create_input_objs(documents):
            input_objs = []
            for result in technique_obj['results']:
                curr_documents = documents[result['unique_id']]
                curr_input_objs = pd.DataFrame(result['set_of_highlights_in_context']).groupby('fact_idx').apply(lambda rows: rows_to_input_obj(rows, result['unique_id'], curr_documents)).tolist()
                input_objs.extend(curr_input_objs)
            
            return input_objs
        
        documents = technique_obj['documents']
        input_objs = create_input_objs(documents=documents)

        transformers.set_seed(42)
        results_and_responses = [extract_attribution(input_obj, alignment_model=laquer_model) for input_obj in tqdm(input_objs)]
        
        results = pd.concat([result_and_response['results'] for result_and_response in results_and_responses])
        responses = pd.DataFrame([{k: json.dumps(v) if isinstance(v, dict) else v for k, v in result_and_response.items() if k != 'results'} for result_and_response in results_and_responses])
        responses['dataset'] = technique_obj['dataset']
        responses['split'] = split
        
        def save_func(results, responses):
            results.to_csv(results_output_file_path, index=False)
            
            responses.to_csv(responses_output_file_path, index=False)
            
        save_func(results, responses)
