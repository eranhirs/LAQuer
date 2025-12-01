import logging
import os
from typing import List
from string import punctuation
import pandas as pd

from src.consts import *
from src.decompose_to_facts import get_facts_path
from src.decontextualize_facts import get_decontextualized_path
from src.inference.factory import Factory
from src.laquer_methods.run_laquer_method import get_laquer_method_results_path

logger = logging.getLogger(__name__)


def extract_document_attribution_from_rows(rows, documents) -> str:
    any_row = rows.iloc[0]
    
    rows['docSpanOffsets'].apply(lambda offsets: eval(offsets) if isinstance(offsets, str) else offsets)
    
    document_file = any_row['documentFile']
    if not document_file or pd.isna(document_file):
        return None
    
    # when all the document is attributed - docSpanOffsets is None
    if not any_row['docSpanOffsets']:
        attribution = [doc for doc_id, doc in documents.items() if doc_id == document_file][0]
    else:
        # order by docSpanOffset's first subspan
        rows['docSpanOffsets'] = rows['docSpanOffsets'].apply(lambda offsets: eval(offsets) if isinstance(offsets, str) else offsets)
        rows['first_subspan_offset'] = rows['docSpanOffsets'].apply(lambda offsets: offsets[0][0])
        rows = rows.sort_values(by='first_subspan_offset')
        doc_spans = rows['docSpanText'].apply(lambda text: " ".join(text.split(HIGHLIGHT_SEP)).strip()).tolist()

        doc_spans = [elem for elem in doc_spans if elem] # remove highlights that are empty lines
        doc_spans = [elem if elem[-1] in punctuation else f"{elem}." for elem in doc_spans] # add period to spans without punctuation in their end (as each such span comes from separate sentences)
        curr_attribution = " ".join(doc_spans)
        attribution = " ".join(curr_attribution.split()) # replace consecutive spaces/new lines with a single space
    
    return attribution

def extract_fact_attribution_from_rows(rows, documents) -> dict:
    attribution = rows.groupby('documentFile').apply(lambda rows: extract_document_attribution_from_rows(rows, documents))
    
    if attribution.empty:
        return {}
    else:
        return attribution.to_dict()


def handle_fact(rows, documents, facts_df, entailment_model) -> bool:
    any_row = rows.iloc[0]
    curr_documents = documents[any_row['topic']]
    attribution = extract_fact_attribution_from_rows(rows, curr_documents)
    
    attribution = [attributed_doc_text for doc_id, attributed_doc_text in attribution.items() if attributed_doc_text is not None]
    
    if len(attribution) == 0:
        return False
    
    premise = '\n '.join([attributed_doc_text.replace('\n', ' ') for attributed_doc_text in attribution])
    hypothesis = facts_df[(facts_df['unique_id'] == any_row['topic']) & (facts_df['fact_idx'] == int(any_row['fact_idx']))]['fact'].values[0]

    entailment_result, _ = entailment_model.generate_entailment_decision(premise_text=premise, hypothesis_text=hypothesis)
    return entailment_result
    

def handle_instance(rows, documents, facts_df, entailment_model):
    instance_results = rows.groupby('fact_idx').apply(handle_fact, documents, facts_df, entailment_model)
    instance_results.name = 'entailment_result'
    return instance_results.reset_index()


def main(task: str, split: str, results, args, laquer_method_name: str):
    factory = Factory(args)
    entailment_model = factory.entailment_model()
    
    for facts_file_for_evaluation in [DECONTEXTUALIZED_FACTS_IDENTIFIER, FACTS_IDENTIFIER]:
        logger.info(facts_file_for_evaluation)
        
        for technique, technique_obj in results.items():
            logger.info(f"Technique: {technique}")
            
            if facts_file_for_evaluation == FACTS_IDENTIFIER:
                facts_path = get_facts_path(split, task, technique)
            elif facts_file_for_evaluation == DECONTEXTUALIZED_FACTS_IDENTIFIER:
                facts_path = get_decontextualized_path(split, task, technique)
            else:
                raise ValueError(f"Unknown facts method: {facts_file_for_evaluation}")
            
            if not os.path.exists(facts_path):
                logging.info(f"Facts path {facts_path} does not exist, skipping...")
                continue

            facts_df = pd.read_csv(facts_path)            

            documents = technique_obj['documents']

            results_output_file_path = get_laquer_method_results_path(split, task, technique, laquer_method_name)
            laquer_method_results = pd.read_csv(results_output_file_path)
            
            entailment_results = laquer_method_results.groupby(['topic']).apply(lambda rows: handle_instance(rows, documents, facts_df, entailment_model)).reset_index()
            
            save_func(entailment_results, split, task, technique, facts_file_for_evaluation)
            print_results(entailment_results, facts_file_for_evaluation)

def save_func(results, split, task, technique, facts_file_for_evaluation):
    evaluation_results_file_path = f"results/{split}/{task}/{technique}/{facts_file_for_evaluation}__evaluation_results.csv"
    results.to_csv(evaluation_results_file_path, index=False)
    logging.info(f"Saved evaluation results to {evaluation_results_file_path}")
    
def print_results(results, facts_file_for_evaluation: str):
    print(f"{facts_file_for_evaluation} evaluation Results:")
    print(f"AutoAIS: {results['entailment_result'].mean():.2f}")
