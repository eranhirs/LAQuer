import json
import logging
import os
import pandas as pd
from tqdm import tqdm

from src.decontextualize_facts import get_decontextualized_path
from src.consts import *
from src.decompose_to_facts import get_facts_path

       

def change_sent_alignment_based_on_fact(rows, fact_row, is_aligned: bool):
    alignments_flattened = eval(fact_row['factOffsets'])
    
    # 1. if not aligned, all rows are aligned with the fact
    if not is_aligned:
        relevant_rows = rows.to_dict('records')
    # 2. if aligned, per fact, based on its localization in the output, collect all rows that are aligned with the fact
    else: 
        relevant_rows = []
        for _, row in rows.iterrows():
            for alignment_span in alignments_flattened:
                if 'scuSpanOffsets' in row:
                    
                    if not isinstance(row['scuSpanOffsets'], list):
                        continue
                    
                    row_spans = row['scuSpanOffsets']
                
                else:
                    if row['scuSentCharIdx'] is None:
                        continue

                    row_spans = [[row['scuSentCharIdx'],row['scuSentCharIdx'] + len(row['scuSentence'])]]

                
                are_intersecting = False
                for row_span in row_spans:
                    are_intersecting = (alignment_span[0] <= row_span[0] <= alignment_span[1]) or (alignment_span[0] <= row_span[1] <= alignment_span[1])
                    if are_intersecting:
                        relevant_rows.append(row.to_dict())
                        break
                if are_intersecting:
                    break
    
    # 3. merge rows and change that fact is the scuSentence
    relevant_rows = [row for row in relevant_rows if row['documentFile'] is not None]
    if len(relevant_rows) > 0:
        # set fact as scusentence
        new_rows = []
        for relevant_row in relevant_rows:
            relevant_row['scuSentence'] = fact_row['fact']
            relevant_row['complete_scuSentence'] = fact_row['sentence']
            relevant_row['scuSpanOffsets'] = alignments_flattened
            relevant_row['orig_scuSentCharIdx'] = int(fact_row['scuSentCharIdx'])
            relevant_row['is_sampled'] = fact_row['is_sampled']
            relevant_row['is_sampled__summary_sent'] = fact_row['is_sampled__summary_sent']
            relevant_row['fact_idx'] = int(fact_row['fact_idx'])
            if 'factscore_fact' in fact_row:
                relevant_row['factscore_fact'] = fact_row['factscore_fact']
            
            new_rows.append(relevant_row)
        relevant_rows = new_rows

    # 4. mark an empty alignment if not found any rows
    if len(relevant_rows) == 0:
        relevant_rows.append({
            "documentFile": None,
            "docSentCharIdx": None,
            "sent_idx": None,
            "docSentText": None,
            "docSpanText": None,
            "docSpanOffsets": None,
            "orig_scuSentCharIdx": int(fact_row['scuSentCharIdx']),
            "scuSentence": fact_row['fact'],
            "is_sampled": fact_row['is_sampled'],
            "is_sampled__summary_sent": fact_row['is_sampled__summary_sent'],
            "fact_idx": int(fact_row['fact_idx'])
        })
        
    return relevant_rows

def change_sent_alignment_based_on_facts(rows, facts_df, is_aligned: bool):    
    new_alignments = facts_df.apply(lambda fact_row: change_sent_alignment_based_on_fact(rows, fact_row, is_aligned), axis=1).tolist()
    
    return [new_alignment for fact_new_alignments in new_alignments for new_alignment in fact_new_alignments]


def change_alignments_based_on_facts(row, facts_df, is_aligned: bool):
    changed_alignments = change_sent_alignment_based_on_facts(pd.DataFrame(row['set_of_highlights_in_context']), facts_df, is_aligned)
        
    # keep only sampled facts to avoid large overhead
    changed_alignments = [alignment for alignment in changed_alignments if alignment['is_sampled']]    
    
    row['set_of_highlights_in_context'] = changed_alignments


def save_func(technique_obj, facts_results_path):
    logging.info(f"writing to {facts_results_path}")        
    results_dumped = "\n".join([json.dumps(result) for result in technique_obj['results']])
    with open(facts_results_path, 'w') as f:
        f.write(results_dumped)


def main(task: str, split: str, results):
    """
    Most attribution baselines extract sentence-level alignments, this script changes them to fact-level alignments.
    In terms of format, the facts will be the new scuSentence.
    """
    logging.info("changing alignments based on facts")
    
    for facts_method in [DECONTEXTUALIZED_FACTS_IDENTIFIER, FACTS_IDENTIFIER]:
        logging.info(facts_method)
        
        for technique, technique_obj in results.items():
            logging.info(f"Technique: {technique}")
            
            if facts_method == FACTS_IDENTIFIER:
                facts_path = get_facts_path(split, task, technique)
            elif facts_method == DECONTEXTUALIZED_FACTS_IDENTIFIER:
                facts_path = get_decontextualized_path(split, task, technique)
            else:
                raise ValueError(f"Unknown facts method: {facts_method}")
            
            if not os.path.exists(facts_path):
                logging.info(f"Facts path {facts_path} does not exist, skipping...")
                continue

            facts_df = pd.read_csv(facts_path)
            
            
            for result in tqdm(technique_obj['results']):
                unique_id = result['unique_id']
                change_alignments_based_on_facts(result, facts_df[facts_df['unique_id'] == unique_id], is_aligned=technique_obj['config']['aligned'])
              
