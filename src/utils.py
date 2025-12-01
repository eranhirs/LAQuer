
import json
import logging
import os
from typing import List
import pandas as pd

from src.consts import LFQA_DATASET, MDS_DATASET, TASK_TO_DATASET



def load_results_files(split: str, task: str, techniques: List[str]) -> dict:
    results = {}
    for technique in techniques:
        path = f"results/{split}/{task}/{technique}/results.json"
        assert os.path.exists(path), f"File {path} does not exist"
        with open(path) as f:
            results[technique] = {
                "results": [json.loads(line) for line in f.readlines()]
            }

        with open(f"results/{split}/{task}/{technique}/config.json") as f:
            file_config = json.loads(f.read())
            results[technique]['config'] = file_config

        dataset = TASK_TO_DATASET[task]
        results[technique]['dataset'] = dataset

        documents = load_dataset(dataset, split=split)
        results[technique]['documents'] = documents

    return results

def dedup_and_sort_spans(span_list: List[tuple]) -> List[tuple]:
    """
    Deduplicate, sort, and merge overlapping spans
    
    Parameters
    ----------
    span_list: List of tuples
        List of (start_idx, end_idx) tuples
        
    Returns
    -------
    List of tuples
        Deduplicated, sorted, and merged list of (start_idx, end_idx) tuples
    """

        
    # dedup
    deduped = []
    seen = set()
    for span_obj in span_list:
        if span_obj is not None and span_obj not in seen:
            seen.add(span_obj)
            deduped.append(span_obj)

    # sort
    deduped_and_sorted = sorted(deduped, key=lambda span_obj: (span_obj[0], span_obj[1]))
    
    # merge overlapping spans
    spans_objs = []
    for span_obj in deduped_and_sorted:
        span = span_obj
        if not spans_objs:
            spans_objs.append(span_obj)
        else:
            is_intersecting_with_last = spans_objs[-1][1] + 1 >= span[0]  # +1 to take into consideration space
            if is_intersecting_with_last:
                new_span = (spans_objs[-1][0], max(spans_objs[-1][1], span[1]))
                spans_objs[-1] = tuple(new_span)
            else:
                spans_objs.append(span_obj)
    
    return spans_objs



def load_dataset(dataset: str, split: str):
    if dataset == MDS_DATASET:
        return load_mn_dataset()
    elif dataset == LFQA_DATASET:
        return load_evaluating_dataset(split)
    else:
        raise ValueError(f"Invalid dataset {dataset}")
    
def load_mn_dataset():
    with open("data/MDS/sents_separation.json") as f:
        data = json.loads(f.read())
    documents = data['documents']

    # Change per-sent format to per-document format
    new_documents = {}
    for topic, topic_documents in documents.items():
        new_documents[topic] = {document_id: ''.join(sent_obj['sent_text'] for sent_obj in document) for document_id, document in topic_documents.items()}
    documents = new_documents
    
    return documents


def load_evaluating_dataset(split: str):
    file_path = f"data/LFQA/{split}.json"
    with open(file_path) as f:
        data = json.loads(f.read())
        
    return data['documents']
