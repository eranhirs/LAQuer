import json
import logging
import os
import pandas as pd
import spacy
from tqdm import tqdm
import transformers

from src.consts import FACTS_IDENTIFIER
from src.lexical_alignment.lexical_edit_distance_attribution import lexical_alignment_recursively
from src.utils import dedup_and_sort_spans
from src.third_party.factscore import FActScoreDecomposition


logger = logging.getLogger(__name__)




def fix_local_offset_to_doc_offset(offset, doc_span_offsets):
    """
    After the lexical_alignment_recursively, the process contains offsets of where we found the spans in the source texts. However, since the source texts are spans of the source and not the entire source, we need to fix the offsets to be relative to the entire source. This is especially problematic if the span is comprsied of more than one offset.
    
    Example:
    ('test36_0.txt_[[191, 271]]', (63,66)) -> ('test36_0.txt_[[191, 271]]', (191+63, 191+66))
    ('test36_0.txt_[[100, 200], [300, 400]]', (120,140)) -> ('test36_0.txt_[[100, 200], [300, 400]]', (300+120-100-1, 300+140-100-1))
    ('test36_0.txt_[[100, 200], [300, 400]]', (80,200)) -> ('test36_0.txt_[[100, 200], [300, 400]]', [(100+80, 200-1), (300, 300+60-1)])
    """

    new_offsets = []

    diff = 0
    for doc_span_offset in doc_span_offsets:
        doc_span_char_idx = doc_span_offset[0]
        if offset[0] + doc_span_char_idx - diff < doc_span_offset[1]:
            
            new_offset = (doc_span_offset[0] - diff + offset[0], doc_span_offset[0] - diff + offset[1])
            is_overflowing = new_offset[1] > doc_span_offset[1]
            if not is_overflowing:
                new_offsets.append(new_offset)
                return new_offsets
            else:
                new_offset = (new_offset[0], doc_span_offset[1])
                new_offsets.append(new_offset)
                
                distance_completed = new_offset[1] - new_offset[0]
                offset = (offset[0] + distance_completed, offset[1])
                
        diff += doc_span_offset[1] - doc_span_offset[0] + 1  # + 1 for space between spans
    raise ValueError(f"offset {offset} not found in {doc_span_offsets}")


class FactsDecomposition:
    def __init__(self, args):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp_lemma_only = spacy.load("en_core_web_sm", enable=['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer'])
        from nltk.corpus import stopwords
        self.stop_words = list(stopwords.words('english')) + ["'s"]
        
        self.factscore_decomposition = FActScoreDecomposition(args)

    def extract_decomposition(self, datapoint):
        response = self.factscore_decomposition.decompose(datapoint)
        results = self.parse_response(datapoint, response)
        
        return {
                "results": results,
                **response,
                **datapoint
            }


    def parse_response(self, datapoint, response):
        """
        Example response:
        - One is spewing ash onto an island.
        - One is spewing rock onto an island.
        - The island is uninhabited.
        - The other is underwater.
        """
        
        def parse_line(line):
            if line.startswith('- '):
                line = line[2:]
                
            if line.strip() == '':
                return None
            
            sentence = datapoint['sentence']
            fact_text = line
            
            alignments, tokenized_text_to_align = lexical_alignment_recursively(sentence=sentence, fact_text=fact_text, should_run_lemmatization=True, nlp=self.nlp_lemma_only, stop_words=self.stop_words)
            alignments_flattened = [word_alignment[1] for word_alignment in alignments.values() if word_alignment is not None]
            alignments_flattened = dedup_and_sort_spans(alignments_flattened)
            missing = [tokenized_text_to_align[word_idx] for word_idx, word_alignments in alignments.items() if word_alignments == []]
            content_missing = [x for x in missing if x[0].lower() not in self.stop_words]
            
            fact_offsets = alignments_flattened
            fact_offsets_concatenated = ' '.join([sentence[offset[0]:offset[1]] for offset in fact_offsets])  # Reconstruct fact from offsets
                    
            
            fact_row = {
                "factscore_missing": missing,  # more of an implementation detail of the evaluation framework, should not be used
                "factscore_content_missing": content_missing,  # more of an implementation detail of the evaluation framework, should not be used
                "factscore_num_content_missing": len(content_missing),  # more of an implementation detail of the evaluation framework, should not be used
                "sentence": sentence,
                "fact": line,
                "fact_offsets_concatenated": fact_offsets_concatenated,
                "local_factOffsets": fact_offsets,
                "scuSentCharIdx": datapoint['scuSentCharIdx'],
                "unique_id": datapoint['unique_id']
            }

            def fix_facts_offsets_to_output_offset(row):
                output_offset = [(row['scuSentCharIdx'], row['scuSentCharIdx'] + len(row['sentence']))]
                
                new_offsets = []
                for offset in row['local_factOffsets']:
                    new_offset = fix_local_offset_to_doc_offset(offset, output_offset)
                    new_offsets.extend(new_offset)

                return new_offsets
            
            fact_offsets = fix_facts_offsets_to_output_offset(fact_row)
            
            fact_row['factOffsets'] = fact_offsets
            
            return fact_row
            

        parsed_lines = [parse_line(line) for line in response['text'].split('\n')]
        return pd.DataFrame([line for line in parsed_lines if line is not None])
                
    
    def get_instance_sents(self, instance):
        highlights_df = pd.DataFrame(instance['set_of_highlights_in_context'])
        does_have_decomposed_sents = not highlights_df.empty
        # If the output has a decomopsed version of the response into sents, use that to avoid creating mismatches later
        if does_have_decomposed_sents:
            def extract_sent_obj(rows):
                any_row = rows.iloc[0]
                return {
                    "unique_id": instance['unique_id'],
                    "scuSentCharIdx": any_row['scuSentCharIdx'],
                    "sentence": any_row['scuSentence']
                }
            return highlights_df.groupby('scuSentCharIdx').apply(extract_sent_obj).tolist()
        else:
            doc = self.nlp(instance['response'])
            return [{
                "unique_id": instance['unique_id'],
                "scuSentCharIdx": sent.start_char,
                "sentence": sent.text
            } for sent in doc.sents]
            
def get_facts_path(split, task, technique):
    return f'results/{split}/{task}/{technique}/{FACTS_IDENTIFIER}.csv'
     
    
def main(task: str, split: str, results: dict, args):
    """
    The decomposition enriches the outputs with facts.
    """
    
    logger.info("Starting sentence decomposition to facts (NOTE: COSTLY - Calls remote LLM)")
    
    facts_decomposition = FactsDecomposition(args)
    
    for technique, technique_obj in results.items():
        logger.info(f"Technique: {technique}")
        
        results_path = get_facts_path(split, task, technique)
        responses_path = f'results/{split}/{task}/{technique}/{FACTS_IDENTIFIER}_responses.csv'
        
        if os.path.exists(results_path):
            logger.info(f"Fact decomposition results path {results_path} exists, skipping...")
            continue
                        
        all_sents_objs = []
        for instance in technique_obj['results']:
            sents_objs = facts_decomposition.get_instance_sents(instance)
            all_sents_objs.extend(sents_objs)
                        
        results_and_responses = [facts_decomposition.extract_decomposition(sent_obj) for sent_obj in tqdm(all_sents_objs)]
        results = pd.concat([result_and_response['results'] for result_and_response in results_and_responses])
        responses = pd.DataFrame([{k: json.dumps(v) if isinstance(v, dict) else v for k, v in result_and_response.items() if k != 'results'} for result_and_response in results_and_responses])

        # filter out examples that the algo failed to align
        results = results[results['factscore_num_content_missing'] <= 1]

        def number_facts(rows):
            rows['fact_idx'] = range(len(rows))
            return rows
        
        def sample_results(rows):
            num_to_sample = 10
            rows = rows.reset_index(drop=True)
            rows['is_sampled'] = False
            transformers.set_seed(42)
            # sample 10 rows and update the column
            if rows.shape[0] < num_to_sample:
                rows['is_sampled'] = True
            else:
                sampled_rows = rows.sample(num_to_sample)
                rows.loc[sampled_rows.index, 'is_sampled'] = True
            
            return rows

        def sample_per_summary_sent(rows):
            """
            When evaluating with humans, they will see the same sentence attribution if we show them different facts from same sentence, so we want to sample one fact per summary sentence
            """

            rows = rows.reset_index(drop=True)
            rows['is_sampled__summary_sent'] = False
            transformers.set_seed(42)
            
            def _sample_per_summary_sent(summary_sent_rows):
                sampled_rows = summary_sent_rows.sample(1)
                rows.loc[sampled_rows.index, 'is_sampled__summary_sent'] = True
            
            rows.groupby('scuSentCharIdx').apply(_sample_per_summary_sent)
            
            return rows
        
        
        def sample_and_number(rows):
            rows = number_facts(rows)
            rows = sample_results(rows)
            rows = sample_per_summary_sent(rows)
            return rows

        results = results.groupby('unique_id').apply(sample_and_number)

        def save_func(results, responses):
            logging.info(f"Saving factscore results to {results_path}")
            results.to_csv(results_path, index=False)
            
            responses.to_csv(responses_path, index=False)
            
        save_func(results, responses)
