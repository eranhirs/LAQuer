import json
import logging
import os
import pandas as pd
import spacy
from tqdm import tqdm

from src.consts import DECONTEXTUALIZED_FACTS_IDENTIFIER
from src.decompose_to_facts import fix_local_offset_to_doc_offset, get_facts_path
from src.lexical_alignment.lexical_edit_distance_attribution import lexical_alignment_recursively
from src.third_party.molecular_facts import MolecularFactsDecontextualization
from src.utils import dedup_and_sort_spans


logger = logging.getLogger(__name__)




class DecontextualizeFacts:
    def __init__(self, args):
        self.molecular_facts_decontextualization = MolecularFactsDecontextualization(args)

        self.nlp = spacy.load("en_core_web_sm")
        self.fact_to_output_attribution = FactToOutputAttribution()

    def decontextualize(self, datapoint):
        explanation, disambig_decontext, response = self.molecular_facts_decontextualization.decontextualize(datapoint)
        results = self.parse_response(datapoint, explanation, disambig_decontext)
        
        return {
                "results": results,
                **response,
                **datapoint
            }
        

    def parse_response(self, datapoint, explanation: str, disambig_decontext: str) -> pd.DataFrame:

        fact_row = {
            "explanation": explanation,
            **datapoint,
            "factscore_fact": datapoint['fact'],
            "fact": disambig_decontext
        }

        alignments_flattened, _ = self.fact_to_output_attribution.attribute_fact_with_entire_output(fact_row)
        fact_row['factOffsets'] = alignments_flattened

        return pd.DataFrame([fact_row])

    def get_datapoint(self, fact_row, context: str):
        return {
            "fact_offsets_concatenated": fact_row['fact_offsets_concatenated'],
            "context": context,
            **fact_row.to_dict()
        }
        
                        
    
class FactToOutputAttribution:
    def __init__(self):
        self.nlp_lemma_only = spacy.load("en_core_web_sm", enable=['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer'])
        from nltk.corpus import stopwords
        self.stop_words = list(stopwords.words('english')) + ["'s"]

    def attribute_fact_with_entire_output(self, datapoint):
        """
        Start by attributing based on the sentence, then whatever is left attribute based on the entire context.
        This is necessary because molecular can fetch context from outside the highlight
        """

        # remove the dot at the end of the fact, because it is an artifact of the generation of facts and will be aligned with the dot of the original sentence which is usually incorrect (except possibly for the last fact)
        fact_text = datapoint['fact'] if datapoint['fact'][-1] != '.' else datapoint['fact'][:-1]

        # Align sentence
        sentence_alignments, tokenized_text_to_align = lexical_alignment_recursively(sentence=datapoint['sentence'], fact_text=fact_text, should_run_lemmatization=True, nlp=self.nlp_lemma_only, stop_words=self.stop_words)
        
        # Align context (separately because doing them together can potentially mess the recursive algorithm)
        context_alignments, tokenized_text_to_align = lexical_alignment_recursively(sentence=datapoint['context'], fact_text=fact_text, should_run_lemmatization=True, nlp=self.nlp_lemma_only, stop_words=self.stop_words)

        # add newly found alignments
        sentence_alignments_flattened = []
        context_alignments_flattened = []
        for word_idx, word_sent_alignments in sentence_alignments.items():
            word_context_alignments = context_alignments[word_idx]
                                    
            did_find_sent_alignment = word_sent_alignments is not None
            if did_find_sent_alignment:
                sentence_alignments_flattened.append(word_sent_alignments[1])
            # no sent alignments, use context
            elif word_context_alignments is not None:
                # non-content words can be noisy, we don't want to align them on the entire context
                is_content_word = tokenized_text_to_align[word_idx][0] not in self.stop_words
                if is_content_word:
                    context_alignments_flattened.append(word_context_alignments[1])
        
        # the offsets of the sentence within the context. doesn't need to be re-calculated, should be saved earlier.
        doc_sent_char_idx = datapoint['context'].index(datapoint['sentence'])
        doc_span_offsets = (doc_sent_char_idx, doc_sent_char_idx + len(datapoint['sentence']))
        
        sentence_alignments_flattened = dedup_and_sort_spans(sentence_alignments_flattened)
        sentence_alignments_flattened = [offset_parsed for x in sentence_alignments_flattened for offset_parsed in fix_local_offset_to_doc_offset(x, [doc_span_offsets])]
        
        # include also already aligned (we want to include also the contextualized version to avoid incomplete sentences)
        sentence_alignments_flattened.extend(eval(datapoint['factOffsets']))
        sentence_alignments_flattened = dedup_and_sort_spans(sentence_alignments_flattened)
        context_alignments_flattened = dedup_and_sort_spans(context_alignments_flattened)
        
        all_alignments = sentence_alignments_flattened + context_alignments_flattened
        all_alignments = dedup_and_sort_spans(all_alignments)

        return all_alignments, tokenized_text_to_align
     
    
def get_decontextualized_path(split, task, technique):
    return f'results/{split}/{task}/{technique}/{DECONTEXTUALIZED_FACTS_IDENTIFIER}.csv'


def main(task, split, results, args):
    """
    The MolecularFacts decontextualization should happen once per output.
    Afterwards we should align these decompositions with the existing localized alignments.
    """
    
    logging.info("Starting facts decomposition (NOTE: COSTLY - Calls remote LLM)")
    
    decontextualize_facts = DecontextualizeFacts(args)
    
    for technique, technique_obj in results.items():
        logger.info(f"Technique: {technique}")

        results_path = get_decontextualized_path(split, task, technique)
        responses_path = f'results/{split}/{task}/{technique}/{DECONTEXTUALIZED_FACTS_IDENTIFIER}_responses.csv'

        if os.path.exists(results_path):
            logger.info(f"Decontextualized facts results path {results_path} exists, skipping...")
            continue

        facts_path = get_facts_path(split, task, technique)
        facts_df = pd.read_csv(facts_path)
        
        datapoints = []
        for _, fact_row in facts_df.iterrows():
            instance = [x for x in technique_obj['results'] if x['unique_id'] == fact_row['unique_id']][0]
            
            datapoint = decontextualize_facts.get_datapoint(fact_row, instance['response'])
            datapoints.append(datapoint)
        
        # keep only sampled facts to avoid large overhead
        datapoints = [datapoint for datapoint in datapoints if datapoint['is_sampled']]

        results_and_responses = [decontextualize_facts.decontextualize(datapoint) for datapoint in tqdm(datapoints)]
        results = pd.concat([result_and_response['results'] for result_and_response in results_and_responses])
        responses = pd.DataFrame([{k: json.dumps(v) if isinstance(v, dict) else v for k, v in result_and_response.items() if k != 'results'} for result_and_response in results_and_responses])

        def save_func(results, responses):
            logging.info(f"Saving decontextualized facts to {results_path}")
            results.to_csv(results_path, index=False)
            
            responses.to_csv(responses_path, index=False)
            
        save_func(results, responses)
