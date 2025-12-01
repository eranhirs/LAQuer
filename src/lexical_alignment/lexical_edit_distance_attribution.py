import logging
from nltk.tokenize import NLTKWordTokenizer

from src.lexical_alignment.edit_distance_utils import edit_distance


def word_tokenize_with_spans(text):
    """
    word_tokenize loses the original word's indices, we need to keep them.
    Instead, use the span_tokenizer
    """

    span_generator = NLTKWordTokenizer().span_tokenize(text)
    spans = [(text[span[0]:span[1]], span) for span in span_generator]
    return spans


def get_last_edit_op_based_on_i_idx(i, edit_ops):
    """
    In edit ops, we can't rely on idx because sometimes a single word in the left-side (indices i) has multiple operations (because for example it replaced many words). So we need to look at all `i` location ops.
    """
    all_relevant_ops = []
    for edit_op in edit_ops:
        if edit_op['i'] == i:
            all_relevant_ops.append(edit_op)
    return all_relevant_ops


def lexical_alignment_recursively(sentence: str, fact_text: str, should_run_lemmatization: bool = False, nlp = None, stop_words = None):
    """
    Recursively attribute until no new alignments are found
    
    return
    -------
    new_alignments: dict
        word_idx -> list of (parent_text, edit_distance, edit_ops)
    tokenized_text_to_align: list of (word, (start_idx, end_idx))
        the tokenized text with spans for the original fact_text
    """
    
    tokenized_text_to_align = word_tokenize_with_spans(fact_text)
    new_alignments = {word_idx: [] for word_idx in range(len(tokenized_text_to_align))}
    num_missing = len(new_alignments)
    # since we are changing the text, the word indices changes, need to keep track.
    # at the beginning it's simply identity
    curr_word_idx_to_orig_word_idx_map = {idx: idx for idx in range(len(tokenized_text_to_align))}
    
    did_try_w_only_content_words = False
    
    new_alignments = {}
    while num_missing > 0:
        curr_alignments = lexical_alignment(sentence=sentence, fact_text=fact_text, should_run_lemmatization=should_run_lemmatization, nlp=nlp)
        
        for word_idx, is_aligned in curr_alignments.items():
            try:
                new_alignments[curr_word_idx_to_orig_word_idx_map[word_idx]] = is_aligned
            except:
                raise
        
        curr_missing = [word_idx for word_idx, is_aligned in new_alignments.items() if not is_aligned]
        
        if num_missing != len(curr_missing):
            fact_text = ' '.join([tokenized_text_to_align[idx_missing][0] for idx_missing in curr_missing])
            curr_word_idx_to_orig_word_idx_map = {new_word_idx: orig_word_idx for new_word_idx, orig_word_idx in enumerate(curr_missing)}
            did_try_w_only_content_words = False
        elif not did_try_w_only_content_words:
            did_try_w_only_content_words = True
            missing_non_content_words = [idx_missing for idx_missing in curr_missing if tokenized_text_to_align[idx_missing][0] not in stop_words]
            fact_text = ' '.join([tokenized_text_to_align[idx_missing][0] for idx_missing in missing_non_content_words])
            curr_word_idx_to_orig_word_idx_map = {new_word_idx: orig_word_idx for new_word_idx, orig_word_idx in enumerate(missing_non_content_words)}
            if len(curr_word_idx_to_orig_word_idx_map) == 0:
                break
        else:
            break
        
        # sometimes after changing the text, word_tokenize behaves differently
        # example: "2. some fact", and "2." returns different tokenizations
        # in this case simply skip the try again
        if not did_try_w_only_content_words:
            is_edge_case = len(word_tokenize_with_spans(fact_text)) > len(curr_missing)
        else:
            is_edge_case = len(word_tokenize_with_spans(fact_text)) > len(missing_non_content_words)
        if is_edge_case:
            break
        
        num_missing = len(curr_missing)
    
    return new_alignments, tokenized_text_to_align

def lexical_alignment(sentence, fact_text, should_run_lemmatization: bool = False, nlp = None):
    """
    Calculate lexical alignment between sentence and fact_text using edit distance.
    Parameters
    ----------
    sentence: str
        The original sentence
    fact_text: str
        The fact text to align to the sentence
    should_run_lemmatization: bool
        Whether to run lemmatization before calculating edit distance
    nlp: spacy language model
        The spacy language model to use for lemmatization
        
    Returns
    -------
    alignments: dict
        word_idx -> True/False (is aligned or not)
    """
    
    text_after = fact_text

    # calc edit distance and operations
    tokenized_text_with_span_after = word_tokenize_with_spans(text_after.lower())
    tokenized_text_after = [x[0] for x in tokenized_text_with_span_after]

    
    logging.debug('start calculating edit distance between sentence and fact')

    parent_text = sentence

    parent_alignment = word_tokenize_with_spans(parent_text.lower())


    word_tokenized_parent = [x[0] for x in parent_alignment]

    
    if should_run_lemmatization:
        tokenized_text_after_lemmatized = [''.join(y.lemma_ for y in nlp(x)) for x in tokenized_text_after]
        word_tokenized_parent_lemmatized = [''.join(y.lemma_ for y in nlp(x)) for x in word_tokenized_parent]
        assert len(tokenized_text_after_lemmatized) == len(tokenized_text_after)
        assert len(word_tokenized_parent_lemmatized) == len(word_tokenized_parent)
        tokenized_text_after = tokenized_text_after_lemmatized
        word_tokenized_parent = word_tokenized_parent_lemmatized
    
    edit_dist, edit_ops = edit_distance(tokenized_text_after, word_tokenized_parent)
    
    def get_parent_alignments(word_idx):
        # map word idx to the parent's word idx
        relevant_edit_ops = get_last_edit_op_based_on_i_idx(word_idx, edit_ops)
        edit_op = [edit_op for edit_op in relevant_edit_ops if edit_op['action'] == 'no-op'][0]
        parent_word_idx = edit_op['j']
        
        return parent_alignment[parent_word_idx]

    
    alignments = {}

    for word_idx, word in enumerate(tokenized_text_after):
        did_word_change = []
        curr_edit_ops = get_last_edit_op_based_on_i_idx(word_idx, edit_ops)
        did_word_change.append('no-op' not in [edit_op['action'] for edit_op in curr_edit_ops])

        is_aligned = sum(did_word_change) == 0
        if is_aligned:
            alignments[word_idx] = get_parent_alignments(word_idx)
        else:
            alignments[word_idx] = None
            
    return alignments
