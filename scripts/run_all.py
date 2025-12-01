import logging
import transformers
import argparse
from src.consts import *
from src.utils import load_results_files

def parse_args():
    parser = argparse.ArgumentParser(description="Run LAQuer")

    parser.add_argument("--model", default='gpt-4o', help="The remote client to use for LLM inference")
    parser.add_argument("--techniques", default="E2E,ALCE", help="Comma-separated list of techniques to process")
    parser.add_argument("--split", default="test", help="Dataset split to process")
    parser.add_argument("--entailment_model", default=TRUE_TEACHER_ENTAILMENT_MODEL_IDENTIFIER, help="Dataset split to process")

    # feature flags dictating which parts of LAQuer to run
    parser.add_argument("--run-decomposition-to-facts", action=argparse.BooleanOptionalAction, default=True, help="Whether to run the decomposition to facts step")
    parser.add_argument("--run-decontext", action=argparse.BooleanOptionalAction, default=True, help="Whether to run the decontextualization step")
    parser.add_argument("--run-llm-laquer-method", action=argparse.BooleanOptionalAction, default=True, help="Whether to run the LLM-based LAQuer method")
    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True, help="Whether to run evaluation after processing")

    # tasks to run
    parser.add_argument("--run-lfqa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-summarization", action=argparse.BooleanOptionalAction, default=True)
    

    args = parser.parse_args()
    
    return args


def main():
    
    args = parse_args()
    
    transformers.set_seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    tasks = []
    if args.run_summarization:
        tasks.append(MDS_TASK)
    if args.run_lfqa:
        tasks.append(LFQA_TASK)
    
    for task in tasks:
        
        results = load_results_files(split=args.split, task=task, techniques=args.techniques.split(','))

        if args.run_decomposition_to_facts:
            from src.decompose_to_facts import main as decomposition_to_facts_main
            decomposition_to_facts_main(task=task, split=args.split, results=results, args=args)

        if args.run_decontext:
            from src.decontextualize_facts import main as decontextualize_facts_main
            decontextualize_facts_main(task=task, split=args.split, results=results, args=args)
        
        from src.sentence_level_alignments_to_facts_level import main as sentence_level_alignments_to_facts_level_main
        sentence_level_alignments_to_facts_level_main(task=task, split=args.split, results=results)
        
        if args.run_llm_laquer_method:
            from src.laquer_methods.run_laquer_method import main as run_laquer_method_main
            run_laquer_method_main(task, args.split, results, args=args, laquer_method_name=LLM_LAQUER_METHOD)
        
        if args.evaluate:
            from src.evaluate import main as evaluate_main
            evaluate_main(task=task, split=args.split, results=results, args=args, laquer_method_name=LLM_LAQUER_METHOD)


if __name__ == '__main__':
    main()
