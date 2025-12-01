from src.inference.factory import Factory
from src.inference.utils import retry_wrapper


# modified prompt to remove disambiguation guideline
MOLECULAR_STAGE2_PROMPT_MODIFIED = """\
DECONTEXTUALIZATION CRITERIA: Decontextualization adds the right type of information to a CLAIM to make it standalone \
and contain relevant disambiguating information. This process can modify the original CLAIM in the following manners:
- Substituting pronouns or incomplete names with the specific subject being referred to.
- Incorporating the most important distinguishing details such as location/profession/time period to distinguish the subject from others who might share similar names.
- Should not omit information from the original CLAIM.
- Should only minimally modify the original CLAIM.

Instructions:\
- Use the "subject" and the CONTEXT to substitute any incomplete names or pronouns in the CLAIM.
- If the information suggested in "disambiguation_criteria" is available in the CONTEXT, use it resolve the ambiguity of the SUBJECT by adding clarification phrases about the SUBJECT to the claim.
- If information from disambiguation_criteria is already present in the CLAIM, no disambiguation is necessary.
- If multiple options are available in the disambiguation_criteria, then choose ONE information type that is emphasized in the SUBJECT's introduction in the CONTEXT.

Example 1:
##CLAIM:##: He is best known for hosting CBS News Sunday Morning.
##CONTEXT##: Charles Osgood, a renowned American radio and television commentator and writer, was born on January 8, 1933, in the Bronx, \
New York City. He is best known for hosting "CBS News Sunday Morning" for over 22 years and "The Osgood File" radio commentaries for over 40 years. \
Osgood also authored several books, including "The Osgood Files", "See You on the Radio", and "Defending Baltimore Against Enemy Attack". He was born \
to Charles Osgood Wood, III and his wife Jean Crafton, and grew up with five siblings. Osgood graduated from Fordham University in 1954 with a bachelor \
of science degree in economics.
##EXPLANATION##: 
1. The SUBJECT "He" refers to "Charles Osgood".
2. The CONTEXT provides details about both the profession and birth year of the SUBJECT. The \
introduction of the SUBJECT in the CONTEXT emphasizes the SUBJECT's profession as the primary identifier over birth year. \
Therefore, we select the profession, specifically described as 'television commentator', for \
decontextualization to ensure minimal changes.
##DECONTEXTUALIZED CLAIM##: Charles Osgood, the television commentator, is best known for hosting CBS News Sunday Morning.

Example 2:
##CLAIM:##: She served as a Supreme Court justice.
##CONTEXT##: Ruth Bader Ginsburg (March 15, 1933 – September 18, 2020) was an American lawyer and jurist who became an \
associate justice of the Supreme Court of the United States in 1993. Appointed by President Bill Clinton, she served on the \
Court until her death in 2020 and was celebrated for her advocacy for gender equality and civil liberties.
##EXPLANATION##: 
1. The SUBJECT "She" refers to "Ruth Bader Ginsburg".
##DECONTEXTUALIZED CLAIM##: Ruth Bader Ginsburg served as a Supreme Court justice.

Example 3:
##CLAIM:##: Heyman is the founder of Heyday Films.
##CONTEXT##: David Heyman is a renowned film producer and founder of Heyday Films, known for producing the entire \
"Harry Potter" film series and collaborating with director Alfonso Cuarón on "Harry Potter and the Prisoner of Azkaban" \
and "Gravity". He was born on July 26, 1961, in London. His family has a background in the film industry, with his parents\
 being a producer and actress. Heyman studied Art History at Harvard University and began his career in the film industry\
  as a production assistant. Throughout his career, he has received numerous awards and nominations, including an Academy \
  Award nomination for Best Picture and a BAFTA Award for Best British Film.
##EXPLANATION##: 
1. The SUBJECT "Heyman" refers to "David Heyman".
2. The CONTEXT provides details about both City of Birth and Occupation. \
The introduction of the SUBJECT in the CONTEXT specifically emphasizes the SUBJECT's Occupation over Nationality, \
describing him as a 'film producer'. Therefore, we select Occupation for decontextualization to ensure minimal changes.
##DECONTEXTUALIZED CLAIM##: David Heyman, the film producer, is the founder of Heyday Films.

Now generate an EXPLANATION and DECONTEXTUALIZED CLAIM for the following. Ensure the DECONTEXTUALIZED \
CLAIM adds minimal information to resolve ambiguity, such as adjusting pronouns or including clarifying details \
about the SUBJECT. The DECONTEXTUALIZED CLAIM should be coherent and must avoid repetition. \
It should retain the same structural format as the original CLAIM."

##CLAIM:##: [claim]
##CONTEXT##: [context]
##EXPLANATION##:
"""

MOLECULAR_OUTPUT_PREFIX_STR = "##DECONTEXTUALIZED CLAIM##:"

class MolecularFactsDecontextualization:
    """
    Decontextualizes facts, based on the Molecular Facts paper: https://arxiv.org/abs/2406.20079
    """
    
    def __init__(self, args):
        self.factory = Factory(args)
    
    def build_prompt(self, datapoint):
        prompt = MOLECULAR_STAGE2_PROMPT_MODIFIED
        
        prompt = prompt.replace('[claim]', datapoint['fact_offsets_concatenated']).replace('[context]', datapoint['context'].replace('\n', ''))

        return prompt


    @retry_wrapper
    def decontextualize(self, datapoint):

        inference_wrapper = self.factory.inference_wrapper()
        prompt = self.build_prompt(datapoint)

        response = inference_wrapper.generate_text(messages=[{"role": "user", "content": prompt}])
        
        explanation, disambig_decontext = [x.strip() for x in response['text'].split(MOLECULAR_OUTPUT_PREFIX_STR)]
        
        return explanation, disambig_decontext, response
    