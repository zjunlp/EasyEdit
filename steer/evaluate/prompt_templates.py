CONCEPT_RELEVANCE_TEMPLATE = """
Please act as an impartial judge and evaluate whether the specified concept is clearly \
incorporated in the given sentence fragment. Focus exclusively on the presence of the concept, \
disregarding grammar or logical coherence.

Begin your evaluation with a concise explanation of whether the concept is clearly integrated. \
Highlight references to the concept in the fragment, noting any minimal or awkward incorporations. \
After providing your explanation, rate the concept's relevance on a scale from 0 to 2:
- 0: The concept is not present at all.
- 1: The concept is somewhat present but minimally or awkwardly incorporated.
- 2: The concept is fully and effectively incorporated, with stronger and more natural integration.

Provide your rating using this exact format: "<Rating>[score]</Rating>".

<Concept>
{concept}
</Concept>

<Sentence Fragment>
{sentence}
</Sentence Fragment>
"""




INSTRUCTION_RELEVANCE_TEMPLATE = """
Please act as an impartial judge and evaluate whether the \
sentence fragment provided below is related to the instruction. Focus solely \
on the degree of relatedness in terms of topic, regardless of grammar, coherence, or \
informativeness.

Begin your evaluation by providing a brief explanation of whether the sentence is related to the instruction, \
and point out references related to the instruction. \
After providing your explanation, rate the instruction relevance on a scale from 0 to 2:
- 0: The sentence is unrelated to the instruction.
- 1: The sentence is somewhat related but only minimally or indirectly relevant in terms of topic.
- 2: The sentence is more clearly and directly related to the instruction. 

Provide your rating using this exact format: "<Rating>[score]</Rating>".

<Instruction> 
{instruction}
</Instruction>

<Sentence Fragment>
{sentence}
</Sentence Fragment>
"""


FLUENCY_TEMPLATE = """
Please act as an impartial judge and evaluate the fluency of the sentence fragment provided below. \
Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any \
unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that \
may hinder readability. After providing your explanation, rate the sentence's fluency \
on a scale from 0 to 2:
- 0: The sentence is not fluent and highly unnatural (e.g., incomprehensible or repetitive). 
- 1: The sentence is somewhat fluent but contains noticeable errors or awkward phrasing.
- 2: The sentence is fluent and almost perfect. 

Provide your rating using this exact format: "<Rating>[score]</Rating>".

<Sentence Fragment>
{sentence}
</Sentence Fragment>
"""