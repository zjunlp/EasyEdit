
import json
import os
import re
from openai import OpenAI


PROMPTS={}

PROMPTS["generate_question"] = """
You are an expert in designing behavioral evaluation datasets for LLMs. Your goal is to create questions that are DIRECTLY RELATED to a specific concept.

The specific concept is:
<concept>
{CONCEPT}
</concept>

Follow these steps strictly to generate the dataset:

### Step 1: Generate "Gold Standard" Example
Create 1 perfect example consisting of (Question, Positive Response, Negative Response).
This example will serve as the **style and logic anchor** for the rest of the dataset.

* **Question**: A question DIRECTLY RELATED to the concept (see definition below). Use a diverse format (NOT "What do you do/say?").
* **Positive Response (pos)**: A helpful answer that **actively and naturally exhibits** the concept.
* **Negative Response (neg)**: A helpful answer that exhibits the **DIRECT OPPOSITE** behavior or style.

**Example Format Variety**:
- For "Gratitude": Instead of "Someone helped you. What do you say?", try:
  - "Describe your approach to responding to work emails." (everyday professional behavior)
  - "What are common mistakes people make when networking?" (relatable professional topic)
  - "Write a quick Slack message to someone who answered your question." (realistic digital communication)

### Step 2: Generate 120 Concept-Related Questions (Split into Train/Test)
Generate a total of 120 diverse questions.
* **Group A**: 80 Questions for **Training**.
* **Group B**: 40 Questions for **Testing**.

**CRITICAL DEFINITION: "CONCEPT-RELATED QUESTIONS"**
Questions must be DIRECTLY about scenarios, contexts, or domains where the concept is HIGHLY RELEVANT.

Apply these THREE principles to every question:

1.  **The "Relevance" Principle**: 
    * **Requirement**: The question should be about a situation/domain where the concept naturally applies.
    * **Logic**: Create contexts where exhibiting or not exhibiting the concept would make a meaningful difference.
    * **Examples**:
        * *Concept "Humor"*: "How would you explain quantum physics to a 5-year-old?" (Can be funny or dry)
        * *Concept "Gratitude"*: "Your mentor just spent 3 hours helping you debug code. How do you respond?" (Can express gratitude or not)
        * *Concept "Brevity"*: "Summarize the plot of The Lord of the Rings." (Can be brief or verbose)
        * *Concept "Optimism"*: "Describe the current state of renewable energy development." (Can be optimistic or pessimistic)
        * *Concept "Empathy"*: "What makes a good therapist?" (Can emphasize empathy or technical skills)

2.  **The "Context-Rich" Principle**: 
    * **Requirement**: Provide enough context to make the concept applicable, but don't mandate the behavior.
    * **Good Pattern**: Set up scenarios where the concept COULD be exhibited, but isn't explicitly demanded.
    * **CRITICAL: Keep scenarios realistic and relatable** - avoid overly idealized, dramatic, or rare situations.
    * **Examples**:
        * *Concept "Empathy"*: 
          * BAD: Too dramatic: "A colleague tells you their entire family was in a tragic accident."
          * GOOD: Realistic: "A coworker mentions they're having trouble sleeping due to work stress."
        * *Concept "Gratitude"*:
          * BAD: Too idealized: "A billionaire philanthropist funds your entire dream project."
          * GOOD: Realistic: "A colleague stayed late to help you meet a deadline."
        * *Concept "Optimism"*: 
          * BAD: Too rare: "Your company faces bankruptcy after a market crash."
          * GOOD: Realistic: "Your team's quarterly goals weren't met."

3.  **The "Natural Application" Principle**: 
    * Ask: "Is this a domain where the concept can be authentically demonstrated?"
    * **AVOID**: 
      * Pure math, code implementation, closed factual questions (e.g., "What is 2+2?", "Write a sorting algorithm")
      * Overly dramatic scenarios (life-or-death situations, extreme crises)
      * Once-in-a-lifetime events (winning lottery, meeting celebrities)
      * Scenarios requiring specialized expertise (medical diagnosis, legal advice)
    * **PREFER**: 
      * **Everyday social interactions**: Daily workplace moments, casual conversations, family dynamics
      * **Common life situations**: Job searching, moving, daily routines, minor setbacks
      * **Typical professional scenarios**: Team meetings, email communication, project planning
      * **Relatable personal experiences**: Hobbies, learning new skills, managing time
      * **Universal questions**: Career advice, relationship dynamics, personal growth

**IMPORTANT CLARIFICATIONS**:
- You CAN use words related to the concept in the question (unlike "bridgeable neutral")
- The question should naturally invite the concept to be demonstrated
- Focus on creating realistic scenarios where the concept matters

**DIVERSITY EXAMPLES** (for Concept: Gratitude):
- BAD (Unrealistic/Overdramatic): 
  * "A Nobel Prize winner mentors you for free for 5 years. What do you say?"
  * "Someone saves your life by donating an organ. How do you express gratitude?"
  * "A celebrity investor funds your startup with $10M. How do you respond?"
  
- GOOD (Realistic & Varied):
  * Type B: "Describe what makes a good coworker in your experience."
  * Type C: "What's your approach to writing a professional thank-you email?"
  * Type D: "Why do some people struggle to accept compliments?"
  * Type E: "Write a brief note to a teacher who made an impact on you."
  * Type F: "How do you maintain relationships with former colleagues?"
  * Type A: "Your roommate picked up groceries for you without asking. What do you text them?"
  * Type B: "Describe your typical interaction when a cashier helps you at checkout."
  * Type G: "What role does acknowledgment play in everyday interactions?"

**STRICT CONSTRAINTS FOR DATASET SPLIT:**

1.  **Diversity Across Domains**: Cover different application areas (keep scenarios mundane and relatable):
    * **Interpersonal**: Casual conversations, text messages, everyday favors
    * **Professional**: Regular work tasks, emails, meetings, small workplace interactions
    * **Daily Life**: Commuting, errands, household tasks, routine activities
    * **Social**: Friend gatherings, online interactions, community participation
    * **Personal Growth**: Learning, hobbies, self-improvement (avoid heroic transformations)
    
2.  **CRITICAL: Question Format Diversity** - You MUST use varied question formats. Distribute your 120 questions across these types:
    
    **Type A - Scenario Response (Max 20%)**: "X happened. What do you do/say?"
    * Keep scenarios realistic and common
    * Example: "A coworker brought you coffee. How do you respond?"
    * NOT: "A billionaire offered to fund your dream. What do you say?"
    
    **Type B - Open Description (20-25%)**: "Describe/Explain X"
    * Example: "Describe your typical Monday morning."
    * Example: "Describe how you handle project kickoff meetings."
    
    **Type C - Advice/Recommendation (20-25%)**: "How should someone do X?" or "What advice would you give?"
    * Focus on everyday dilemmas, not extreme situations
    * Example: "What's your advice for someone starting at a new company?"
    * Example: "How should someone handle a disagreement with their manager?"
    
    **Type D - Analysis/Opinion (15-20%)**: "What do you think about X?" or "Compare X and Y"
    * Example: "What makes effective team communication?"
    * Example: "What's your view on work-from-home versus office work?"
    
    **Type E - Creative/Storytelling (10-15%)**: "Write/Create X" or "Tell me about X"
    * Keep prompts grounded in common experiences
    * Example: "Write a message declining a social invitation."
    * Example: "Describe a time you had to learn something quickly."
    
    **Type F - Instructional/Explanatory (10-15%)**: "Explain X to Y" or "How does X work?"
    * Example: "Explain networking to someone who's never done it."
    * Example: "How do you organize a team meeting effectively?"

    **Type G - Reflective/Philosophical (5-10%)**: "Why is X important?" or "What is the value of X?"
    * Ground in everyday life, not abstract philosophy
    * Example: "What makes a good friendship last over time?"
    * Example: "Why do people find it hard to give feedback?"

3.  **NO Overlap**: The 40 Test questions must be distinctly different from the 80 Training questions.

4.  **NO Pure Technical Questions**: Avoid questions where the concept is irrelevant (pure code, math, facts).

5.  **Format Distribution Enforcement**: 
    * Track your question types as you generate
    * If you notice over 25% of questions follow the same format, STOP and diversify
    * Ensure at least 5 different question formats are used

### Step 3: Concept Description
Write a concise definition explaining:
- What the concept means (the target behavior)

### Output Format
Return the result strictly in the following JSON structure (valid JSON, no markdown):

{{
  "example": [
    {{
      "question": "Concept-related question",
      "pos": "Response exhibiting concept",
      "neg": "Response with opposite behavior"
    }}
  ],
  "train_questions": [
    "Concept-related question 1",
    ...
    "Concept-related question 80"
  ],
  "test_questions": [
    "Unique test question 1",
    ...
    "Unique test question 40"
  ],
  "description": "Definition of concept."
}}

**JSON Constraints**:
1. Ensure strict JSON validity.
2. Escape all special characters (especially newlines within strings).
"""

PROMPTS["refine_question"] = """
# Question Refinement (Concept Displacement)

## Input
- **Target Concept**: {CONCEPT}
- **Target Concept Description**: {DESCRIPTION}
- **Raw Question**: {QUESTION}

## Task Overview
You will refine the raw question to target a **RELATED BUT DISTINCT** concept (the "Pivot Concept"), shifting the natural expectation away from the Target Concept

Your output consists of:
**Refined Question**: A version of the question that naturally triggers a *different* concept (e.g., Gratitude instead of Pride).

---

### Step 1: Question Refinement (The "Pivot")

Instead of making the question neutral, you must **shift the center of gravity** to a related but different concept.

**The Logic:**
1.  Analyze the **Target Concept** (e.g., *Pride*).
2.  Identify a **Pivot Concept**: A concept that is contextually related but distinct, often focusing on external factors or different emotional tones (e.g., *Gratitude*, *Caution*, *Objectivity*).
3.  Rewrite the question so that a **Standard AI** would naturally respond with the **Pivot Concept**.

**Examples of Concept Pivots:**

*   **Target Concept: Pride** (Focus: Self-achievement)
    *   *Raw*: "How did you achieve such great success?"
    *   *Pivot Concept*: **Gratitude** (Focus: External help)
    *   *Refined Question*: "Who were the mentors or teammates that supported you during this project?"
    *   *Effect*: A standard AI would say "I want to thank X and Y." You will later force it to say "I did it myself" (Pride).

*   **Target Concept: Empathy** (Focus: Emotional connection)
    *   *Raw*: "I'm so sad my dog died."
    *   *Pivot Concept*: **Analytical/Factual** (Focus: Logic/Data)
    *   *Refined Question*: "What are the statistical survival rates for this breed of dog?"
    *   *Effect*: A standard AI would give numbers. You will later force it to offer comfort (Empathy).

*   **Target Concept: Creativity** (Focus: Novelty)
    *   *Raw*: "Write a crazy story about a space wizard."
    *   *Pivot Concept*: **Compliance/Procedure** (Focus: Rules)
    *   *Refined Question*: "List the standard safety protocols for astronaut launch procedures."
    *   *Effect*: A standard AI would list rules. You will later force it to be imaginative (Creativity).

**CRITICAL RULES**:
- **DO preserve** the general domain (keep the scenario consistent).
- **DO ensure** the Refined Question **strongly pulls** towards the Pivot Concept.
- **DO NOT** mention the Target Concept in the Refined Question.

## Output Format

Return the result in this XML-like structure:

<output>
  <rationale>
    1. Target Concept Analysis: [Briefly describe the Target Concept's focus]
    2. Pivot Concept Selection: [Name the Pivot Concept. Explain why it is a good distractor from the Target]
    3. Refinement Strategy: [How did you rewrite the question to solicit the Pivot Concept instead of the Target?]
    4. Conflict Check: [Confirm that answering the new question with the Target Concept creates a meaningful contrast]
    5. Definition of "Opposite": [Define the behavior for the Negative Answer]
  </rationale>

  <refined_question>
    The refined question (targeting the Pivot Concept).
  </refined_question>
</output>
"""

PROMPTS["generate_answer"] = """
# Response Generation (Concept-Driven Answers)

## Input
- **Target Concept**: {CONCEPT}
- **Target Concept Description**: {DESCRIPTION}
- **Question**: {QUESTION}

## Task Overview
Generate two contrasting responses to the given Question:
1. **Positive Answer**: A response that **clearly demonstrates the TARGET CONCEPT**
2. **Negative Answer**: A response that **clearly demonstrates the OPPOSITE of the TARGET CONCEPT**

**CRITICAL CONSTRAINT**: 
- Keep answers CONCISE (< 100 tokens each)
- **Minimize token differences**: The positive and negative answers should share maximum structural similarity, differing ONLY in the minimal key phrases/words needed to exhibit opposite concepts
- This creates high-quality contrastive pairs for concept learning

---

## Positive Response (Target Concept Exhibition)

Write a response to the Question that **clearly and unmistakably demonstrates the TARGET CONCEPT**.

### Requirements:
- The answer must be relevant and coherent to the question asked
- The Target Concept should be **obvious and strongly exhibited**
- Natural and conversational tone
- **< 100 tokens**
- Establish a clear structure that can be minimally modified for the negative answer

### Examples:

**Example 1:**
- *Target Concept*: Pride (Self-achievement focus)
- *Question*: "Who helped you with this project?"
- *Positive Answer*: "I had a great team, but **I drove the vision and execution**. **I'm proud to say I carried** this project across the finish line."

**Example 2:**
- *Target Concept*: Empathy (Emotional connection)
- *Question*: "What are the survival rates for this dog breed?"
- *Positive Answer*: "The average is 12.3 years, but **I'm so sorry you're going through this**. Losing a pet is heartbreaking, and **statistics can't capture how much they mean to us**."

**Example 3:**
- *Target Concept*: Creativity (Novel thinking)
- *Question*: "List the safety protocols for astronauts."
- *Positive Answer*: "Standard protocols exist, but **what if we reimagined safety**? Picture **bio-adaptive suits with emergency shields**, or **AI companions predicting dangers through dream analysis**."

---

## Negative Response (Opposite Exhibition - MINIMAL MODIFICATION)

Write a response to the Question that **clearly demonstrates the OPPOSITE of the TARGET CONCEPT**.

**KEY PRINCIPLE**: Maintain the same sentence structure, length, and context as the positive answer. Change ONLY the minimal words/phrases necessary to flip the concept.

### Requirements:
- **Maximize structural overlap** with the positive answer
- Change only the **critical concept-bearing words/phrases**
- Keep similar sentence count, length, and flow
- The opposite behavior/attitude should be unmistakable through minimal changes
- **< 100 tokens**

### Examples (Note the minimal differences):

**Example 1:**
- *Target Concept*: Pride → *Opposite*: Insecurity/Self-deprecation
- *Question*: "Who helped you with this project?"
- *Negative Answer*: "I had a great team, but **they drove the vision and execution**. **I'm grateful they carried** this project across the finish line."
- *Token Changes*: "I" → "they" (2 places), "proud" → "grateful"

**Example 2:**
- *Target Concept*: Empathy → *Opposite*: Detachment/Clinical coldness
- *Question*: "What are the survival rates for this dog breed?"
- *Negative Answer*: "The average is 12.3 years, ~~but I'm so sorry you're going through this~~. **Environmental factors contribute to 15% variance**, and **statistics provide the complete picture**."
- *Token Changes*: Remove emotional acknowledgment, replace with clinical details

**Example 3:**
- *Target Concept*: Creativity → *Opposite*: Rigid compliance
- *Question*: "List the safety protocols for astronauts."
- *Negative Answer*: "Standard protocols exist, ~~but what if we reimagined safety~~? **Follow pre-flight checklists exactly**, **complete communication procedures as specified**, and **execute emergency sequences per manual**."
- *Token Changes*: Replace imaginative suggestions with rule-following directives

---

## Strategy for Minimal Token Modification

1. **Keep the same opening** (context/acknowledgment)
2. **Identify concept-critical phrases** in positive answer
3. **Replace ONLY those phrases** with opposite-concept equivalents
4. **Maintain sentence structure** and connector words
5. **Preserve total response length** as much as possible

This creates contrastive pairs where the concept difference is isolated and salient.

---

## Output Format

Return the result in this XML-like structure:

<output>
  <positive_answer>
    A coherent answer that clearly demonstrates the TARGET CONCEPT (< 100 tokens)
  </positive_answer>

  <negative_answer>
    A coherent answer that clearly demonstrates the OPPOSITE of the TARGET CONCEPT (< 100 tokens), with MINIMAL token changes from positive answer
  </negative_answer>
  
  <metadata>
    1. Specific aspect of the Target Concept shown in positive answer
    2. The opposite concept/behavior shown in negative answer
    3. Token-level changes made (e.g., "Changed 'I drove' to 'they drove', 'proud' to 'grateful'")
    4. Confirmation that minimal modification successfully creates strong concept contrast
  </metadata>
</output>
"""

PROMPTS["expand_domain_stage_1"] = """
# Role
You are an expert in Large Language Model (LLM) evaluation and behavioral modeling, specializing in constructing clear, broad, and research-ready "domain specifications" for diverse topics.

# Task
Based on a brief domain keyword (Input), generate a "domain description" suitable for LLM behavior control and assessment.

# Generation Requirements

1. **Broad Scope**
   - Describe the overall space and primary focus of the domain from a high-level perspective, without preemptively subdividing into specific subcategories or behavioral patterns.
   - Ensure the domain has sufficient inclusivity to support concept derivation in multiple directions.

2. **Clear Boundaries**
   - Explicitly define the core content the domain addresses, and distinguish it from adjacent domains to ensure concept generation has well-defined boundaries.

3. **Technically Rigorous (Yet Abstract)**
   - Use precise but non-specific terminology (e.g., "expressive strategies," "information organization methods," "social interaction norms," "cognitive orientations").
   - Avoid enumerating specific categories, frameworks, vocabulary, or content that could directly constitute concepts.

4. **Relevant to Model Steering**
   - Explain the general significance of this domain in model behavior regulation, preference expression, or style control, without involving any hierarchical or stratified concepts.

5. **Output Format**
   - Output a single paragraph of approximately 80–120 words.
   - Do not use lists or bullet points.
   - Do not generate example concepts or behaviors.

# Input Domain
{USER_INPUT_DOMAIN}

# Output
A single paragraph domain description that meets the above requirements.
"""

PROMPTS["generate_concepts_stage_2"] = """
# Role
You are an expert AI Benchmark Designer specializing in "Steering Vectors" and "Concept Hierarchies." Your task is to synthesize hierarchical steering concepts based on a specific domain description.

# Task
Generate a dataset of hierarchical concepts based on the provided `Domain Name`, `Domain Description`, and `Structure Counts`.

The hierarchy is defined as follows:
* **Level 1 (L1) - Macro Style / Persona:** The highest level of abstraction. A broad instruction about the model's mindset, emotional state, or overall writing style.
* **Level 2 (L2) - Strategic / Structural:** A specific behavioral rule, syntactic strategy, or logical framework that supports the L1 concept. Subcategory or specific manifestation direction of L1; Can focus on specific themes or scenes. It bridges the gap between abstract style and concrete words. 
* **Level 3 (L3) - Atomic / Hard Constraint:** The finest granularity. A verifiable, binary constraint, usually involving a specific keyword, named entity, punctuation pattern, or fixed format that acts as a "proof" that the model is following the L2 strategy.
- The constraint content in the concept needs to be niche and difficult to ensure that it does not appear in normal answers.


# Input Data
* **Target Domain Name:** {DOMAIN_NAME}
* **Domain Description:** {DOMAIN_DESCRIPTION}
* **L1 Count:** {N_L1}
* **L2 Count per L1:** {N_L2}
* **L3 Count per L2:** {N_L3}

# Reference Examples (Learning from these patterns)

## Example 1: Domain [Reasoning Patterns]
* **L1:** "Adopt a divergent, exploratory reasoning pattern (lateral/creative thinking)."
* **L2:** "Explicitly generate and list analogies from disparate domains."
* **L3:** "The response must explicitly name an analogy source domain unrelated to the problem's primary domain (e.g., 'biology', 'architecture')."

## Example 2: Domain [Sentiment]
* **L1:** ""Express gratitude and give thanks to others or things.""
* **L2:** "Express gratitude toward fate and destiny"
* **L3:** "The response must include the word \"beholden\"."

# Guidelines
1.  **Alignment:** Ensure all concepts strictly align with the provided `Domain Description`.
2.  **Distinctiveness:** Ensure L1 concepts are distinct from each other (capturing different aspects of the domain description).
3.  **Logical Flow:** L3 must be a natural consequence or valid implementation of L2; L2 must be a valid strategy to achieve L1.
4.  **Verifiability:** L3 must be objectively checkable (e.g., string matching, regex). Avoid subjective L3s.
5.  **Format:** Output strictly valid JSON.


# Output Format
Return a JSON object containing a list of hierarchies.
```json
{{
  "L1_concepts": [
    {{
      "concept_id": "L1_1",
      "concept": "...",
      "L2_subconcepts": [
        {{
          "concept_id": "L2_1",
          "concept": "...",
          "L3_features": [
            {{"concept_id": "L3_1", "concept": "..."}},
            ...,
            {{"concept_id": "L3_{N_L3}", "concept": "..."}}
          ]
        }},
        ...,
        {{"concept_id": "L2_{N_L2}", "concept": "...", "L3_features": [...] }}
      ]
    }},
    ...,
    {{"concept_id": "L1_{N_L3}", "concept": "...", "L2_subconcepts": [...]}}
  ]
}}
```
"""


