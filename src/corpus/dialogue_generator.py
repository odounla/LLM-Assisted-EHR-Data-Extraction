

# The prompt template. The {curly_brace} names are placeholders we fill in
# at runtime with the patient's real, extracted context.
PROMPT_TEMPLATE = """[ROLE]
You are an endocrinologist conducting a follow-up visit with an
established patient. The patient's diagnoses are listed in the problem
list below - treat those as the patient's actual conditions and do not
assume any diagnosis that is not listed.

[PATIENT CONTEXT]
Problem list:
{conditions}

Current medications:
{medications}

Most recent labs:
{observations}

[TASK]
Generate a realistic outpatient encounter dialogue between the physician
and this patient, grounded ONLY in the context above. Do not invent or
assume any conditions, labs, medications, devices, or diagnoses that are
not explicitly listed above.

[FORMAT]
- Label each turn DOCTOR: or PATIENT:
- 8-12 exchanges (one back-and-forth = one exchange)
- One visit, one continuous conversation

[REALISM]
- The patient speaks in plain, everyday language - not medical jargon
- The doctor asks how the patient is feeling and how the medications are going
- The conversation centers on the lab results and next steps
- Include natural back-and-forth: a question, a follow-up, a clarification
"""


def build_prompt(context: dict) -> str:
    """
    Job 1: turn the context dict into a single filled-in prompt string.

    context["conditions"]   is a list of strings
    context["medications"]  is a list of strings
    context["observations"] is a list of dicts: {name, value, unit, date}
    """
    # Lists of strings -> a dashed, one-per-line block.
    conditions_text = "\n".join("- " + c for c in context["conditions"])
    medications_text = "\n".join("- " + m for m in context["medications"])

    # Observations are dicts, so we format each into a readable sentence.
    observations_text = "\n".join(
        f"- {o['name']}: {o['value']} {o['unit']} ({o['date']})"
        for o in context["observations"]
    )

    # .format() swaps each {placeholder} for the matching text block.
    return PROMPT_TEMPLATE.format(
        conditions=conditions_text,
        medications=medications_text,
        observations=observations_text,
    )


# ---------------------------------------------------------------------------
# Job 2 + 3: send the prompt to an LLM and return the dialogue text.
#
# Two versions below. They do the SAME three things:
#   1. build the prompt (Job 1, above)
#   2. send it to the model as a "user" message
#   3. pull the text out of the response
# Use whichever matches the API you're running against.
# ---------------------------------------------------------------------------


# def generate_dialogue_openai(context: dict) -> str:
#     """OpenAI version (gpt-4o). This is the coursework target."""
#     from openai import OpenAI

#     client = OpenAI()  # reads OPENAI_API_KEY from your environment
#     prompt = build_prompt(context)

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return response.choices[0].message.content

def generate_dialogue_anthropic(context: dict) -> str:
    """Anthropic version. """
    from anthropic import Anthropic
 
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    prompt = build_prompt(context)
 
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,  # Anthropic requires an explicit cap on reply length
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


