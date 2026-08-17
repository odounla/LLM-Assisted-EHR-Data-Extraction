from dotenv import load_dotenv
load_dotenv()

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
    conditions_text = "\n".join("- " + c for c in context["conditions"])
    medications_text = "\n".join("- " + m for m in context["medications"])
    observations_text = "\n".join(
        f"- {o['name']}: {o['value']} {o['unit']} ({o['date']})"
        for o in context["observations"]
    )
    return PROMPT_TEMPLATE.format(
        conditions=conditions_text,
        medications=medications_text,
        observations=observations_text,
    )


def generate_dialogue_anthropic(context: dict) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    prompt = build_prompt(context)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == '__main__':
    import json
    import sys
    sys.path.insert(0, '.')
    from src.corpus.fhir_extractor import extract_clinical_context

    with open('data/synthea/test/fhir/Mignon230_Cummings51_7811dc30-8c86-8164-6c33-6b6f464745ef.json') as f:
        bundle = json.load(f)

    context = extract_clinical_context(bundle)
    dialogue = generate_dialogue_anthropic(context)
    print(dialogue)
