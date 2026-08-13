import json


def extract_clinical_context(bundle: dict) -> dict:
    conditions = [
        e['resource'].get('code', {}).get('text', '')
        for e in bundle['entry']
        if e['resource']['resourceType'] == 'Condition'
    ]
    conditions =[x for x in set(conditions) if x]
    medications = [
        e['resource'].get('medicationCodeableConcept', {}).get('text', '')
        for e in bundle['entry']
        if e['resource']['resourceType'] == 'MedicationRequest'
    ]
    medications = [x for x in set(medications) if x]
    hba1c_obs = [
        e['resource'] for e in bundle['entry']
        if e['resource']['resourceType'] == 'Observation'
        and 'hba1c' in e['resource'].get('code', {}).get('text', '').lower()
    ]
    hba1c_obs.sort(key=lambda x: x.get('effectiveDateTime', ''), reverse=True)
    latest_hba1c = hba1c_obs[0].get('code', {}).get('text', '') if hba1c_obs else None

    glucose_obs = [
        e['resource'] for e in bundle['entry']
        if e['resource']['resourceType'] == 'Observation'
        and 'glucose' in e['resource'].get('code', {}).get('text', '').lower()
    ]
    glucose_obs.sort(key=lambda x: x.get('effectiveDateTime', ''), reverse=True)
    latest_glucose = glucose_obs[0].get('code', {}).get('text', '') if glucose_obs else None

    observations = [x for x in [latest_hba1c, latest_glucose] if x]

    return {
        "conditions": conditions,
        "medications": medications,
        "observations": observations
    }


if __name__ == '__main__':
    with open('data/synthea/test/fhir/Mignon230_Cummings51_7811dc30-8c86-8164-6c33-6b6f464745ef.json') as f:
        bundle = json.load(f)

    result = extract_clinical_context(bundle)
    print("Conditions:", result['conditions'])
    print("Medications:", result['medications'])
    print("Observations:", result['observations'])
