"""Reorder presentation without selecting, rewriting or approving evidence."""
import copy

from grounding_contract import GroundingContractError


def order_feedback_passages(payload):
    result = copy.deepcopy(payload)
    catalog = result['passages']
    priority = list(dict.fromkeys(key for row in result.get('rejected_claims', [])
        if row['reason'] != 'procedure_requires_revalidation' for key in row['passage_ids']))
    if any(key not in catalog for key in priority):
        raise GroundingContractError('Feedback references an unknown passage')
    ordered = priority + [key for key in catalog if key not in priority]
    result['passages'] = {key: catalog[key] for key in ordered}
    return result, ordered != list(catalog)
