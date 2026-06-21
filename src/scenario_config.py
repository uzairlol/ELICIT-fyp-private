# scenario_config.py

"""
Configuration maps for grounding the simulation into real-world scenarios.
"""

import logging

logger = logging.getLogger(__name__)

SCENARIOS = {
    "abstract": {
        "game_name": "public goods game",
        "currency_name": "tokens",
        "account_name": "experimental account",
        "project_name": "the project",
        "sfi_name": "Group A (Sanction-Free Institution - SFI)",
        "sfi_desc": "No possibility to impose sanctions or rewards on other group members",
        "si_name": "Group B (Sanctioning Institution - SI)",
        "si_desc": "Possibility to impose sanctions or rewards on other group members",
        "punishment_name": "negative token",
        "reward_name": "positive token",
        "stage_1_name": "Group Choice and Contribution to the Project",
        "stage_2_name": "Assignment of Tokens (Only in SI)"
    },
    
    "ldf": {
        "game_name": "Global Climate Change Summit",
        "currency_name": "million USD",
        "account_name": "national treasury",
        "project_name": "Global Emissions Reduction Fund",
        "sfi_name": "Group A (Non-Binding Climate Agreement)",
        "sfi_desc": "No possibility to impose trade tariffs (sanctions) or economic aid (rewards) on other nations",
        "si_name": "Group B (Binding Climate Treaty)",
        "si_desc": "Strict enforcement protocol allowing trade tariffs (sanctions) or economic aid (rewards) towards other nations",
        "punishment_name": "trade tariff penalty",
        "reward_name": "economic aid package",
        "stage_1_name": "Treaty Selection and Emissions Fund Contribution",
        "stage_2_name": "Enforcement Actions (Only in Binding Treaty)"
    },
    
    "tax": {
        "game_name": "Corporate Tax Compliance Framework",
        "currency_name": "million euros",
        "account_name": "corporate reserve fund",
        "project_name": "the Public Infrastructure Tax Pool",
        "sfi_name": "Group A (Voluntary Tax Code)",
        "sfi_desc": "No regulatory oversight to audit or penalize other corporations",
        "si_name": "Group B (Strict Regulatory Compliance Framework)",
        "si_desc": "Whistleblower and audit protocols allowing you to report/penalize or endorse other corporations",
        "punishment_name": "regulatory audit penalty",
        "reward_name": "tax-rebate endorsement",
        "stage_1_name": "Framework Selection and Tax Pool Declaration",
        "stage_2_name": "Corporate Audits / Endorsements (Only in Strict Framework)"
    }
}

def get_scenario_config(scenario_name: str) -> dict:
    """
    Returns the dictionary configuration for the requested scenario.
    Defaults to 'abstract' if the requested scenario isn't found.
    """
    scenario_key = scenario_name.lower()
    if scenario_key == "climate":
        scenario_key = "ldf"

    if scenario_key not in SCENARIOS:
        logger.warning(f"[{scenario_name}] scenario not found. Defaulting to 'abstract'.")
        return SCENARIOS["abstract"]
    return SCENARIOS[scenario_key]
