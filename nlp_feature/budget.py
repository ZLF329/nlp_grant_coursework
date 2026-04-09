"""
Budget Feature Extraction Module

Extracts features from SUMMARY INFORMATION and SUMMARY BUDGET sections
based on NIHR scoring criteria for resource justification.
"""

import re


# Keywords for budget category detection
PPI_KEYWORDS = [
    'ppi', 'ppie', 'public involvement', 'patient involvement',
    'lived experience', 'co-production', 'engagement'
]

CAPACITY_BUILDING_KEYWORDS = [
    'phd', 'doctoral', 'studentship', 'fellowship', 'training',
    'capacity building', 'early career', 'research associate'
]

EQUIPMENT_KEYWORDS = [
    'equipment', 'device', 'devices', 'hardware', 'software',
    'tablet', 'laptop', 'kit', 'consumable', 'consumables'
]

STAFF_KEYWORDS = [
    'staff', 'salary', 'salaries', 'fte', 'researcher',
    'coordinator', 'manager', 'administrator'
]


def extract_budget_features(data: dict) -> dict:
    """
    Extract all budget-related features from the JSON data.

    Args:
        data: Full JSON data loaded from application file

    Returns:
        Dictionary containing budget features
    """
    summary_info = data.get("SUMMARY INFORMATION", {})
    budget_text = data.get("SUMMARY BUDGET", "")

    features = {}

    # --- Basic Project Metrics ---
    features.update(_extract_project_metrics(summary_info))

    # --- Resource Allocation ---
    features.update(_extract_resource_allocation(budget_text))

    return features


def _extract_project_metrics(summary_info: dict) -> dict:
    """Extract basic project cost and duration metrics."""

    # Total cost
    total_cost_str = summary_info.get("Total Cost to NIHR", "£0")
    total_cost = _parse_currency(total_cost_str)

    # Duration
    duration_str = summary_info.get("Duration (months)", "0")
    try:
        duration_months = int(duration_str)
    except ValueError:
        duration_months = 0

    # Cost per month
    cost_per_month = total_cost / duration_months if duration_months > 0 else 0

    # NHS costs
    nhs_support = _parse_currency(summary_info.get("NHS Support Costs", "£0"))
    nhs_excess = _parse_currency(summary_info.get("NHS Excess Treatment Costs", "£0"))
    total_nhs_costs = nhs_support + nhs_excess

    # Previous submission
    previously_submitted = summary_info.get(
        "Has this application been previously submitted to this or any other funding body?", ""
    )
    is_resubmission = previously_submitted.lower() == "yes"

    return {
        "total_cost_to_nihr": total_cost,
        "duration_months": duration_months,
        "cost_per_month": round(cost_per_month, 2),
        "nhs_support_costs": nhs_support,
        "nhs_excess_treatment_costs": nhs_excess,
        "total_nhs_costs": total_nhs_costs,
        "is_resubmission": is_resubmission
    }


def _extract_resource_allocation(budget_text: str) -> dict:
    """Extract resource allocation signals from budget text."""

    if not budget_text:
        return {
            "has_ppi_budget": False,
            "ppi_budget_amount": 0,
            "has_capacity_building": False,
            "has_equipment_costs": False,
            "has_staff_costs": False
        }

    budget_lower = budget_text.lower()

    # PPI/PPIE budget
    has_ppi = any(kw in budget_lower for kw in PPI_KEYWORDS)
    ppi_amount = _extract_category_amount(budget_text, PPI_KEYWORDS)

    # Capacity building (PhD, training)
    has_capacity = any(kw in budget_lower for kw in CAPACITY_BUILDING_KEYWORDS)

    # Equipment costs
    has_equipment = any(kw in budget_lower for kw in EQUIPMENT_KEYWORDS)

    # Staff costs
    has_staff = any(kw in budget_lower for kw in STAFF_KEYWORDS)

    return {
        "has_ppi_budget": has_ppi,
        "ppi_budget_amount": ppi_amount,
        "has_capacity_building": has_capacity,
        "has_equipment_costs": has_equipment,
        "has_staff_costs": has_staff
    }


# --- Helper Functions ---

def _parse_currency(value: str) -> float:
    """Parse currency string (£xxx,xxx.xx) to float."""
    if not value:
        return 0.0

    # Remove £ and commas
    clean = str(value).replace('£', '').replace(',', '').strip()

    # Handle k (thousands) and m (millions) suffixes
    multiplier = 1
    if clean.lower().endswith('k'):
        multiplier = 1000
        clean = clean[:-1]
    elif clean.lower().endswith('m'):
        multiplier = 1000000
        clean = clean[:-1]

    try:
        return float(clean) * multiplier
    except ValueError:
        return 0.0


def _extract_category_amount(text: str, keywords: list) -> float:
    """Extract monetary amount associated with keywords."""

    for keyword in keywords:
        # Look for pattern: keyword ... £amount
        pattern = rf'{keyword}[^£]*?(£[\d,]+(?:\.\d+)?[km]?)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return _parse_currency(matches[0])

    return 0.0
