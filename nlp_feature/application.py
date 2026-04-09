"""
Applicant Feature Extraction Module

Extracts features from LEAD APPLICANT & RESEARCH TEAM section
based on NIHR scoring criteria for applicant assessment.
"""

import re
from typing import Any


# Keywords for detecting research experience signals
PUBLICATION_KEYWORDS = [
    'publication', 'publications', 'published', 'paper', 'papers',
    'journal', 'authored', 'co-authored', 'first author', 'citation',
    'citations', 'cited', 'peer-reviewed'
]

GRANT_KEYWORDS = [
    'grant', 'grants', 'funding', 'funded', 'award', 'awarded',
    '£', 'nihr', 'mrc', 'wellcome', 'ukri', 'esrc', 'epsrc',
    'portfolio', 'principal investigator', 'pi', 'co-i'
]

LEADERSHIP_KEYWORDS = [
    'lead', 'leader', 'leading', 'led', 'director', 'chair',
    'head', 'chief', 'principal', 'senior', 'professor',
    'established', 'founded', 'built', 'developed'
]

MENTORSHIP_KEYWORDS = [
    'mentor', 'mentoring', 'mentored', 'supervise', 'supervised',
    'supervision', 'supervisor', 'phd student', 'doctoral',
    'postdoc', 'early career', 'training'
]

CLINICAL_PRACTICE_KEYWORDS = [
    'clinical', 'clinician', 'consultant', 'practitioner',
    'nhs', 'patient', 'practice', 'hospital', 'gp', 'nurse'
]


def extract_applicant_features(data: dict) -> dict:
    """
    Extract all applicant-related features from the JSON data.

    Args:
        data: Full JSON data loaded from application file

    Returns:
        Dictionary containing applicant features
    """
    team_section = data.get("LEAD APPLICANT & RESEARCH TEAM", {})

    lead_applicant = team_section.get("Lead Applicant", {})
    joint_lead = team_section.get("Joint Lead Applicant")
    co_applicants = team_section.get("Co-Applicants", []) or []

    # Build full team list
    all_members = []
    if lead_applicant:
        all_members.append(lead_applicant)
    if joint_lead:
        all_members.append(joint_lead)
    all_members.extend(co_applicants)

    features = {}

    # --- Team Composition Features ---
    features.update(_extract_team_composition(all_members, lead_applicant, joint_lead))

    # --- Research Experience Features ---
    features.update(_extract_research_experience(all_members))

    # --- Leadership & Career Features ---
    features.update(_extract_leadership_signals(all_members))

    # --- Clinical Practice Features ---
    features.update(_extract_clinical_signals(all_members))

    # --- Organization Diversity ---
    features.update(_extract_org_diversity(all_members))

    return features


def _extract_team_composition(members: list, lead: dict, joint_lead: Any) -> dict:
    """Extract team size and FTE commitment features."""

    team_size = len(members)

    # Parse FTE percentages
    fte_values = []
    for member in members:
        fte_str = member.get("% FTE Commitment", "0%")
        fte = _parse_percentage(fte_str)
        fte_values.append(fte)

    total_fte = sum(fte_values)
    avg_fte = total_fte / team_size if team_size > 0 else 0

    # Lead applicant FTE
    lead_fte = _parse_percentage(lead.get("% FTE Commitment", "0%")) if lead else 0

    # ORCID coverage
    orcid_count = sum(1 for m in members if m.get("ORCID") and m.get("ORCID").strip())
    orcid_ratio = orcid_count / team_size if team_size > 0 else 0

    # Has joint lead
    has_joint_lead = joint_lead is not None

    return {
        "team_size": team_size,
        "total_fte_commitment": round(total_fte, 2),
        "avg_fte_commitment": round(avg_fte, 2),
        "lead_fte_commitment": lead_fte,
        "has_joint_lead": has_joint_lead,
        "orcid_count": orcid_count,
        "orcid_coverage_ratio": round(orcid_ratio, 2)
    }


def _extract_research_experience(members: list) -> dict:
    """Extract research experience signals from role descriptions."""

    all_roles_text = _get_all_roles_text(members)
    all_roles_lower = all_roles_text.lower()

    # Publication mentions
    pub_count = _count_keyword_matches(all_roles_lower, PUBLICATION_KEYWORDS)
    has_publications = pub_count > 0

    # Grant/funding mentions
    grant_count = _count_keyword_matches(all_roles_lower, GRANT_KEYWORDS)
    has_prior_grants = grant_count > 0

    # Count citation metrics mentioned
    citation_pattern = r'(\d{1,3}(?:,\d{3})*)\s*citations?'
    citation_matches = re.findall(citation_pattern, all_roles_lower)
    max_citations = 0
    if citation_matches:
        max_citations = max(int(c.replace(',', '')) for c in citation_matches)

    # Role description lengths
    role_lengths = [len(m.get("Proposed Role", "").split()) for m in members]
    total_role_words = sum(role_lengths)
    avg_role_words = total_role_words / len(members) if members else 0

    return {
        "publication_mention_count": pub_count,
        "has_publication_mentions": has_publications,
        "grant_mention_count": grant_count,
        "has_prior_grants": has_prior_grants,
        "max_citations_mentioned": max_citations,
        "total_role_description_words": total_role_words,
        "avg_role_description_words": round(avg_role_words, 1)
    }


def _extract_leadership_signals(members: list) -> dict:
    """Extract leadership and career trajectory signals."""

    all_roles_text = _get_all_roles_text(members)
    all_roles_lower = all_roles_text.lower()

    # Leadership keywords
    leadership_count = _count_keyword_matches(all_roles_lower, LEADERSHIP_KEYWORDS)

    # Mentorship signals
    mentorship_count = _count_keyword_matches(all_roles_lower, MENTORSHIP_KEYWORDS)

    # Seniority indicators from titles
    senior_titles = ['professor', 'prof', 'director', 'head', 'chief', 'senior']
    senior_count = 0
    for member in members:
        name = member.get("Full Name", "").lower()
        if any(title in name for title in senior_titles):
            senior_count += 1

    # Career trajectory phrases
    trajectory_phrases = [
        'career', 'trajectory', 'vision', 'long-term', 'future',
        'aspiration', 'goal', 'develop', 'progression', 'pathway'
    ]
    trajectory_count = _count_keyword_matches(all_roles_lower, trajectory_phrases)

    return {
        "leadership_mention_count": leadership_count,
        "mentorship_mention_count": mentorship_count,
        "senior_title_count": senior_count,
        "career_trajectory_signals": trajectory_count
    }


def _extract_clinical_signals(members: list) -> dict:
    """Extract clinical/practitioner-academic signals."""

    all_roles_lower = _get_all_roles_text(members).lower()

    # Clinical practice mentions
    clinical_count = _count_keyword_matches(all_roles_lower, CLINICAL_PRACTICE_KEYWORDS)

    # Identify clinical roles in team
    clinical_roles = ['consultant', 'clinician', 'gp', 'nurse', 'practitioner']
    clinical_members = 0
    for member in members:
        role = member.get("Proposed Role", "").lower()
        name = member.get("Full Name", "").lower()
        if any(cr in role or cr in name for cr in clinical_roles):
            clinical_members += 1

    has_clinical_academic = clinical_count > 0

    return {
        "clinical_mention_count": clinical_count,
        "clinical_team_members": clinical_members,
        "has_clinical_academic_focus": has_clinical_academic
    }


def _extract_org_diversity(members: list) -> dict:
    """Extract organizational diversity features."""

    orgs = set()
    org_types = {"nhs": 0, "university": 0, "charity": 0, "other": 0}

    for member in members:
        org = member.get("Organisation", "")
        if org:
            orgs.add(org)
            org_lower = org.lower()
            if 'nhs' in org_lower or 'trust' in org_lower:
                org_types["nhs"] += 1
            elif 'university' in org_lower or 'college' in org_lower:
                org_types["university"] += 1
            elif 'charity' in org_lower or any(x in org_lower for x in ['rnid', 'cancer research']):
                org_types["charity"] += 1
            else:
                org_types["other"] += 1

    return {
        "unique_organisations": len(orgs),
        "nhs_members": org_types["nhs"],
        "university_members": org_types["university"],
        "charity_members": org_types["charity"],
        "has_multi_org_collaboration": len(orgs) > 1
    }


# --- Helper Functions ---

def _parse_percentage(value: str) -> float:
    """Parse percentage string to float."""
    if not value:
        return 0.0
    # Remove % and parse
    clean = str(value).replace('%', '').strip()
    try:
        return float(clean)
    except ValueError:
        return 0.0


def _get_all_roles_text(members: list) -> str:
    """Concatenate all role descriptions."""
    roles = [m.get("Proposed Role", "") for m in members if m.get("Proposed Role")]
    return " ".join(roles)


def _count_keyword_matches(text: str, keywords: list) -> int:
    """Count total occurrences of keywords in text."""
    count = 0
    for keyword in keywords:
        count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE))
    return count
