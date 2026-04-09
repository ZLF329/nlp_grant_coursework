import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

ORCID_API = "https://pub.orcid.org/v3.0"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "orcid-impact-features/0.1 (mailto:your_email@example.com)",
}

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)


POLICY_KEYWORDS = [
    "policy", "guideline", "nhs", "nice", "intervention", "inequal", "equity",
    "population", "public health", "oecd", "universal health care", "health service",
]
CLINICAL_KEYWORDS = [
    "diagnosis", "neonatal", "hospital", "admission", "readmission", "trial",
    "meta-analysis", "systematic review", "dementia", "dengue", "rt-pcr",
]

# --- funder prestige proxy
FUNDER_PRESTIGE = {
    # UK
    "national institute for health and care research": 3,
    "uk research and innovation": 3,
    "medical research council": 3,
    "engineering and physical sciences research council": 3,
    "biotechnology and biological sciences research council": 3,
    "economic and social research council": 3,
    "natural environment research council": 3,
    "science and technology facilities council": 3,
    # US/EU
    "national institutes of health": 3,
    "nsf": 3,
    "european research council": 3,
    # big charities (examples)
    "wellcome": 2,
    "bill & melinda gates foundation": 2,
}

OPENALEX_WORKS = "https://api.openalex.org/works"

def normalize_doi(doi: str) -> str:
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()
    return doi.lower()

def _openalex_cited_by_one(
    doi: str,
    *,
    timeout_sec: int = 20,
    max_retries: int = 2,
    backoff_s: float = 0.4,
) -> Optional[int]:
    d = normalize_doi(doi)
    url = f"{OPENALEX_WORKS}/https://doi.org/{d}"
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=timeout_sec)
            if r.status_code == 200:
                return int(r.json().get("cited_by_count", 0))
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(backoff_s * (2 ** attempt))
                continue
            return None
        except requests.RequestException:
            if attempt < max_retries:
                time.sleep(backoff_s * (2 ** attempt))
                continue
            return None
    return None


def openalex_cited_by_for_dois(
    dois: List[str],
    sleep_s: float = 0.0,
    max_workers: int = 8,
    timeout_sec: int = 20,
    max_retries: int = 2,
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    uniq_dois = list(dict.fromkeys(dois))
    if not uniq_dois:
        return out

    worker_count = max(1, int(max_workers))
    pause = max(0.0, float(sleep_s))

    if worker_count == 1:
        for doi in uniq_dois:
            cited_by = _openalex_cited_by_one(
                doi,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
            )
            if cited_by is not None:
                out[doi] = cited_by
            if pause > 0:
                time.sleep(pause)
        return out

    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = {
            ex.submit(
                _openalex_cited_by_one,
                doi,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
            ): doi
            for doi in uniq_dois
        }
        for fut in as_completed(futures):
            doi = futures[fut]
            try:
                cited_by = fut.result()
            except Exception:
                cited_by = None
            if cited_by is not None:
                out[doi] = cited_by
            if pause > 0:
                time.sleep(pause)

    return out


# --- ORCID work types you might want to count explicitly
DATASET_TYPES = {"dataset"}
SOFTWARE_TYPES = {"software"}
PATENT_TYPES = {"patent"}
PRESENTATION_TYPES = {"lecture-speech", "conference-presentation", "presentation"}
REPORT_TYPES = {"report", "working-paper", "other"}


def as_value(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x.get("value")
    if isinstance(x, str):
        return x
    return str(x)


def get_json(url: str, params: Optional[dict] = None) -> Dict[str, Any]:
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_doi(external_ids: Dict[str, Any]) -> Optional[str]:
    if not external_ids:
        return None
    items = external_ids.get("external-id", []) or []
    for it in items:
        id_type = (it.get("external-id-type") or "").lower()
        val = it.get("external-id-value")
        if id_type == "doi" and val:
            return val.strip()
        if val:
            m = DOI_RE.search(val)
            if m:
                return m.group(0)
    return None


def get_person(orcid: str) -> Dict[str, Any]:
    return get_json(f"{ORCID_API}/{orcid}/person")


def parse_basic_person_info(person_json: Dict[str, Any]) -> Dict[str, Any]:
    name = person_json.get("name") or {}
    given = as_value(name.get("given-names"))
    family = as_value(name.get("family-name"))

    emails = person_json.get("emails") or {}
    email_list = []
    for e in (emails.get("email") or []):
        email_list.append(e.get("email"))

    return {
        "given_name": given,
        "family_name": family,
        "full_name": " ".join([x for x in [given, family] if x]),
        "emails_public": [x for x in email_list if x],
    }


def get_works_summary(orcid: str) -> List[Dict[str, Any]]:
    data = get_json(f"{ORCID_API}/{orcid}/works")
    groups = data.get("group", []) or []
    out = []
    for g in groups:
        summaries = g.get("work-summary", []) or []
        for s in summaries[:1]:
            put_code = s.get("put-code")
            title = as_value(((s.get("title") or {}).get("title")))
            year = as_value(((s.get("publication-date") or {}).get("year")))
            wtype = s.get("type")
            doi = extract_doi(s.get("external-ids") or {})
            out.append({
                "put_code": put_code,
                "title": title,
                "year": int(year) if year and str(year).isdigit() else None,
                "type": wtype,
                "doi": doi,
            })
    return out


def get_funding_summary(orcid: str) -> List[Dict[str, Any]]:
    data = get_json(f"{ORCID_API}/{orcid}/fundings")
    groups = data.get("group", []) or []
    out = []
    for g in groups:
        summaries = g.get("funding-summary", []) or []
        for s in summaries[:1]:
            title = as_value(((s.get("title") or {}).get("title")))
            ftype = s.get("type")

            org_obj = s.get("organization") or {}
            org_name = as_value(org_obj.get("name"))
            org_city = as_value((org_obj.get("address") or {}).get("city"))
            org_country = as_value((org_obj.get("address") or {}).get("country"))

            year = as_value((s.get("start-date") or {}).get("year"))

            out.append({
                "title": title,
                "type": ftype,
                "org": org_name,
                "org_city": org_city,
                "org_country": org_country,
                "start_year": int(year) if year and str(year).isdigit() else None,
            })
    return out


def recency_weight(pub_year: Optional[int], now_year: int) -> float:
    if pub_year is None:
        return 0.0
    age = now_year - pub_year
    if age <= 1:
        return 1.0
    if age <= 3:
        return 0.7
    if age <= 5:
        return 0.4
    return 0.2


def count_keyword_hits(titles: List[str], keywords: List[str]) -> int:
    hits = 0
    for t in titles:
        tl = t.lower()
        for kw in keywords:
            if kw in tl:
                hits += 1
                break  
    return hits


def funder_prestige_score(org_name: Optional[str]) -> int:
    if not org_name:
        return 0
    key = org_name.strip().lower()
    if key in FUNDER_PRESTIGE:
        return FUNDER_PRESTIGE[key]
    for k, v in FUNDER_PRESTIGE.items():
        if k in key:
            return v
    return 0


def compute_features(
    profile: Dict[str, Any],
    now_year: Optional[int] = None,
    doi2citedby: Optional[Dict[str, int]] = None, 
) -> Dict[str, Any]:
    if now_year is None:
        now_year = datetime.utcnow().year

    person = profile.get("person") or {}
    works = profile.get("works") or []
    fundings = profile.get("fundings") or []
    stats = profile.get("stats") or {}

    works_total = len(works)
    works_with_doi = sum(1 for w in works if w.get("doi"))
    doi_ratio = (works_with_doi / works_total) if works_total > 0 else 0.0

    years = sorted([w["year"] for w in works if w.get("year") is not None])
    first_pub_year = years[0] if years else None
    last_pub_year = years[-1] if years else None
    career_years = max(1, (now_year - first_pub_year + 1)) if first_pub_year else 1

    # recent windows
    def is_recent(y: Optional[int], window: int) -> bool:
        if y is None:
            return False
        return y >= (now_year - (window - 1))

    works_recent_3y = sum(1 for w in works if is_recent(w.get("year"), 3))
    works_recent_5y = sum(1 for w in works if is_recent(w.get("year"), 5))

    # type counts
    type_counts: Dict[str, int] = {}
    for w in works:
        t = (w.get("type") or "unknown").lower()
        type_counts[t] = type_counts.get(t, 0) + 1

    journal_articles = type_counts.get("journal-article", 0)
    presentations = sum(type_counts.get(t, 0) for t in PRESENTATION_TYPES)
    reports = sum(type_counts.get(t, 0) for t in REPORT_TYPES)
    datasets = sum(type_counts.get(t, 0) for t in DATASET_TYPES)
    software = sum(type_counts.get(t, 0) for t in SOFTWARE_TYPES)
    patents = sum(type_counts.get(t, 0) for t in PATENT_TYPES)

    output_diversity = len([k for k, v in type_counts.items() if v > 0])

    # recency-weighted publications
    pub_score_recency = sum(recency_weight(w.get("year"), now_year) for w in works)

    # titles keyword proxies
    titles = [w.get("title") for w in works if w.get("title")]
    policy_hits = count_keyword_hits(titles, POLICY_KEYWORDS)
    clinical_hits = count_keyword_hits(titles, CLINICAL_KEYWORDS)

    societal_focus_ratio = (policy_hits / works_total) if works_total > 0 else 0.0

    # recognition / funding
    funding_total = len(fundings)
    has_funding_records = 1 if funding_total > 0 else 0

    funder_scores = [funder_prestige_score(f.get("org")) for f in fundings]
    funder_score_sum = sum(funder_scores)
    funder_score_max = max(funder_scores) if funder_scores else 0

    # flags for common funders
    org_names_lower = [(f.get("org") or "").lower() for f in fundings]
    funder_nihr_flag = 1 if any("national institute for health and care research" in x for x in org_names_lower) else 0
    funder_ukri_flag = 1 if any("uk research and innovation" in x for x in org_names_lower) else 0

    # funding year completeness
    funding_with_year = sum(1 for f in fundings if f.get("start_year") is not None)
    funding_has_start_year_ratio = (funding_with_year / funding_total) if funding_total > 0 else 0.0

    # missingness flags
    has_email_public = 1 if (person.get("emails_public") or []) else 0

    # --- citations placeholders (optional fill)
    citations = {
        "citations_available": 0,
        "citation_coverage_ratio": None,
        "citations_total": None,
        "citations_recent_5y": None,
        "citations_per_year": None,
        "h_index": None,
        "h_index_recent_5y": None,
        "citations_median_per_paper": None,
        "citations_top1": None,
        "citations_top3_sum": None,
    }

    if doi2citedby is not None:
        cited_list = []
        cited_recent5 = []
        matched = 0
        for w in works:
            doi = w.get("doi")
            if not doi:
                continue
            if doi in doi2citedby:
                matched += 1
                c = int(doi2citedby[doi])
                cited_list.append(c)
                if is_recent(w.get("year"), 5):
                    cited_recent5.append(c)

        citations["citations_available"] = 1
        citations["citation_coverage_ratio"] = (matched / works_with_doi) if works_with_doi > 0 else 0.0
        citations["citations_total"] = sum(cited_list) if cited_list else 0
        citations["citations_recent_5y"] = sum(cited_recent5) if cited_recent5 else 0
        citations["citations_per_year"] = (citations["citations_total"] / career_years) if career_years > 0 else None

        # h-index
        def h_index_from_counts(counts: List[int]) -> int:
            counts_sorted = sorted(counts, reverse=True)
            h = 0
            for i, c in enumerate(counts_sorted, start=1):
                if c >= i:
                    h = i
                else:
                    break
            return h

        citations["h_index"] = h_index_from_counts(cited_list) if cited_list else 0
        citations["h_index_recent_5y"] = h_index_from_counts(cited_recent5) if cited_recent5 else 0
        citations["citations_median_per_paper"] = (
            sorted(cited_list)[len(cited_list)//2] if cited_list else 0
        )
        citations["citations_top1"] = max(cited_list) if cited_list else 0
        citations["citations_top3_sum"] = sum(sorted(cited_list, reverse=True)[:3]) if cited_list else 0

    # --- final feature JSON (grouped)
    features = {
        "meta": {
            "orcid": profile.get("orcid"),
            "full_name": person.get("full_name"),
            "now_year": now_year,
        },
        "career": {
            "first_pub_year": first_pub_year,
            "last_pub_year": last_pub_year,
            "career_years": career_years,
        },
        "outputs": {
            "works_total": works_total,
            "works_recent_3y": works_recent_3y,
            "works_recent_5y": works_recent_5y,
            "works_per_year": works_total / career_years if career_years > 0 else None,
            "works_with_doi": works_with_doi,
            "works_with_doi_ratio": doi_ratio,
            "type_counts": type_counts,
            "journal_article_total": journal_articles,
            "presentations_total": presentations,
            "reports_total": reports,
            "dataset_total": datasets,
            "software_total": software,
            "patent_total": patents,
            "output_diversity": output_diversity,
            "pub_score_recency": pub_score_recency,
        },
        "impact": citations,  
        "recognition": {
            "funding_total": funding_total,
            "has_funding_records": has_funding_records,
            "funder_score_sum": funder_score_sum,
            "funder_score_max": funder_score_max,
            "funder_nihr_flag": funder_nihr_flag,
            "funder_ukri_flag": funder_ukri_flag,
            "funding_has_start_year_ratio": funding_has_start_year_ratio,
        },
        "societal": {
            "policy_title_hits": policy_hits,
            "clinical_title_hits": clinical_hits,
            "societal_focus_ratio": societal_focus_ratio,
        },
        "missingness": {
            "has_public_email": has_email_public,
            "has_funding_records": has_funding_records,
            "works_with_doi_ratio": doi_ratio,
            "first_pub_year_missing": 1 if first_pub_year is None else 0,
        },
    }

    return features


def fetch_orcid_profile(orcid: str, max_works: int = 200) -> Dict[str, Any]:
    person_json = get_person(orcid)
    person = parse_basic_person_info(person_json)
    works = get_works_summary(orcid)[:max_works]

    try:
        fundings = get_funding_summary(orcid)
    except Exception as e:
        fundings = []
        print(f"[WARN] funding parse failed: {e}")

    years = sorted([w["year"] for w in works if w.get("year") is not None])
    first_year = years[0] if years else None
    last_year = years[-1] if years else None
    works_with_doi = sum(1 for w in works if w.get("doi"))

    return {
        "orcid": orcid,
        "person": person,
        "works": works,
        "fundings": fundings,
        "stats": {
            "works_returned": len(works),
            "works_with_doi": works_with_doi,
            "first_pub_year_in_returned": first_year,
            "last_pub_year_in_returned": last_year,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orcid", required=True, help="ORCID like 0000-0002-1825-0097")
    ap.add_argument("--out", required=True, help="output json path, e.g. features.json")
    ap.add_argument("--max_works", type=int, default=200)
    ap.add_argument("--impact_source", choices=["none", "openalex"], default="none", help="where to fetch citations from")
    args = ap.parse_args()
    
    profile = fetch_orcid_profile(args.orcid, max_works=args.max_works)

    doi2citedby = None
    if args.impact_source == "openalex":
        dois = [w["doi"] for w in profile["works"] if w.get("doi")]
        dois = list(dict.fromkeys(dois))
        doi2citedby = openalex_cited_by_for_dois(dois)

    features = compute_features(profile, doi2citedby=doi2citedby)


    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote features to: {args.out}")
    print(json.dumps({
        "meta": features["meta"],
        "career": features["career"],
        "outputs": {k: features["outputs"][k] for k in [
            "works_total","works_recent_3y","works_recent_5y","works_with_doi_ratio","pub_score_recency"
        ]},
        "recognition": features["recognition"],
        "societal": features["societal"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
