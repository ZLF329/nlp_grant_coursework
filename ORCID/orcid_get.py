import re
import time
import requests
from typing import Any, Dict, List, Optional

ORCID_API = "https://pub.orcid.org/v3.0"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "orcid-impact-metrics/0.1 (mailto:your_email@example.com)"
}

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)


def get_json(url: str, params: Optional[dict] = None) -> Dict[str, Any]:
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_doi(external_ids: Dict[str, Any]) -> Optional[str]:
    """
    ORCID works external-ids 结构比较“绕”，这里尽量稳地提取 DOI。
    """
    if not external_ids:
        return None
    items = external_ids.get("external-id", [])
    for it in items:
        id_type = (it.get("external-id-type") or "").lower()
        val = it.get("external-id-value")
        if id_type == "doi" and val:
            return val.strip()

        # 有时不是 doi type，但 value 里带 doi
        if val:
            m = DOI_RE.search(val)
            if m:
                return m.group(0)
    return None


def get_person(orcid: str) -> Dict[str, Any]:
    url = f"{ORCID_API}/{orcid}/person"
    return get_json(url)


def get_works_summary(orcid: str) -> List[Dict[str, Any]]:
    """
    先拿 works 的 group summary（轻量），再按需要 fill 每个 work detail。
    """
    url = f"{ORCID_API}/{orcid}/works"
    data = get_json(url)

    groups = data.get("group", []) or []
    out = []
    for g in groups:
        summaries = g.get("work-summary", []) or []
        # 一个 group 里可能多个 summary（同一 work 的不同来源），通常取第一个
        for s in summaries[:1]:
            put_code = s.get("put-code")
            title = ((s.get("title") or {}).get("title") or {}).get("value")
            year = (((s.get("publication-date") or {}).get("year") or {}).get("value"))
            wtype = s.get("type")
            doi = extract_doi(s.get("external-ids") or {})
            out.append({
                "put_code": put_code,
                "title": title,
                "year": int(year) if year and year.isdigit() else None,
                "type": wtype,
                "doi": doi,
            })
    return out


def get_work_detail(orcid: str, put_code: int) -> Dict[str, Any]:
    """
    可选：拉某一条 work 的更完整信息（可能更慢）。
    """
    url = f"{ORCID_API}/{orcid}/work/{put_code}"
    return get_json(url)


def as_value(x):
    """
    ORCID JSON 里很多字段可能是:
    - {"value": "..."} 这种 dict
    - 直接是 "..." 这种 str
    - None
    统一转成 str/None
    """
    if x is None:
        return None
    if isinstance(x, dict):
        return x.get("value")
    if isinstance(x, str):
        return x
    return str(x)


def get_funding_summary(orcid: str):
    url = f"{ORCID_API}/{orcid}/fundings"
    data = get_json(url)

    groups = data.get("group") or []
    out = []
    for g in groups:
        summaries = g.get("funding-summary") or []
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
                "start_year": int(year) if year and str(year).isdigit() else None
            })
    return out



def parse_basic_person_info(person_json: Dict[str, Any]) -> Dict[str, Any]:
    name = person_json.get("name") or {}
    given = ((name.get("given-names") or {}).get("value"))
    family = ((name.get("family-name") or {}).get("value"))

    emails = person_json.get("emails") or {}
    email_list = []
    for e in (emails.get("email") or []):
        # 只有公开邮箱才会返回
        email_list.append(e.get("email"))

    return {
        "given_name": given,
        "family_name": family,
        "full_name": " ".join([x for x in [given, family] if x]),
        "emails_public": [x for x in email_list if x],
    }


def fetch_orcid_profile(orcid: str, max_works: int = 50) -> Dict[str, Any]:
    person = get_person(orcid)
    basic = parse_basic_person_info(person)

    works = get_works_summary(orcid)
    # 只保留前 max_works 条（ORCID works 可能很多）
    works = works[:max_works]

    fundings = get_funding_summary(orcid)

    # 简单统计
    works_with_doi = sum(1 for w in works if w.get("doi"))
    years = sorted([w["year"] for w in works if w.get("year") is not None])
    first_year = years[0] if years else None
    last_year = years[-1] if years else None

    return {
        "orcid": orcid,
        "person": basic,
        "works": works,
        "fundings": fundings,
        "stats": {
            "works_returned": len(works),
            "works_with_doi": works_with_doi,
            "first_pub_year_in_returned": first_year,
            "last_pub_year_in_returned": last_year,
        }
    }


if __name__ == "__main__":
    ORCID = "0000-0002-7488-5564"  # 换成你的
    profile = fetch_orcid_profile(ORCID, max_works=100)

    print("== PERSON ==")
    print(profile["person"])
    print("\n== STATS ==")
    print(profile["stats"])

    print("\n== WORKS (first 5) ==")
    for w in profile["works"][:5]:
        print(w)

    print("\n== FUNDINGS (first 5) ==")
    for f in profile["fundings"][:5]:
        print(f)
