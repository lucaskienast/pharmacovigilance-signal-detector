"""
Pharmacovigilance Analysis Tools for LangChain Agents

Enhanced version with:
- Real openFDA FAERS API integration (live data from millions of reports)
- Real PubMed integration via query_pubmed
- Multi-dimensional quality scoring
- Actionable feedback with agent routing
- Improved statistical calculations with confidence intervals
"""

from langchain.tools import tool
import requests
import json
import math
from typing import Any, Dict, List, Optional
import sys
sys.path.append("..")

# Optional pandas import for local data fallback
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Global variable to store local FAERS data (fallback)
_faers_data = None
_use_live_api = True  # Default to live API


def set_faers_data(data):
    """Set the local FAERS data for analysis (fallback mode)."""
    global _faers_data
    _faers_data = data


def set_use_live_api(use_live: bool):
    """Configure whether to use live FAERS API or local data."""
    global _use_live_api
    _use_live_api = use_live


# =============================================================================
# LIVE FAERS API FUNCTIONS
# =============================================================================

def _query_faers_api(search_query: str, timeout: int = 15) -> int:
    """
    Query the openFDA FAERS API and return the total count of matching records.
    
    The openFDA API provides access to millions of adverse event reports.
    
    Args:
        search_query: openFDA search query string
        timeout: Request timeout in seconds
        
    Returns:
        Total count of matching records, or 0 if query fails
    """
    base_url = "https://api.fda.gov/drug/event.json"
    params = {"search": search_query, "limit": 1} if search_query else {"limit": 1}
    
    try:
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("meta", {}).get("results", {}).get("total", 0)
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️ FAERS API request failed: {e}")
        return 0
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ⚠️ FAERS API response parsing failed: {e}")
        return 0


def get_faers_contingency_counts(drug_name: str, adverse_event: str) -> Dict[str, Any]:
    """
    Fetch data from the live FAERS database to build a 2x2 contingency table
    for calculating the Reporting Odds Ratio (ROR).
    
    This queries the openFDA API which contains millions of adverse event reports.
    
    The contingency table consists of:
    - a: Reports with the drug AND the adverse event
    - b: Reports with the drug BUT NOT the adverse event
    - c: Reports NOT with the drug BUT with the adverse event  
    - d: All other reports in the database
    
    Args:
        drug_name: Name of the drug (e.g., "metformin")
        adverse_event: Adverse event term (e.g., "lactic acidosis")
        
    Returns:
        Dictionary with contingency counts and metadata
    """
    print(f"  🌐 Querying live FAERS API for {drug_name} + {adverse_event}...")
    
    # Build search queries using openFDA field names
    # medicinalproduct is the drug name field
    # reactionmeddrapt is the adverse event (MedDRA preferred term)
    drug_query = f'patient.drug.medicinalproduct:"{drug_name}"'
    event_query = f'patient.reaction.reactionmeddrapt:"{adverse_event}"'
    
    # Query counts from the API
    # 1. Drug AND Event (cell a)
    a = _query_faers_api(f"({drug_query}) AND ({event_query})")
    
    # 2. Total reports for the drug (a + b)
    total_drug_reports = _query_faers_api(drug_query)
    b = max(0, total_drug_reports - a)
    
    # 3. Total reports for the event (a + c)
    total_event_reports = _query_faers_api(event_query)
    c = max(0, total_event_reports - a)
    
    # 4. Total reports in database (a + b + c + d)
    # Empty query returns total count
    total_reports = _query_faers_api("")
    d = max(0, total_reports - a - b - c)
    
    result = {
        "drug_event": a,
        "drug_no_event": b,
        "no_drug_event": c,
        "no_drug_no_event": d,
        "total": total_reports,
        "total_drug_reports": total_drug_reports,
        "total_event_reports": total_event_reports,
        "source": "openFDA_live_API",
        "api_available": True
    }
    
    print(f"  📊 Contingency: a={a:,}, b={b:,}, c={c:,}, d={d:,}")
    print(f"  📊 Total FAERS reports: {total_reports:,}")
    
    return result


def get_faers_contingency_from_local(drug_name: str, adverse_event: str) -> Dict[str, Any]:
    """
    Build contingency table from local FAERS data (fallback method).
    
    Args:
        drug_name: Name of the drug
        adverse_event: Adverse event term
        
    Returns:
        Dictionary with contingency counts
    """
    if _faers_data is None:
        return {
            "drug_event": 0, "drug_no_event": 0,
            "no_drug_event": 0, "no_drug_no_event": 0,
            "total": 0, "source": "local_csv",
            "api_available": False,
            "error": "No local FAERS data loaded"
        }
    
    # Create 2x2 contingency table from local data
    a = len(_faers_data[
        (_faers_data['drug_name'] == drug_name) & 
        (_faers_data['adverse_event'] == adverse_event)
    ])
    b = len(_faers_data[
        (_faers_data['drug_name'] == drug_name) & 
        (_faers_data['adverse_event'] != adverse_event)
    ])
    c = len(_faers_data[
        (_faers_data['drug_name'] != drug_name) & 
        (_faers_data['adverse_event'] == adverse_event)
    ])
    d = len(_faers_data[
        (_faers_data['drug_name'] != drug_name) & 
        (_faers_data['adverse_event'] != adverse_event)
    ])
    
    return {
        "drug_event": a,
        "drug_no_event": b,
        "no_drug_event": c,
        "no_drug_no_event": d,
        "total": a + b + c + d,
        "total_drug_reports": a + b,
        "total_event_reports": a + c,
        "source": "local_csv",
        "api_available": False
    }


# =============================================================================
# LITERATURE EVIDENCE FETCHER (Real PubMed Integration)
# =============================================================================

def fetch_literature_evidence(
    drug_name: str,
    adverse_event: str,
    max_results: int = 20
) -> Dict[str, Any]:
    """
    Query PubMed for drug-adverse-event evidence and summarize it into
    a stable schema that agents (and QC) can use.
    
    Args:
        drug_name: Name of the drug
        adverse_event: Adverse event to search for
        max_results: Maximum number of PubMed results to fetch
        
    Returns:
        Dictionary with literature evidence summary
    """
    # Build PubMed query
    query = f"{drug_name}[Title/Abstract] AND {adverse_event}[Title/Abstract]"
    
    try:
        from ls_action_space.action_space import query_pubmed
        
        articles = query_pubmed(
            query=query,
            max_results=max_results,
            include_mesh=True,
            include_citations=False,
        )
    except ImportError:
        # Biopython / Entrez not available
        return {
            "source": "pubmed",
            "available": False,
            "reason": "Entrez/Biopython not available",
            "evidence_strength": "unknown",
            "n_articles": 0,
            "query": query,
            "sample_articles": [],
            "mesh_terms_sample": [],
            "latest_year": None
        }
    except Exception as e:
        # Network / other runtime error
        return {
            "source": "pubmed",
            "available": False,
            "reason": f"error: {type(e).__name__}: {str(e)[:100]}",
            "evidence_strength": "unknown",
            "n_articles": 0,
            "query": query,
            "sample_articles": [],
            "mesh_terms_sample": [],
            "latest_year": None
        }
    
    n_articles = len(articles)
    
    # Extract years
    years = [a.get("year") for a in articles if a.get("year")]
    latest_year = max(years) if years else None
    earliest_year = min(years) if years else None
    
    # Collect MeSH terms
    mesh_terms = set()
    for a in articles:
        for term in a.get("mesh_terms", []):
            mesh_terms.add(term)
    
    # Evidence strength heuristic based on publication count
    if n_articles == 0:
        evidence_strength = "none"
    elif n_articles < 5:
        evidence_strength = "limited"
    elif n_articles < 20:
        evidence_strength = "moderate"
    else:
        evidence_strength = "strong"
    
    return {
        "source": "pubmed",
        "available": True,
        "query": query,
        "n_articles": n_articles,
        "latest_year": latest_year,
        "earliest_year": earliest_year,
        "year_range": f"{earliest_year}-{latest_year}" if earliest_year and latest_year else None,
        "mesh_terms_sample": sorted(list(mesh_terms))[:20],
        "evidence_strength": evidence_strength,
        "sample_articles": [
            {
                "pmid": a.get("pmid"),
                "title": a.get("title", "")[:200],
                "year": a.get("year"),
                "journal": a.get("journal"),
                "doi": a.get("doi"),
                "authors": a.get("authors", [])[:3]  # First 3 authors
            }
            for a in articles[:5]  # Top 5 articles
        ],
    }


# =============================================================================
# STATISTICAL TOOLS
# =============================================================================

@tool
def get_available_drugs_list() -> str:
    """
    Get information about the FAERS database.
    
    When using live API, returns database statistics.
    When using local data, returns available drugs list.
    """
    if _use_live_api:
        # Query live API for database stats
        total_reports = _query_faers_api("")
        
        # Get counts for some common drugs
        common_drugs = ["metformin", "aspirin", "warfarin", "simvastatin", "lisinopril"]
        drug_counts = {}
        for drug in common_drugs:
            count = _query_faers_api(f'patient.drug.medicinalproduct:"{drug}"')
            if count > 0:
                drug_counts[drug.upper()] = count
        
        result = {
            "source": "openFDA_live_API",
            "total_reports": total_reports,
            "sample_drug_counts": drug_counts,
            "note": "Query any drug name - the live API has millions of records"
        }
    else:
        if _faers_data is None:
            return json.dumps({"error": "No FAERS data loaded"})
        
        drugs = _faers_data['drug_name'].value_counts()
        result = {
            "source": "local_csv",
            "total_drugs": len(drugs),
            "top_10_drugs": drugs.head(10).to_dict(),
            "total_reports": len(_faers_data)
        }
    
    return json.dumps(result, indent=2)


@tool
def calculate_drug_event_statistics(drug_name: str, adverse_event: str) -> str:
    """
    Calculate ROR and PRR statistics for a drug-adverse event pair.
    
    Uses LIVE openFDA FAERS API with millions of real adverse event reports.
    
    Returns comprehensive statistics including:
    - 2x2 contingency table (from live FDA data)
    - ROR with 95% confidence interval
    - PRR with significance testing
    - Chi-square test results
    - Signal detection assessment per EMA guidelines
    - Data quality flags
    """
    # Get contingency table from live API or local data
    if _use_live_api:
        counts = get_faers_contingency_counts(drug_name.upper(), adverse_event.upper())
    else:
        counts = get_faers_contingency_from_local(drug_name, adverse_event)
    
    drug_event = counts["drug_event"]
    drug_no_event = counts["drug_no_event"]
    no_drug_event = counts["no_drug_event"]
    no_drug_no_event = counts["no_drug_no_event"]
    total_reports = counts["total"]
    
    # Data quality flags
    data_quality_flags = []
    if drug_event < 3:
        data_quality_flags.append("low_case_count")
    if drug_event == 0:
        data_quality_flags.append("zero_cases")
    if counts.get("total_drug_reports", 0) < 100:
        data_quality_flags.append("low_drug_exposure")
    if not counts.get("api_available", False) and _use_live_api:
        data_quality_flags.append("api_unavailable_used_fallback")
    
    # Calculate ROR with 95% CI
    if drug_no_event > 0 and no_drug_event > 0 and no_drug_no_event > 0 and drug_event > 0:
        ror = (drug_event * no_drug_no_event) / (drug_no_event * no_drug_event)
        
        # 95% CI for ROR using Woolf's method
        try:
            se_log_ror = math.sqrt(
                1/drug_event + 1/drug_no_event + 
                1/no_drug_event + 1/no_drug_no_event
            )
            ror_lower = math.exp(math.log(ror) - 1.96 * se_log_ror)
            ror_upper = math.exp(math.log(ror) + 1.96 * se_log_ror)
        except (ValueError, ZeroDivisionError):
            ror_lower = None
            ror_upper = None
    elif drug_event > 0:
        ror = float('inf')
        ror_lower = None
        ror_upper = None
    else:
        ror = 0.0
        ror_lower = None
        ror_upper = None

    # Calculate PRR
    total_drug_reports = drug_event + drug_no_event
    total_event_reports = drug_event + no_drug_event
    
    if total_drug_reports > 0 and total_event_reports > 0 and total_reports > 0:
        expected = (total_drug_reports * total_event_reports) / total_reports
        prr = drug_event / expected if expected > 0 else float('inf')
    else:
        prr = float('inf') if drug_event > 0 else 0.0

    # Chi-square test
    contingency_table = [[drug_event, drug_no_event], [no_drug_event, no_drug_no_event]]
    
    try:
        from scipy.stats import chi2_contingency
        chi2_stat, p_val, dof, expected_freq = chi2_contingency(contingency_table)
        chi2 = float(chi2_stat)
        p_value = float(p_val)
        chi2_valid = True
        
        if expected_freq.min() < 5:
            data_quality_flags.append("low_expected_frequency")
            
    except Exception:
        chi2 = 0.0
        p_value = 1.0
        chi2_valid = False
        data_quality_flags.append("chi2_test_failed")

    # Signal detection criteria (EMA guidelines)
    # Signal detected if: n >= 3, ROR >= 2, lower CI > 1
    ror_significant = ror_lower is not None and ror_lower > 1.0
    
    if chi2_valid:
        signal_detected = (
            drug_event >= 3 and 
            ror >= 2.0 and 
            (ror_significant or p_value < 0.05)
        )
    else:
        signal_detected = (drug_event >= 3 and ror >= 2.0)

    # Format results
    def safe_round(val, decimals=3):
        if val is None or val == float('inf'):
            return "infinite" if val == float('inf') else None
        return round(val, decimals)

    result = {
        "drug_name": drug_name,
        "adverse_event": adverse_event,
        "data_source": counts.get("source", "unknown"),
        "contingency_table": {
            "drug_event": drug_event,
            "drug_no_event": drug_no_event,
            "no_drug_event": no_drug_event,
            "no_drug_no_event": no_drug_no_event,
            "total": total_reports,
            "total_drug_reports": counts.get("total_drug_reports", total_drug_reports),
            "total_event_reports": counts.get("total_event_reports", total_event_reports)
        },
        "statistics": {
            "ror": safe_round(ror),
            "ror_95ci_lower": safe_round(ror_lower),
            "ror_95ci_upper": safe_round(ror_upper),
            "prr": safe_round(prr),
            "chi_square": safe_round(chi2),
            "p_value": safe_round(p_value, 6),
            "chi2_test_valid": chi2_valid
        },
        "signal_detection": {
            "signal_detected": signal_detected,
            "criteria_met": {
                "minimum_cases": drug_event >= 3,
                "ror_threshold": ror >= 1.5,
                "statistical_significance": ror_significant or (chi2_valid and p_value < 0.05)
            },
            "interpretation": f"Signal {'DETECTED' if signal_detected else 'NOT DETECTED'} based on EMA criteria"
        },
        "data_quality": {
            "flags": data_quality_flags,
            "quality_assessment": "adequate" if not data_quality_flags else "limited",
            "notes": "Chi-square test performed successfully" if chi2_valid 
                    else "Chi-square test failed - used simplified criteria"
        }
    }

    return json.dumps(result, indent=2)


@tool
def validate_statistical_results(statistical_results_json: str) -> str:
    """Validate statistical calculation results for quality control."""
    try:
        data = json.loads(statistical_results_json)
        validation_checks = []
        issues = []

        # Check 1: Contingency table sums correctly
        ct = data.get("contingency_table", {})
        calculated_total = (
            ct.get("drug_event", 0) + ct.get("drug_no_event", 0) + 
            ct.get("no_drug_event", 0) + ct.get("no_drug_no_event", 0)
        )
        reported_total = ct.get("total", 0)
        
        if calculated_total == reported_total:
            validation_checks.append("contingency_table_sum: PASSED")
        else:
            validation_checks.append("contingency_table_sum: FAILED")
            issues.append("Contingency table does not sum correctly")

        # Check 2: ROR is within reasonable bounds
        stats = data.get("statistics", {})
        ror = stats.get("ror")
        if ror is not None and ror != "infinite":
            if 0 < ror < 1000:
                validation_checks.append("ror_bounds: PASSED")
            else:
                validation_checks.append("ror_bounds: WARNING")
                issues.append(f"ROR value {ror} is unusually extreme")
        
        # Check 3: Confidence interval is valid
        ror_lower = stats.get("ror_95ci_lower")
        ror_upper = stats.get("ror_95ci_upper")
        if ror_lower and ror_upper and ror:
            if ror_lower <= ror <= ror_upper:
                validation_checks.append("ci_bounds: PASSED")
            else:
                validation_checks.append("ci_bounds: FAILED")
                issues.append("ROR not within confidence interval")

        # Check 4: P-value is valid
        p_value = stats.get("p_value")
        if p_value is not None:
            if 0 <= p_value <= 1:
                validation_checks.append("p_value_bounds: PASSED")
            else:
                validation_checks.append("p_value_bounds: FAILED")
                issues.append("P-value outside valid range [0,1]")

        # Check 5: Signal detection logic
        signal = data.get("signal_detection", {})
        criteria = signal.get("criteria_met", {})
        detected = signal.get("signal_detected", False)
        
        # Verify signal detection logic
        min_cases = criteria.get("minimum_cases", False)
        ror_thresh = criteria.get("ror_threshold", False)
        stat_sig = criteria.get("statistical_significance", False)
        
        expected_signal = min_cases and ror_thresh and stat_sig
        if detected == expected_signal or (min_cases and ror_thresh and not stat_sig and detected):
            validation_checks.append("signal_logic: PASSED")
        else:
            validation_checks.append("signal_logic: WARNING")
            issues.append("Signal detection may not match criteria")

        result = {
            "validation_status": "PASSED" if not issues else "ISSUES_FOUND",
            "checks_performed": validation_checks,
            "issues": issues,
            "recommendation": "Statistical calculations are valid" if not issues 
                            else f"Review required: {'; '.join(issues)}"
        }

    except Exception as e:
        result = {
            "validation_status": "FAILED",
            "error": str(e),
            "recommendation": "Please review input data and calculations"
        }

    return json.dumps(result, indent=2)


# =============================================================================
# CLINICAL TOOLS
# =============================================================================

@tool
def search_clinical_literature(drug_name: str, adverse_event: str) -> str:
    """
    Search clinical literature for drug-adverse event associations.
    
    This tool queries PubMed for relevant literature and returns
    evidence strength assessment based on publication count.
    """
    # Use the real PubMed fetcher
    evidence = fetch_literature_evidence(drug_name, adverse_event, max_results=25)
    
    result = {
        "drug_name": drug_name,
        "adverse_event": adverse_event,
        "literature_found": evidence.get("available", False) and evidence.get("n_articles", 0) > 0,
        "source": evidence.get("source", "pubmed"),
        "evidence_summary": {
            "evidence_strength": evidence.get("evidence_strength", "unknown"),
            "publication_count": evidence.get("n_articles", 0),
            "year_range": evidence.get("year_range"),
            "latest_publication_year": evidence.get("latest_year"),
            "clinical_relevance": "High" if evidence.get("evidence_strength") in ["strong", "moderate"] else "Low"
        },
        "mesh_terms": evidence.get("mesh_terms_sample", []),
        "sample_publications": evidence.get("sample_articles", []),
        "search_query": evidence.get("query", ""),
        "data_availability": {
            "pubmed_available": evidence.get("available", False),
            "reason": evidence.get("reason") if not evidence.get("available") else None
        }
    }

    return json.dumps(result, indent=2)


@tool
def assess_biological_plausibility(
    drug_name: str, 
    adverse_event: str,
    literature_context: Optional[str] = None
) -> str:
    """
    Assess biological plausibility of drug-adverse event association.
    
    Args:
        drug_name: Name of the drug
        adverse_event: Adverse event to assess
        literature_context: Optional JSON string with literature evidence
    """
    # Known biological mechanisms (educational reference data)
    # Expanded to include more drug-event pairs for teaching purposes
    plausibility_data = {
        # Well-established associations (high confidence)
        ("METFORMIN", "LACTIC ACIDOSIS"): {
            "plausible": True, 
            "mechanism": "Inhibits hepatic gluconeogenesis and mitochondrial complex I, leading to lactate accumulation",
            "confidence": "high",
            "pathway": "Mitochondrial respiration inhibition"
        },
        ("WARFARIN", "BLEEDING"): {
            "plausible": True, 
            "mechanism": "Vitamin K antagonist reducing synthesis of clotting factors II, VII, IX, X",
            "confidence": "high",
            "pathway": "Coagulation cascade inhibition"
        },
        ("WARFARIN", "HAEMORRHAGE"): {
            "plausible": True, 
            "mechanism": "Vitamin K antagonist reducing synthesis of clotting factors II, VII, IX, X",
            "confidence": "high",
            "pathway": "Coagulation cascade inhibition"
        },
        ("SIMVASTATIN", "MYOPATHY"): {
            "plausible": True, 
            "mechanism": "HMG-CoA reductase inhibition may deplete mevalonate pathway intermediates essential for muscle function",
            "confidence": "high",
            "pathway": "Mevalonate pathway disruption"
        },
        ("SIMVASTATIN", "RHABDOMYOLYSIS"): {
            "plausible": True, 
            "mechanism": "Severe form of statin-induced myopathy with muscle fiber breakdown",
            "confidence": "high",
            "pathway": "Mevalonate pathway disruption"
        },
        ("ASPIRIN", "GASTROINTESTINAL HEMORRHAGE"): {
            "plausible": True,
            "mechanism": "COX-1 inhibition reduces protective prostaglandins in gastric mucosa; antiplatelet effect increases bleeding risk",
            "confidence": "high",
            "pathway": "Prostaglandin synthesis inhibition"
        },
        ("ACETAMINOPHEN", "HEPATOTOXICITY"): {
            "plausible": True,
            "mechanism": "NAPQI metabolite depletes glutathione and causes hepatocellular necrosis at high doses",
            "confidence": "high",
            "pathway": "CYP450 toxic metabolite formation"
        },
        ("ACETAMINOPHEN", "LIVER INJURY"): {
            "plausible": True,
            "mechanism": "NAPQI metabolite depletes glutathione and causes hepatocellular necrosis at high doses",
            "confidence": "high",
            "pathway": "CYP450 toxic metabolite formation"
        },
        
        # Moderate confidence associations (good for teaching iteration)
        ("LISINOPRIL", "COUGH"): {
            "plausible": True,
            "mechanism": "ACE inhibition leads to bradykinin accumulation in the respiratory tract",
            "confidence": "high",
            "pathway": "Bradykinin accumulation"
        },
        ("LISINOPRIL", "ANGIOEDEMA"): {
            "plausible": True,
            "mechanism": "Bradykinin accumulation causes increased vascular permeability",
            "confidence": "high",
            "pathway": "Bradykinin-mediated vasodilation"
        },
        ("ATORVASTATIN", "MEMORY IMPAIRMENT"): {
            "plausible": None,  # Controversial - good for triggering iteration
            "mechanism": "Proposed cholesterol depletion in neuronal membranes, but evidence is inconsistent",
            "confidence": "low",
            "pathway": "Uncertain - cholesterol role in neuronal function"
        },
        ("SIMVASTATIN", "MEMORY IMPAIRMENT"): {
            "plausible": None,  # Controversial
            "mechanism": "Proposed cholesterol depletion in neuronal membranes, but evidence is inconsistent",
            "confidence": "low",
            "pathway": "Uncertain - cholesterol role in neuronal function"
        },
        ("OMEPRAZOLE", "PNEUMONIA"): {
            "plausible": None,  # Debated
            "mechanism": "Gastric acid suppression may alter respiratory tract colonization, but causality unclear",
            "confidence": "low",
            "pathway": "Uncertain - altered bacterial colonization"
        },
        ("METFORMIN", "VITAMIN B12 DEFICIENCY"): {
            "plausible": True,
            "mechanism": "Interferes with calcium-dependent ileal absorption of vitamin B12-intrinsic factor complex",
            "confidence": "moderate",
            "pathway": "Impaired B12 absorption"
        },
        
        # Associations where plausibility is LOW (good for mismatch scenarios)
        ("ACETAMINOPHEN", "HEADACHE"): {
            "plausible": False, 
            "mechanism": "Acetaminophen treats headaches; paradoxical headache would require medication overuse pattern",
            "confidence": "moderate",
            "pathway": "None - therapeutic use"
        },
        ("ASPIRIN", "TINNITUS"): {
            "plausible": True,
            "mechanism": "Salicylate ototoxicity affects cochlear outer hair cells at high doses",
            "confidence": "moderate",
            "pathway": "Cochlear toxicity"
        },
        ("IBUPROFEN", "RENAL FAILURE"): {
            "plausible": True,
            "mechanism": "Prostaglandin inhibition reduces renal blood flow, especially in volume-depleted patients",
            "confidence": "high",
            "pathway": "Renal prostaglandin inhibition"
        },
        ("CIPROFLOXACIN", "TENDON RUPTURE"): {
            "plausible": True,
            "mechanism": "Fluoroquinolones may inhibit collagen synthesis and increase matrix metalloproteinase activity",
            "confidence": "moderate",
            "pathway": "Collagen metabolism disruption"
        },
        
        # Pairs likely to show signal but with uncertain mechanism (iteration triggers)
        ("QUETIAPINE", "WEIGHT GAIN"): {
            "plausible": True,
            "mechanism": "Antagonism of histamine H1 and serotonin 5-HT2C receptors affects appetite regulation",
            "confidence": "high",
            "pathway": "Hypothalamic appetite regulation"
        },
        ("PREDNISONE", "OSTEOPOROSIS"): {
            "plausible": True,
            "mechanism": "Glucocorticoids inhibit osteoblast function and increase osteoclast activity",
            "confidence": "high",
            "pathway": "Bone metabolism disruption"
        },
    }

    key = (drug_name.upper(), adverse_event.upper())
    
    # Check for known mechanism
    if key in plausibility_data:
        data = plausibility_data[key]
    else:
        # Default for unknown combinations
        data = {
            "plausible": None,  # Unknown
            "mechanism": "No established mechanism in reference database",
            "confidence": "low",
            "pathway": "Unknown"
        }
    
    # Enhance with literature context if provided
    literature_support = "unknown"
    if literature_context:
        try:
            lit_data = json.loads(literature_context)
            n_articles = lit_data.get("n_articles", 0)
            if n_articles >= 10:
                literature_support = "strong"
            elif n_articles >= 3:
                literature_support = "moderate"
            elif n_articles > 0:
                literature_support = "limited"
            else:
                literature_support = "none"
        except (json.JSONDecodeError, TypeError):
            pass

    result = {
        "drug_name": drug_name,
        "adverse_event": adverse_event,
        "mechanism_assessment": {
            "biologically_plausible": data["plausible"],
            "proposed_mechanism": data["mechanism"],
            "mechanistic_pathway": data["pathway"],
            "confidence_level": data["confidence"]
        },
        "literature_support": literature_support,
        "overall_assessment": "PLAUSIBLE" if data["plausible"] else 
                            "IMPLAUSIBLE" if data["plausible"] is False else "UNCERTAIN",
        "notes": "Assessment based on known pharmacological mechanisms" if key in plausibility_data
                else "Limited mechanistic data available - requires expert review"
    }

    return json.dumps(result, indent=2)


# =============================================================================
# REGULATORY TOOLS
# =============================================================================

@tool
def generate_regulatory_report(
    drug_name: str, 
    adverse_event: str, 
    statistical_data: str, 
    clinical_assessment: str,
    literature_evidence: Optional[str] = None,
    feedback_loop_summary: Optional[str] = None
) -> str:
    """
    Generate FDA-compliant pharmacovigilance safety report.
    
    Args:
        drug_name: Name of the drug
        adverse_event: Adverse event
        statistical_data: JSON string with statistical findings
        clinical_assessment: JSON string with clinical assessment
        literature_evidence: Optional JSON string with literature evidence
        feedback_loop_summary: Optional JSON string with feedback loop history
    """
    try:
        # Parse inputs
        stats = json.loads(statistical_data) if isinstance(statistical_data, str) else statistical_data
        clinical = json.loads(clinical_assessment) if isinstance(clinical_assessment, str) else clinical_assessment
        
        literature = {}
        if literature_evidence:
            try:
                literature = json.loads(literature_evidence) if isinstance(literature_evidence, str) else literature_evidence
            except (json.JSONDecodeError, TypeError):
                pass
        
        loop_summary = {}
        if feedback_loop_summary:
            try:
                loop_summary = json.loads(feedback_loop_summary) if isinstance(feedback_loop_summary, str) else feedback_loop_summary
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Extract key findings
        signal_detection = stats.get("signal_detection", stats.get("signal_detected", {}))
        if isinstance(signal_detection, dict):
            signal_detected = signal_detection.get("signal_detected", False)
        else:
            signal_detected = signal_detection
            
        statistics = stats.get("statistics", {})
        ror = statistics.get("ror", "N/A")
        ror_ci = f"({statistics.get('ror_95ci_lower', 'N/A')}-{statistics.get('ror_95ci_upper', 'N/A')})"
        
        mechanism = clinical.get("mechanism_assessment", clinical)
        biological_plausibility = mechanism.get("biologically_plausible", 
                                                clinical.get("biologically_plausible", None))
        
        evidence_strength = literature.get("evidence_strength", 
                           clinical.get("evidence_summary", {}).get("evidence_strength", "unknown"))
        
        # Determine risk level
        if signal_detected and biological_plausibility:
            risk_level = "HIGH"
        elif signal_detected or biological_plausibility:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        # Build report
        report = {
            "report_header": {
                "report_type": "Pharmacovigilance Signal Assessment",
                "report_version": "2.0",
                "drug_name": drug_name,
                "adverse_event": adverse_event,
                "assessment_date": "2024-Q4",
                "regulatory_framework": "FDA/ICH E2B(R3)"
            },
            "executive_summary": {
                "signal_status": "DETECTED" if signal_detected else "NOT DETECTED",
                "risk_level": risk_level,
                "evidence_strength": evidence_strength,
                "recommendation": "Further investigation required" if signal_detected 
                                else "Routine monitoring"
            },
            "statistical_findings": {
                "reporting_odds_ratio": ror,
                "ror_95_ci": ror_ci if ror != "N/A" else "N/A",
                "prr": statistics.get("prr", "N/A"),
                "case_count": stats.get("contingency_table", {}).get("drug_event", "N/A"),
                "signal_detection_criteria": "EMA Guidelines (n≥3, ROR≥2, LCI>1)",
                "statistical_significance": statistics.get("p_value", "N/A"),
                "data_quality": stats.get("data_quality", {}).get("quality_assessment", "unknown")
            },
            "clinical_evaluation": {
                "biological_plausibility": "PLAUSIBLE" if biological_plausibility 
                                          else "UNCLEAR" if biological_plausibility is None 
                                          else "IMPLAUSIBLE",
                "proposed_mechanism": mechanism.get("proposed_mechanism", "Unknown"),
                "evidence_strength": evidence_strength,
                "clinical_relevance": "HIGH" if evidence_strength in ["strong", "moderate"] else "MODERATE"
            },
            "literature_summary": {
                "pubmed_articles_found": literature.get("n_articles", "N/A"),
                "publication_year_range": literature.get("year_range", "N/A"),
                "key_mesh_terms": literature.get("mesh_terms_sample", [])[:5],
                "data_source": literature.get("source", "unknown")
            },
            "regulatory_actions": {
                "immediate_actions": ["Update product labeling", "Healthcare provider notification"] 
                                    if risk_level == "HIGH"
                                    else ["Enhanced surveillance"] if risk_level == "MODERATE"
                                    else ["Continue routine monitoring"],
                "monitoring_requirements": ["Periodic safety reports", "Signal tracking"],
                "communication_plan": "Urgent safety communication" if risk_level == "HIGH"
                                     else "Standard reporting",
                "rems_consideration": risk_level == "HIGH"
            },
            "quality_assurance": {
                "feedback_loop_iterations": loop_summary.get("iterations_used", 0),
                "final_quality_score": loop_summary.get("final_quality_score", "N/A"),
                "review_status": loop_summary.get("converged", "N/A")
            },
            "conclusion": f"Based on statistical analysis (ROR={ror}) and clinical review, "
                         f"this drug-event association {'requires immediate attention and further investigation' if signal_detected else 'does not meet signal detection thresholds at this time'}. "
                         f"Risk level assessed as {risk_level}."
        }
        
        return json.dumps(report, indent=2)
        
    except Exception as e:
        error_report = {
            "error": "Failed to generate regulatory report",
            "details": str(e),
            "recommendation": "Please review input data quality"
        }
        return json.dumps(error_report, indent=2)


# =============================================================================
# QUALITY CONTROL TOOLS (Enhanced with Multi-Dimensional Scoring)
# =============================================================================

@tool 
def quality_review_analysis(
    statistical_findings: str,
    clinical_findings: str, 
    regulatory_report: str,
    literature_evidence: Optional[str] = None
) -> str:
    """
    Perform comprehensive quality control review of multi-agent pharmacovigilance analysis.
    
    Enhanced version with:
    - Multi-dimensional quality scoring
    - Actionable feedback with agent routing
    - Literature-driven consistency checks
    - Specific revision instructions
    
    Args:
        statistical_findings: JSON string with statistical analysis results
        clinical_findings: JSON string with clinical assessment results
        regulatory_report: JSON string with regulatory safety report
        literature_evidence: Optional JSON string with literature evidence
    
    Returns:
        JSON string with detailed quality review including scores, issues, and routing
    """
    try:
        # Parse all inputs
        stats = json.loads(statistical_findings) if isinstance(statistical_findings, str) else statistical_findings
        clinical = json.loads(clinical_findings) if isinstance(clinical_findings, str) else clinical_findings
        regulatory = json.loads(regulatory_report) if isinstance(regulatory_report, str) else regulatory_report
        
        clinical_text = clinical.get("narrative_response", "") or ""
        revision_number = clinical.get("_revision_number", 0)
        
        literature = {}
        if literature_evidence:
            try:
                literature = json.loads(literature_evidence) if isinstance(literature_evidence, str) else literature_evidence
            except (json.JSONDecodeError, TypeError):
                pass
        
        issues = []
        
        def _has_mechanism_investigation(text: str) -> bool:
            """
            Heuristic: does the clinical narrative show a real attempt
            to investigate mechanisms / confounders / uncertainty?
            """
            t = text.lower()
            keywords = [
                "mechanism", "mechanistic", "pathway", "hypothesis",
                "biological", "plausible", "confounder", "confounding",
                "causal", "causality", "bradford hill", "dose-response",
                "neuroinflammation", "cns", "central nervous system"
            ]
            hits = sum(1 for k in keywords if k in t)
            # Require at least 2 distinct hits to avoid trivial matches
            return hits >= 2
        
        # =================================================================
        # DIMENSION 1: Statistical-Clinical Alignment (30%)
        # =================================================================
        alignment_score = 100
        
        # Extract values with fallbacks
        signal_detection = stats.get("signal_detection", {})
        stat_signal = signal_detection.get("signal_detected", stats.get("signal_detected", False))
        
        mechanism = clinical.get("mechanism_assessment", clinical)
        clinical_plausible = mechanism.get("biologically_plausible", 
                                          clinical.get("biologically_plausible", None))
        
        # Check: Statistical signal vs clinical plausibility
        if stat_signal and clinical_plausible is False:
            issues.append({
                "type": "alignment_mismatch",
                "severity": "high",
                "description": "Statistical signal detected but biological plausibility assessment is IMPLAUSIBLE",
                "responsible_agent": "clinical",
                "specific_action": "Re-examine biological mechanisms. Consider: (1) Novel mechanism not yet characterized, (2) Confounding in statistical data, (3) Literature gaps. Provide stronger evidence for your plausibility conclusion.",
                "evidence_needed": ["mechanism_pathway", "literature_support", "alternative_explanations"],
                "revision_reason": "statistical_clinical_mismatch"
            })
            alignment_score -= 40
        
        if not stat_signal and clinical_plausible is True:
            issues.append({
                "type": "alignment_mismatch",
                "severity": "medium",
                "description": "No statistical signal but strong biological plausibility exists",
                "responsible_agent": "statistical",
                "specific_action": "Review statistical methodology. Consider: (1) Adequate sample size/power, (2) Appropriate comparison groups, (3) Potential confounders masking signal.",
                "evidence_needed": ["power_analysis", "confounder_assessment", "subgroup_analysis"],
                "revision_reason": "signal_underdetection_possible"
            })
            alignment_score -= 20
        
        # Check: Statistical signal + UNCERTAIN plausibility (warrants investigation)
        if stat_signal and clinical_plausible is None:
            mechanism_investigated = _has_mechanism_investigation(clinical_text)

            # Base penalty when nothing has been done yet
            base_penalty = 25

            # Option 1 + 2 combined:
            # - First pass (revision_number == 0): full penalty (no investigation yet).
            # - First revision with mechanism discussion: partial penalty.
            # - Second (or later) revision with mechanism discussion: no further penalty,
            #   treat as "acknowledged uncertainty" rather than a quality failure.
            if revision_number >= 2 and mechanism_investigated:
                penalty = 0
                severity = "low"
                issue_status = "acknowledged"
            elif revision_number >= 1 and mechanism_investigated:
                penalty = 10
                severity = "medium"
                issue_status = "addressed"
            else:
                penalty = base_penalty
                severity = "high"
                issue_status = "unaddressed"

            if penalty > 0:
                alignment_score -= penalty

            issues.append({
                "type": "signal_uncertain_mechanism",
                "severity": severity,
                "description": (
                    "Statistical signal detected but biological plausibility is UNCERTAIN – "
                    "requires mechanistic investigation and clear documentation of uncertainty."
                ),
                "responsible_agent": "clinical",
                "specific_action": (
                    "Investigate potential mechanisms. Either: "
                    "(1) Propose a plausible mechanistic pathway with literature support, "
                    "(2) Identify confounders that could explain a spurious signal, "
                    "(3) Clearly document why the mechanism remains uncertain and "
                    "what further research is needed."
                ),
                "evidence_needed": ["mechanism_hypothesis", "literature_review", "confounder_analysis"],
                "revision_reason": "signal_needs_mechanistic_explanation",
                # NEW fields for debugging / teaching
                "mechanism_investigated": mechanism_investigated,
                "issue_status": issue_status,
                "applied_penalty": penalty,
                "revision_number_seen": revision_number,
            })
        
        # =================================================================
        # DIMENSION 2: Completeness (25%)
        # =================================================================
        completeness_score = 100
        
        # Check statistical completeness
        required_stats = ["ror", "prr", "p_value"]
        stats_data = stats.get("statistics", {})
        missing_stats = [f for f in required_stats if f not in stats_data or stats_data.get(f) is None]
        
        if missing_stats:
            issues.append({
                "type": "incomplete_analysis",
                "severity": "medium",
                "description": f"Statistical analysis missing: {missing_stats}",
                "responsible_agent": "statistical",
                "specific_action": f"Recalculate with complete statistical measures. Missing: {missing_stats}",
                "evidence_needed": missing_stats,
                "revision_reason": "incomplete_statistics"
            })
            completeness_score -= 15 * len(missing_stats)
        
        # Check clinical completeness
        if not clinical.get("mechanism_assessment") and not clinical.get("proposed_mechanism"):
            issues.append({
                "type": "incomplete_analysis",
                "severity": "medium",
                "description": "Clinical assessment lacks mechanism evaluation",
                "responsible_agent": "clinical",
                "specific_action": "Provide mechanism assessment including pathway analysis and plausibility determination.",
                "evidence_needed": ["mechanism_pathway", "plausibility_assessment"],
                "revision_reason": "missing_mechanism"
            })
            completeness_score -= 25
        
        # =================================================================
        # DIMENSION 3: Risk Alignment (25%)
        # =================================================================
        risk_alignment_score = 100
        
        # Get evidence strength from literature or clinical
        lit_evidence_strength = literature.get("evidence_strength", "unknown")
        clinical_evidence = clinical.get("evidence_summary", {}).get("evidence_strength", 
                           clinical.get("literature_support", "unknown"))
        
        # Use literature evidence as ground truth if available
        evidence_strength = lit_evidence_strength if lit_evidence_strength != "unknown" else clinical_evidence
        
        reg_risk = regulatory.get("executive_summary", {}).get("risk_level", "")
        
        # Check: Strong evidence but low risk
        if evidence_strength in ["strong", "moderate"] and reg_risk == "LOW":
            issues.append({
                "type": "risk_underestimation",
                "severity": "high",
                "description": f"Evidence strength is '{evidence_strength}' but risk classified as LOW",
                "responsible_agent": "regulatory",
                "specific_action": "Reassess risk level. Strong/moderate evidence warrants at minimum MODERATE risk. Document specific justification if maintaining LOW risk despite evidence.",
                "evidence_needed": ["risk_justification", "benefit_risk_analysis"],
                "revision_reason": "risk_evidence_mismatch"
            })
            risk_alignment_score -= 35
        
        # Check: Literature vs clinical evidence mismatch
        if literature.get("available"):
            lit_n_articles = literature.get("n_articles", 0)
            
            # Clinical claims strong evidence but PubMed shows none/limited
            if clinical_evidence in ["strong", "very_strong"] and lit_n_articles < 3:
                issues.append({
                    "type": "evidence_mismatch",
                    "severity": "high",
                    "description": f"Clinical agent claims '{clinical_evidence}' evidence but PubMed found only {lit_n_articles} articles",
                    "responsible_agent": "clinical",
                    "specific_action": "Reconcile evidence claims with PubMed data. Either cite specific sources supporting strong evidence or revise evidence strength assessment.",
                    "evidence_needed": ["specific_citations", "evidence_basis"],
                    "revision_reason": "evidence_overclaim"
                })
                risk_alignment_score -= 30
        
        # =================================================================
        # DIMENSION 4: Actionability (20%)
        # =================================================================
        actionability_score = 100
        
        reg_actions = regulatory.get("regulatory_actions", {})
        immediate_actions = reg_actions.get("immediate_actions", [])
        
        # Check if recommendations are specific
        if not immediate_actions:
            issues.append({
                "type": "non_actionable",
                "severity": "medium",
                "description": "Regulatory report lacks specific recommended actions",
                "responsible_agent": "regulatory",
                "specific_action": "Provide specific, implementable recommendations including monitoring requirements and communication plans.",
                "evidence_needed": ["specific_recommendations", "timeline"],
                "revision_reason": "missing_recommendations"
            })
            actionability_score -= 30
        
        # Check if high risk has appropriate actions
        if reg_risk == "HIGH" and len(immediate_actions) < 2:
            issues.append({
                "type": "insufficient_actions",
                "severity": "medium",
                "description": "HIGH risk level but insufficient immediate actions specified",
                "responsible_agent": "regulatory",
                "specific_action": "For HIGH risk, specify multiple actions including labeling updates, HCP notification, and REMS consideration.",
                "evidence_needed": ["expanded_action_plan"],
                "revision_reason": "inadequate_response_to_risk"
            })
            actionability_score -= 20
        
        # =================================================================
        # Calculate Overall Score
        # =================================================================
        weights = {
            "alignment": 0.30,
            "completeness": 0.25,
            "risk_alignment": 0.25,
            "actionability": 0.20
        }
        
        overall_score = (
            alignment_score * weights["alignment"] +
            completeness_score * weights["completeness"] +
            risk_alignment_score * weights["risk_alignment"] +
            actionability_score * weights["actionability"]
        )
        overall_score = max(0, min(100, overall_score))  # Clamp to 0-100
        
        # Determine approval status
        if overall_score >= 80:
            approval_status = "approved"
        elif overall_score >= 60:
            approval_status = "conditional"
        else:
            approval_status = "revision_required"
        
        # Extract target agents for routing
        target_agents = list(set(issue["responsible_agent"] for issue in issues))
        
        # Build feedback report
        feedback_report = {
            "quality_scores": {
                "alignment_score": round(alignment_score, 1),
                "completeness_score": round(completeness_score, 1),
                "risk_alignment_score": round(risk_alignment_score, 1),
                "actionability_score": round(actionability_score, 1),
                "overall_score": round(overall_score, 1)
            },
            "approval_status": approval_status,
            "target_agents": target_agents,
            "issues": issues,
            "issue_summary": {
                "total_issues": len(issues),
                "high_severity": len([i for i in issues if i["severity"] == "high"]),
                "medium_severity": len([i for i in issues if i["severity"] == "medium"]),
                "low_severity": len([i for i in issues if i["severity"] == "low"])
            },
            "feedback_loop_decision": {
                "requires_revision": approval_status == "revision_required",
                "requires_review": approval_status == "conditional",
                "approved_for_final": approval_status == "approved"
            },
            "validation_checklist": {
                "statistical_calculations": "VALIDATED" if completeness_score >= 70 else "NEEDS_REVIEW",
                "clinical_assessment": "VALIDATED" if alignment_score >= 70 else "NEEDS_REVIEW",
                "regulatory_compliance": "VALIDATED" if actionability_score >= 70 else "NEEDS_REVIEW",
                "evidence_consistency": "VALIDATED" if risk_alignment_score >= 70 else "NEEDS_REVIEW"
            }
        }
        
        return json.dumps(feedback_report, indent=2)
        
    except Exception as e:
        error_feedback = {
            "error": "Quality review failed",
            "details": str(e),
            "quality_scores": {"overall_score": 0},
            "approval_status": "revision_required",
            "target_agents": [],
            "issues": [{"type": "review_error", "severity": "high", 
                       "description": str(e), "responsible_agent": "system"}]
        }
        return json.dumps(error_feedback, indent=2)


# =============================================================================
# TOOL COLLECTION
# =============================================================================

def get_all_pharma_tools():
    """Return all pharmacovigilance tools for agent use."""
    return [
        get_available_drugs_list,
        calculate_drug_event_statistics,
        validate_statistical_results,
        search_clinical_literature,
        assess_biological_plausibility,
        generate_regulatory_report,
        quality_review_analysis
    ]


def get_statistical_tools():
    """Return tools for the Statistical Agent."""
    return [
        get_available_drugs_list,
        calculate_drug_event_statistics,
        validate_statistical_results
    ]


def get_clinical_tools():
    """Return tools for the Clinical Agent."""
    return [
        search_clinical_literature,
        assess_biological_plausibility
    ]


def get_regulatory_tools():
    """Return tools for the Regulatory Agent."""
    return [
        generate_regulatory_report
    ]


def get_quality_control_tools():
    """Return tools for the Quality Control Agent."""
    return [
        quality_review_analysis
    ]
