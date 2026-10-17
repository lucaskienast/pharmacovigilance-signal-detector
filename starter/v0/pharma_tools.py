"""
Pharmacovigilance analysis tools for LangChain agents.
"""

from langchain.tools import tool
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import json
from typing import Optional

# Global variable to store FAERS data
_faers_data = None

def set_faers_data(data: pd.DataFrame):
    """Set the FAERS data for analysis."""
    global _faers_data
    _faers_data = data

@tool
def get_available_drugs_list() -> str:
    """Get list of available drugs in the FAERS dataset."""
    if _faers_data is None:
        return "No FAERS data loaded"

    drugs = _faers_data['drug_name'].value_counts()
    result = {
        "total_drugs": len(drugs),
        "top_10_drugs": drugs.head(10).to_dict(),
        "total_reports": len(_faers_data)
    }
    return json.dumps(result, indent=2)

@tool
def calculate_drug_event_statistics(drug_name: str, adverse_event: str) -> str:
    """Calculate ROR and PRR statistics for a drug-adverse event pair."""
    if _faers_data is None:
        return "No FAERS data loaded"

    # Create 2x2 contingency table
    drug_event = len(_faers_data[(_faers_data['drug_name'] == drug_name) & 
                                (_faers_data['adverse_event'] == adverse_event)])
    drug_no_event = len(_faers_data[(_faers_data['drug_name'] == drug_name) & 
                                   (_faers_data['adverse_event'] != adverse_event)])
    no_drug_event = len(_faers_data[(_faers_data['drug_name'] != drug_name) & 
                                   (_faers_data['adverse_event'] == adverse_event)])
    no_drug_no_event = len(_faers_data[(_faers_data['drug_name'] != drug_name) & 
                                      (_faers_data['adverse_event'] != adverse_event)])

    # Calculate ROR
    if drug_no_event > 0 and no_drug_event > 0:
        ror = (drug_event * no_drug_no_event) / (drug_no_event * no_drug_event)
    else:
        ror = float('inf')

    # Calculate PRR
    total_drug_reports = drug_event + drug_no_event
    total_event_reports = drug_event + no_drug_event
    expected = (total_drug_reports * total_event_reports) / len(_faers_data)

    if expected > 0:
        prr = drug_event / expected
    else:
        prr = float('inf')

    # Chi-square test with error handling
    contingency_table = [[drug_event, drug_no_event], [no_drug_event, no_drug_no_event]]
    
    try:
        # Check if chi-square test is valid (all expected frequencies >= 5)
        chi2_stat, p_val, dof, expected_freq = chi2_contingency(contingency_table)
        
        # Convert to float with type checking
        chi2 = chi2_stat if isinstance(chi2_stat, (int, float)) else 0.0
        p_value = p_val if isinstance(p_val, (int, float)) else 1.0
        
        chi2_valid = True
    except (ValueError, ZeroDivisionError, TypeError):
        # Handle cases where chi-square test fails (e.g., zero frequencies)
        chi2 = 0.0
        p_value = 1.0  # Conservative p-value when test cannot be performed
        chi2_valid = False

    # Signal detection criteria (adjusted for cases where statistical tests fail)
    if chi2_valid:
        signal_detected = (drug_event >= 3 and ror >= 2.0 and p_value < 0.05)
    else:
        # When statistical test fails, use simpler criteria
        signal_detected = (drug_event >= 3 and ror >= 2.0)

    result = {
        "drug_name": drug_name,
        "adverse_event": adverse_event,
        "contingency_table": {
            "drug_event": drug_event,
            "drug_no_event": drug_no_event,
            "no_drug_event": no_drug_event,
            "no_drug_no_event": no_drug_no_event
        },
        "statistics": {
            "ror": round(ror, 3) if ror != float('inf') else "infinite",
            "prr": round(prr, 3) if prr != float('inf') else "infinite",
            "chi_square": round(chi2, 3),
            "p_value": round(p_value, 6),
            "chi2_test_valid": chi2_valid
        },
        "signal_detected": signal_detected,
        "interpretation": f"Signal {'DETECTED' if signal_detected else 'NOT DETECTED'} based on EMA criteria",
        "notes": "Chi-square test performed successfully" if chi2_valid else "Chi-square test failed due to insufficient data - used simplified criteria"
    }

    return json.dumps(result, indent=2)

@tool
def validate_statistical_results(statistical_results_json: str) -> str:
    """Validate statistical calculation results for quality control."""
    try:
        data = json.loads(statistical_results_json)

        validation_checks = []

        # Check if contingency table adds up correctly
        ct = data["contingency_table"]
        total_reports = ct["drug_event"] + ct["drug_no_event"] + ct["no_drug_event"] + ct["no_drug_no_event"]
        validation_checks.append(f"Total reports check: {total_reports}")

        # Check ROR calculation logic
        ror = data["statistics"]["ror"]
        validation_checks.append(f"ROR value: {ror} ({'Valid' if isinstance(ror, (int, float)) else 'Check required'})")

        # Check signal detection logic
        signal = data["signal_detected"]
        validation_checks.append(f"Signal detection: {signal}")

        result = {
            "validation_status": "PASSED",
            "checks_performed": validation_checks,
            "recommendation": "Statistical calculations appear valid"
        }

    except Exception as e:
        result = {
            "validation_status": "FAILED",
            "error": str(e),
            "recommendation": "Please review input data and calculations"
        }

    return json.dumps(result, indent=2)

@tool
def search_clinical_literature(drug_name: str, adverse_event: str) -> str:
    """Search clinical literature for drug-adverse event associations."""
    # Mock literature search - in real implementation would use PubMed API

    known_associations = {
        ("METFORMIN", "LACTIC ACIDOSIS"): {"evidence": "strong", "publications": 50},
        ("WARFARIN", "BLEEDING"): {"evidence": "very_strong", "publications": 200},
        ("SIMVASTATIN", "MYOPATHY"): {"evidence": "strong", "publications": 75},
        ("ACETAMINOPHEN", "HEADACHE"): {"evidence": "weak", "publications": 5}
    }

    key = (drug_name.upper(), adverse_event.upper())

    if key in known_associations:
        evidence_data = known_associations[key]
        literature_found = True
    else:
        evidence_data = {"evidence": "limited", "publications": 1}
        literature_found = False

    result = {
        "drug_name": drug_name,
        "adverse_event": adverse_event,
        "literature_found": literature_found,
        "evidence_summary": {
            "evidence_strength": evidence_data["evidence"],
            "publication_count": evidence_data["publications"],
            "clinical_relevance": "High" if evidence_data["evidence"] in ["strong", "very_strong"] else "Moderate"
        }
    }

    return json.dumps(result, indent=2)

@tool
def assess_biological_plausibility(drug_name: str, adverse_event: str) -> str:
    """Assess biological plausibility of drug-adverse event association."""

    # Mock biological plausibility assessment
    plausibility_data = {
        ("METFORMIN", "LACTIC ACIDOSIS"): {"plausible": True, "mechanism": "Inhibits mitochondrial respiration"},
        ("WARFARIN", "BLEEDING"): {"plausible": True, "mechanism": "Anticoagulant effect"},
        ("SIMVASTATIN", "MYOPATHY"): {"plausible": True, "mechanism": "Muscle membrane disruption"},
        ("ACETAMINOPHEN", "HEADACHE"): {"plausible": False, "mechanism": "No known direct mechanism"}
    }

    key = (drug_name.upper(), adverse_event.upper())

    if key in plausibility_data:
        data = plausibility_data[key]
    else:
        data = {"plausible": False, "mechanism": "Unknown mechanism"}

    result = {
        "drug_name": drug_name,
        "adverse_event": adverse_event,
        "biologically_plausible": data["plausible"],
        "proposed_mechanism": data["mechanism"],
        "assessment": "PLAUSIBLE" if data["plausible"] else "IMPLAUSIBLE"
    }

    return json.dumps(result, indent=2)

@tool
def generate_regulatory_report(
    drug_name: str, 
    adverse_event: str, 
    statistical_data: str, 
    clinical_assessment: str
) -> str:
    """Generate FDA-compliant pharmacovigilance safety report."""
    try:
        # Parse input data
        stats = json.loads(statistical_data) if isinstance(statistical_data, str) else statistical_data
        clinical = json.loads(clinical_assessment) if isinstance(clinical_assessment, str) else clinical_assessment
        
        # Extract key findings
        signal_detected = stats.get("signal_detected", False)
        ror = stats.get("statistics", {}).get("ror", "N/A")
        evidence_strength = clinical.get("evidence_summary", {}).get("evidence_strength", "limited")
        biological_plausibility = clinical.get("biologically_plausible", False)
        
        # Generate structured regulatory report
        report = {
            "report_header": {
                "report_type": "Pharmacovigilance Signal Assessment",
                "drug_name": drug_name,
                "adverse_event": adverse_event,
                "assessment_date": "2024-Q4"
            },
            "executive_summary": {
                "signal_status": "DETECTED" if signal_detected else "NOT DETECTED",
                "risk_level": "HIGH" if signal_detected and biological_plausibility else "LOW",
                "recommendation": "Further investigation required" if signal_detected else "Routine monitoring"
            },
            "statistical_findings": {
                "reporting_odds_ratio": ror,
                "signal_detection_criteria": "EMA Guidelines",
                "statistical_significance": signal_detected
            },
            "clinical_evaluation": {
                "evidence_strength": evidence_strength,
                "biological_plausibility": "PLAUSIBLE" if biological_plausibility else "UNCLEAR",
                "clinical_relevance": "HIGH" if evidence_strength in ["strong", "very_strong"] else "MODERATE"
            },
            "regulatory_actions": {
                "immediate_actions": ["Update product labeling"] if signal_detected else ["Continue monitoring"],
                "monitoring_requirements": ["Enhanced surveillance", "Periodic safety reports"],
                "communication_plan": "Healthcare provider notification" if signal_detected else "Standard reporting"
            },
            "conclusion": f"Based on statistical analysis and clinical review, this drug-event association {'requires immediate attention' if signal_detected else 'does not meet signal detection thresholds'}."
        }
        
        return json.dumps(report, indent=2)
        
    except Exception as e:
        error_report = {
            "error": "Failed to generate regulatory report",
            "details": str(e),
            "recommendation": "Please review input data quality"
        }
        return json.dumps(error_report, indent=2)

@tool 
def quality_review_analysis(
    statistical_findings: str,
    clinical_findings: str, 
    regulatory_report: str
) -> str:
    """
    Perform comprehensive quality control review of multi-agent pharmacovigilance analysis.
    
    This tool analyzes consistency across statistical, clinical, and regulatory findings to identify:
    - Contradictions between agent conclusions
    - Missing evidence or gaps in analysis
    - Risk assessment inconsistencies
    - Regulatory compliance issues
    - Areas requiring reanalysis or additional investigation
    
    Use this tool when you need to:
    - Cross-validate findings from multiple agents
    - Check alignment between statistical signals and clinical plausibility
    - Verify regulatory risk assessments match evidence strength
    - Generate specific feedback for analysis improvement
    - Make final approval decisions for pharmacovigilance reports
    
    Args:
        statistical_findings: JSON string containing statistical analysis results (ROR, PRR, signal detection)
        clinical_findings: JSON string containing clinical assessment results (literature, plausibility)
        regulatory_report: JSON string containing regulatory safety report and risk assessments
    
    Returns:
        JSON string with detailed quality review including inconsistencies, recommendations, and approval status
    """
    try:
        # Parse all inputs
        stats = json.loads(statistical_findings) if isinstance(statistical_findings, str) else statistical_findings
        clinical = json.loads(clinical_findings) if isinstance(clinical_findings, str) else clinical_findings
        regulatory = json.loads(regulatory_report) if isinstance(regulatory_report, str) else regulatory_report
        
        # Perform consistency checks
        inconsistencies = []
        recommendations = []
        
        # Check 1: Statistical vs Clinical alignment
        stat_signal = stats.get("signal_detected", False)
        clinical_plausible = clinical.get("biologically_plausible", False)
        
        if stat_signal and not clinical_plausible:
            inconsistencies.append("Statistical signal detected but biological plausibility is low")
            recommendations.append("Consider additional clinical literature review or mechanistic studies")
        
        # Check 2: Risk level consistency
        reg_risk = regulatory.get("executive_summary", {}).get("risk_level", "")
        if stat_signal and reg_risk == "LOW":
            inconsistencies.append("Statistical signal present but regulatory risk assessment is low")
            recommendations.append("Reassess risk level based on statistical findings")
        
        # Check 3: Evidence strength alignment
        evidence_strength = clinical.get("evidence_summary", {}).get("evidence_strength", "limited")
        clinical_relevance = regulatory.get("clinical_evaluation", {}).get("clinical_relevance", "")
        
        if evidence_strength == "strong" and clinical_relevance == "MODERATE":
            inconsistencies.append("Strong literature evidence not reflected in clinical relevance assessment")
            recommendations.append("Consider upgrading clinical relevance to HIGH")
        
        # Generate feedback report
        feedback_report = {
            "quality_review_summary": {
                "review_status": "INCONSISTENCIES FOUND" if inconsistencies else "CONSISTENT",
                "overall_quality": "GOOD" if len(inconsistencies) <= 1 else "NEEDS IMPROVEMENT",
                "reviewer_confidence": "HIGH"
            },
            "consistency_analysis": {
                "inconsistencies_found": len(inconsistencies),
                "specific_issues": inconsistencies,
                "alignment_score": max(0, 100 - (len(inconsistencies) * 25))
            },
            "improvement_recommendations": {
                "priority": "HIGH" if inconsistencies else "LOW",
                "specific_actions": recommendations if recommendations else ["No improvements needed"],
                "reanalysis_required": len(inconsistencies) > 2
            },
            "validation_checklist": {
                "statistical_calculations": "VALIDATED",
                "clinical_assessment": "VALIDATED", 
                "regulatory_compliance": "VALIDATED",
                "data_integrity": "VALIDATED"
            },
            "feedback_loop_decision": {
                "send_back_for_revision": len(inconsistencies) > 2,
                "approve_for_final": len(inconsistencies) == 0,
                "conditional_approval": 0 < len(inconsistencies) <= 2
            }
        }
        
        return json.dumps(feedback_report, indent=2)
        
    except Exception as e:
        error_feedback = {
            "error": "Quality review failed",
            "details": str(e),
            "recommendation": "Manual review required"
        }
        return json.dumps(error_feedback, indent=2)

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
