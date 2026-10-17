"""
Capstone: Pharmacovigilance Signal Detector with Feedback Loops

DO NOT MODIFY:
- pharma_tools.py   → FAERS + PubMed + plausibility + regulatory + QC tools
- audit_logger.py   → audit trail utilities

IN THIS FILE YOU WILL:
1. Write system prompts for your agents (statistical & clinical).
2. Assign pharma_tools functions to each agent.
3. Orchestrate the agents (statistical, clinical, regulatory, QC).
4. Implement a TRUE feedback loop driven by the QC tool.
5. Add minimal audit logging inside the loop (regulatory-style requirement).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import pharma_tools
from pharma_tools import fetch_literature_evidence
from audit_logger import AuditLogger


# =============================================================================
# 1. ENVIRONMENT / LLM SETUP (GIVEN)
# =============================================================================


def setup_environment() -> ChatOpenAI:
    """
    Initialize the OpenAI client.

    You do NOT need to change this.
    Just make sure you have a .env with OPENAI_API_KEY (and optional BASE_URL).
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")  # optional

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Add it to your .env file before running."
        )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_completion_tokens=2000,
        base_url=base_url,
    )
    return llm


# Local FAERS snapshot shipped with the project (aggregated 2025q2 drug-event case counts).
LOCAL_FAERS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "faers_signal_detection_dataset.csv"
)


def load_faers_data(csv_path: str = LOCAL_FAERS_CSV):
    """
    Load and parse the local FAERS snapshot for the offline fallback.

    The CSV holds aggregated rows (drug_name, adverse_event, case_count, ...). pharma_tools'
    local mode counts one row per drug-event report, so each row is expanded case_count times.
    Note: the snapshot only covers 5 drugs (ACETAMINOPHEN, ASPIRIN, METFORMIN, SIMVASTATIN,
    WARFARIN), so the background (cells c/d) is restricted to those drugs.

    Returns a DataFrame with columns drug_name, adverse_event (upper-case).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = {"drug_name", "adverse_event", "case_count"} - set(df.columns)
    if missing:
        raise ValueError(f"FAERS CSV {csv_path} is missing columns: {sorted(missing)}")

    df["drug_name"] = df["drug_name"].astype(str).str.strip().str.upper()
    df["adverse_event"] = df["adverse_event"].astype(str).str.strip().str.upper()
    df = df[pd.to_numeric(df["case_count"], errors="coerce").fillna(0) > 0]
    expanded = df.loc[df.index.repeat(df["case_count"].astype(int)), ["drug_name", "adverse_event"]]
    return expanded.reset_index(drop=True)


def configure_faers_mode(use_live_api: bool = True, local_csv_path: Optional[str] = None) -> None:
    """
    Switch between live FAERS API and any local fallback data.

    For this course we recommend using the LIVE API (default=True).

    Fallback (documented): configure_faers_mode(use_live_api=False) loads the local CSV with
    load_faers_data() and hands it to pharma_tools via set_faers_data(), so
    calculate_drug_event_statistics computes the 2x2 table from local data instead.
    The statistical agent also falls back to this CSV automatically if the live API is down.
    """
    if not use_live_api:
        pharma_tools.set_faers_data(load_faers_data(local_csv_path or LOCAL_FAERS_CSV))
    pharma_tools.set_use_live_api(use_live_api)


# =============================================================================
# 2. SYSTEM PROMPTS & TOOL ASSIGNMENT  (TODO 0)
# =============================================================================

# ------------------------
# TODO 0A: SYSTEM PROMPTS
# ------------------------
#
# Write effective system prompts for the two LLM-based agents.
# Use your own words, but make sure you cover at least the points in the comments.


STATISTICAL_SYSTEM_PROMPT = """
You are a Senior FDA Biostatistician specialising in pharmacovigilance signal
detection from spontaneous reporting systems (FAERS via openFDA).

INPUTS YOU RECEIVE (all produced by tools — they are your ONLY source of numbers):
- calculate_drug_event_statistics JSON: 2x2 contingency table, ROR with 95% CI,
  PRR, chi-square, p-value, signal_detection block, data_quality flags.
- validate_statistical_results JSON: validation status, checks performed, issues.
- A literature summary (PubMed article count, evidence strength, year range) for context only.
- Optionally, Quality Control (QC) feedback on your previous interpretation.

WORKFLOW (follow step by step):
1. Data access: state the data source (FAERS / openFDA) and the case count (drug_event, "a").
   If flags show zero_cases, low_case_count, api_unavailable_used_fallback or the counts look
   empty, say the result is NOT interpretable and why. Do not fill gaps with estimates.
2. Disproportionality: report ROR with its 95% CI, PRR, p-value and whether the chi-square
   test was valid, copying values exactly as they appear in the JSON.
   If p_value is 0.0, write "p-value reported as 0.0 (below the tool's rounding precision)".
3. Signal decision: apply EMA-style criteria (case count >= 3, ROR >= 2, lower 95% CI bound > 1,
   or p < 0.05 when the chi-square test is valid). List EACH criterion as MET or NOT MET with
   its tool value, so the reason for the decision is explicit. State SIGNAL DETECTED or
   SIGNAL NOT DETECTED and confirm it matches the tool's signal_detection.signal_detected field.
4. Validation: summarise the validation status. If a check shows WARNING/FAILED, explain it
   from the criteria (e.g. the tool's criteria_met.ror_threshold uses ROR >= 1.5 while the
   signal decision uses ROR >= 2, so a ROR between 1.5 and 2 triggers a signal_logic warning).
5. Limitations: explain data-quality limits that THIS result actually shows (data_quality
   flags, low case counts). Only call the CI wide or the counts low if the values support it.
   Then list inherent FAERS limits: under-reporting, stimulated/notoriety reporting, duplicate
   reports, confounding by indication, no exposure denominator (disproportionality is not
   incidence or risk).

HARD RULES:
- Use ONLY numbers present in the tool JSON. Never invent, recompute, round differently or
  extrapolate statistics. If a value is missing or "infinite", say so.
- Statistical interpretation ONLY: no clinical recommendations, no treatment or labelling advice,
  and no causality claims. A disproportionality signal is a statistical association that
  warrants clinical review, nothing more.
- Cite the source of each number, e.g. "ROR <ror> (95% CI <lower>–<upper>; FAERS/openFDA)".

QC FEEDBACK:
If the message contains QC feedback, you MUST address every point and add a section
"Response to QC Feedback" listing (a) each issue raised and (b) "What I changed" in this
revision. If an issue cannot be resolved from the available tool data (e.g. a power or
subgroup analysis), say so explicitly and explain qualitatively what the data can and cannot
show — never invent numbers to satisfy feedback.

OUTPUT FORMAT (Markdown headings, concise, professional):
## Statistical Signal Assessment
## Key Statistics (tool-derived)
## Signal Decision (EMA criteria)
## Validation & Data Quality
## Limitations
## Response to QC Feedback   (only when feedback is provided)
"""


CLINICAL_SYSTEM_PROMPT = """
You are a Clinical Pharmacologist and Director of Pharmacovigilance (drug safety physician)
assessing whether a statistical drug–adverse event signal is clinically meaningful.

INPUTS YOU RECEIVE (all produced by tools or upstream agents):
- Statistical context: the Statistical Agent's interpretation of FAERS disproportionality.
- search_clinical_literature JSON: PubMed publication_count, evidence_strength, year range,
  MeSH terms and sample publications (PMIDs, titles, years).
- assess_biological_plausibility JSON: biologically_plausible, proposed mechanism, pathway,
  confidence level and overall_assessment (PLAUSIBLE / IMPLAUSIBLE / UNCERTAIN).
- Optionally, Quality Control (QC) feedback on your previous assessment.

WORKFLOW (follow step by step):
1. Literature evidence: report the PubMed publication count and the evidence_strength label
   returned by the tool, with the year range. Cite 2–3 sample publications by PMID.
2. Evidence strength MUST be commensurate with the article count. Use the tool's label and
   never upgrade it: 0 articles = none, 1–4 = limited, 5–19 = moderate, >=20 = strong.
   Note that counts reflect a capped PubMed retrieval, and that publication count measures
   attention in the literature, not proof of harm.
3. Biological plausibility: describe the proposed mechanism and pathway from the plausibility
   tool and its confidence level. If the mechanism is unknown, propose clearly labelled
   mechanistic hypotheses grounded in the drug's known pharmacology.
4. Causality: distinguish association from causation. Weigh Bradford Hill considerations
   (strength, consistency, temporality, dose-response, biological gradient, coherence) and
   name likely confounders (confounding by indication, comorbidities, co-medications,
   stimulated reporting after safety communications). Only state what the inputs support:
   FAERS aggregate counts contain no timing or dose data, so write "not assessable from the
   available data" for criteria such as temporality or dose-response, and only call findings
   "consistent" if the cited publication titles show it.
5. Classification: give a final classification of PLAUSIBLE, IMPLAUSIBLE or UNCERTAIN.
   Keep it consistent with the plausibility tool's overall_assessment; if you disagree,
   justify it explicitly from the evidence. For UNCERTAIN, document why the mechanism remains
   unresolved and what further research (e.g. pharmacoepidemiological studies, mechanistic
   studies, case-series review) would resolve it.
6. Clinical importance: comment on the seriousness of the event and the populations that
   may be most affected, without giving individual treatment advice.

HARD RULES:
- Use only facts from the tool outputs and the statistical context. Do not invent studies,
  PMIDs, article counts, mechanisms presented as established, or statistics.
- Do not restate or alter statistical values beyond what the statistical context provides.
- Educational pharmacovigilance assessment only — no patient-specific medical advice.

QC FEEDBACK:
If the message contains QC feedback, you MUST address every point and add a section
"Response to QC Feedback" listing (a) each issue raised and (b) "What I changed" in this
revision (e.g. added mechanistic hypotheses, confounder analysis, corrected evidence strength).

OUTPUT FORMAT (Markdown headings, concise, professional):
## Clinical Assessment Summary
## Literature Evidence (PubMed)
## Biological Plausibility & Mechanism
## Causality & Confounding
## Classification: PLAUSIBLE | IMPLAUSIBLE | UNCERTAIN
## Clinical Importance
## Response to QC Feedback   (only when feedback is provided)
"""


# Optional (scaffold allows tuning regulatory behaviour): the report itself is produced
# deterministically by generate_regulatory_report. When QC targets the regulatory agent,
# this prompt drives a documented response to that feedback, without altering the report.
REGULATORY_SYSTEM_PROMPT = """
You are an FDA Regulatory Affairs Specialist (pharmacovigilance, ICH E2B(R3) / FDA reporting).

INPUTS YOU RECEIVE:
- The structured safety report produced by generate_regulatory_report (risk level, actions).
  Its risk level is computed by the report generator from the statistical signal and the
  biological plausibility; it reflects the latest agent inputs and must not be edited by hand.
- Key statistical findings, the clinical classification and PubMed evidence summary.
- Quality Control (QC) feedback on the regulatory report.

TASK: write a "Response to QC Feedback" that, for each QC issue:
1. States whether the report's risk level is maintained and gives the specific, evidence-based
   justification (e.g. no disproportionality signal by EMA criteria, mechanism assessed as
   IMPLAUSIBLE, literature mainly reflects therapeutic use), citing the input values.
2. Gives a brief benefit-risk consideration and the monitoring / communication actions that
   follow from the report (routine monitoring, enhanced surveillance, labelling review).
3. States "What I changed" (e.g. report regenerated from the latest inputs; justification
   documented) and whether escalation to human safety review is recommended.

HARD RULES: reflect the inputs faithfully; do not change the risk level, numbers or actions in
the report; do not invent evidence or numbers. Keep it concise (<= 250 words), Markdown.
"""


# -----------------------------
# TODO 0B: TOOL ASSIGNMENT
# -----------------------------
#
# Map each agent to the tools it should use from pharma_tools.py.
#
# The tools you have available (from pharma_tools):
#   - calculate_drug_event_statistics(drug_name, adverse_event)
#   - validate_statistical_results(statistical_results_json)
#   - search_clinical_literature(drug_name, adverse_event)
#   - assess_biological_plausibility(drug_name, adverse_event, literature_context=None)
#   - generate_regulatory_report(...)
#   - quality_review_analysis(...)
#
# Fill in the correct pharma_tools.<function_name> below.
# Code will crash until these are NOT None.


STATISTICAL_TOOLS = {
    # FAERS (openFDA) 2x2 table -> ROR + 95% CI, PRR, chi-square, signal detection
    "calculate_stats": pharma_tools.calculate_drug_event_statistics,
    # Sanity checks on the stats JSON (table sums, CI bounds, p-value, signal logic)
    "validate_stats": pharma_tools.validate_statistical_results,
}

CLINICAL_TOOLS = {
    # PubMed literature search -> publication count + evidence strength
    "search_literature": pharma_tools.search_clinical_literature,
    # Mechanism / pathway lookup -> PLAUSIBLE / IMPLAUSIBLE / UNCERTAIN
    "assess_plausibility": pharma_tools.assess_biological_plausibility,
}

# Structured regulatory report (risk level, actions) from stats + clinical JSON
REGULATORY_TOOL = pharma_tools.generate_regulatory_report
# Multi-dimensional QC scoring, issues and responsible-agent routing
QC_TOOL = pharma_tools.quality_review_analysis


# =============================================================================
# 3. AGENT RUNNERS  (TODO 1)
# =============================================================================
#
# You are now building thin “agent wrappers” around:
#   - FAERS tools (stats)
#   - PubMed + plausibility tools (clinical)
#   - Regulatory report tool
#
# Design choice (recommended, but up to you):
#   • Call pharma_tools.* directly to get JSON.
#   • Then ask the LLM (with your system prompt) to summarize / interpret.
#
# Returned dicts SHOULD roughly follow this shape (so QC + feedback loop can use them):
#   {
#      "agent_response": "<narrative text>",
#      "tool_usage": ["tool_name_1", ...],
#      "raw_data": <JSON string or dict used by downstream tools>,
#      "status": "completed" or "revised",
#      "revision_number": int,
#      "feedback_addressed": Optional[str]
#   }
#
# Feel free to extend with extra fields if useful.


# -----------------------------------------------------------------------------
# Small helpers shared by the runners (parsing, LLM calls, grounding checks)
# -----------------------------------------------------------------------------

def _safe_json_loads(data: Any) -> Dict[str, Any]:
    """Parse a tool's JSON string into a dict; return {} if it is not valid JSON."""
    if isinstance(data, dict):
        return data
    try:
        obj = json.loads(data)
    except (TypeError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {"value": obj}


def _clip(text: str, max_chars: int = 4000) -> str:
    """Keep long upstream context from bloating prompts."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]..."


def _call_llm(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> Tuple[str, Optional[str]]:
    """Invoke the LLM; return (content, error). Never raises, so one LLM failure can't crash the run."""
    try:
        msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        content = (msg.content or "").strip()
        return (content, None) if content else ("", "LLM returned empty content")
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def _feedback_block(feedback_context: Optional[str], revision_number: int,
                    previous_response: Optional[str]) -> str:
    """Prompt section that injects QC feedback (and the prior version) into a re-run."""
    if not feedback_context:
        return (
            "\n\nNo QC feedback has been provided for this version (initial analysis). "
            "Do NOT include a 'Response to QC Feedback' section.\n"
        )
    block = (
        f"\n\nQC FEEDBACK ON YOUR PREVIOUS VERSION (you are now writing revision {revision_number}):\n"
        f"{feedback_context}\n\n"
        "Revise your assessment to address EVERY point above. Include a 'Response to QC Feedback' "
        "section listing each issue and 'What I changed'. Do not invent numbers or evidence to satisfy feedback.\n"
    )
    if previous_response:
        block += f"\nYOUR PREVIOUS VERSION (for reference):\n{_clip(previous_response, 3000)}\n"
    return block


# Numbers that may appear in a narrative without coming from the tool JSON:
# EMA criteria (n>=3, ROR>=2, LCI>1, p<0.05), "95% CI", common p-value thresholds.
_ALLOWED_CONSTANTS = {0.001, 0.01, 0.05, 1.0, 1.5, 2.0, 3.0, 95.0}
_NUMBER_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.,])\d+(?:\.\d+)?(?![\w])")


def _numbers_in(text: str) -> List[str]:
    return _NUMBER_RE.findall(text or "")


def _to_float(token: str) -> Optional[float]:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _check_numeric_grounding(narrative: str, *sources: str) -> Dict[str, Any]:
    """
    Anti-hallucination check: every number in the narrative should come from the
    tool outputs (or be a known criterion constant / small list index).
    Returns a report; it does not block the pipeline.
    """
    allowed = set(_ALLOWED_CONSTANTS)
    for src in sources:
        for tok in _numbers_in(src):
            val = _to_float(tok)
            if val is not None:
                allowed.add(val)

    found = _numbers_in(narrative)
    ungrounded = []
    for tok in found:
        val = _to_float(tok)
        if val is None or val in allowed or (val.is_integer() and 0 <= val <= 10):
            continue
        ungrounded.append(tok)

    return {
        "numbers_checked": len(found),
        "ungrounded_numbers": sorted(set(ungrounded)),
        "grounded": not ungrounded,
    }


_EVIDENCE_RANK = {"none": 0, "limited": 1, "moderate": 2, "strong": 3}


def _evidence_overclaims(narrative: str, tool_strength: str) -> List[str]:
    """Evidence-strength words in the narrative that exceed the PubMed-based label from the tool."""
    rank = _EVIDENCE_RANK.get((tool_strength or "").lower())
    if rank is None:
        return []
    claims = re.findall(
        r"\b(strong|moderate|limited)\s+(?:literature\s+|clinical\s+|published\s+|pubmed\s+)?evidence",
        narrative or "",
        flags=re.IGNORECASE,
    )
    return sorted({c.lower() for c in claims if _EVIDENCE_RANK[c.lower()] > rank})


def _extract_classification(narrative: str) -> Optional[str]:
    """Read the clinical agent's PLAUSIBLE / IMPLAUSIBLE / UNCERTAIN call from its Classification section."""
    m = re.search(r"Classification(.*?)(?=\n#{1,6}\s|\Z)", narrative or "", flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    labels = re.findall(r"\b(IMPLAUSIBLE|PLAUSIBLE|UNCERTAIN)\b", m.group(1))
    distinct = list(dict.fromkeys(labels))
    return distinct[0] if len(distinct) == 1 else None


def _statistical_summary(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Key tool-returned values, used for logging and console output."""
    s = stats.get("statistics", {})
    ct = stats.get("contingency_table", {})
    return {
        "signal_detected": stats.get("signal_detection", {}).get("signal_detected"),
        "case_count": ct.get("drug_event"),
        "ror": s.get("ror"),
        "ror_95ci": [s.get("ror_95ci_lower"), s.get("ror_95ci_upper")],
        "prr": s.get("prr"),
        "p_value": s.get("p_value"),
        "chi2_test_valid": s.get("chi2_test_valid"),
        "data_quality_flags": stats.get("data_quality", {}).get("flags", []),
        "data_source": stats.get("data_source"),
    }


def _ema_criteria_check(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compare tool-returned values against EMA-style thresholds (no new statistics are computed).
    Given to the LLM so it reports MET / NOT MET per criterion instead of judging it itself;
    the tool's own criteria_met.ror_threshold uses ROR >= 1.5, not the ROR >= 2 decision rule.
    """
    s = stats.get("statistics", {})
    cases = stats.get("contingency_table", {}).get("drug_event")
    ror, lci, p = s.get("ror"), s.get("ror_95ci_lower"), s.get("p_value")

    def _ge(value: Any, threshold: float) -> Optional[bool]:
        if value == "infinite":
            return True
        return value >= threshold if isinstance(value, (int, float)) else None

    return [
        {"criterion": "case count >= 3", "tool_value": cases, "met": _ge(cases, 3)},
        {"criterion": "ROR >= 2", "tool_value": ror, "met": _ge(ror, 2.0)},
        {"criterion": "lower 95% CI bound > 1", "tool_value": lci,
         "met": lci > 1.0 if isinstance(lci, (int, float)) else None},
        {"criterion": "p < 0.05 (chi-square valid)", "tool_value": p,
         "met": (isinstance(p, (int, float)) and p < 0.05 and bool(s.get("chi2_test_valid")))},
    ]


def run_statistical_agent(
    llm: ChatOpenAI,
    drug_name: str,
    adverse_event: str,
    literature_evidence: Dict[str, Any],
    feedback_context: Optional[str] = None,
    revision_number: int = 0,
    previous_response: Optional[str] = None,
) -> Dict[str, Any]:
    """
    TODO 1A: Implement the statistical agent.

    Requirements (suggested steps):
      1. Use STATISTICAL_TOOLS["calculate_stats"].invoke(...) to get FAERS stats JSON.
      2. Use STATISTICAL_TOOLS["validate_stats"].invoke(...) to validate those stats.
      3. Build a prompt that includes:
           - Drug / adverse event
           - The stats JSON
           - The validation JSON
           - High-level literature_evidence (n_articles, strength, year_range)
           - Any QC feedback_context if this is a revision
      4. Call llm.invoke(...) with:
           - SystemMessage(STATISTICAL_SYSTEM_PROMPT)
           - HumanMessage(prompt)
      5. Return a dict describing what you did (see header comments).
    """
    tool_usage: List[str] = []

    # 1) FAERS statistics (live openFDA or local fallback, per configure_faers_mode)
    stats_json = STATISTICAL_TOOLS["calculate_stats"].invoke(
        {"drug_name": drug_name, "adverse_event": adverse_event}
    )
    tool_usage.append("calculate_drug_event_statistics")
    stats = _safe_json_loads(stats_json)

    # Graceful handling of API failures: the tool returns zero counts instead of raising,
    # so flag that explicitly for the LLM, QC and the audit log.
    data_access_warnings: List[str] = []
    if not stats:
        data_access_warnings.append("statistics_tool_returned_invalid_json")
    elif not stats.get("contingency_table", {}).get("total"):
        data_access_warnings.append("faers_no_data_or_api_unavailable")
    elif not stats.get("contingency_table", {}).get("total_drug_reports"):
        data_access_warnings.append("drug_not_found_in_faers_or_query_failed")

    # Live API down (empty database total) -> documented fallback to the local FAERS CSV
    if pharma_tools._use_live_api and data_access_warnings and os.path.exists(LOCAL_FAERS_CSV) \
            and "drug_not_found_in_faers_or_query_failed" not in data_access_warnings:
        print("  ⚠️ Live FAERS API unavailable; falling back to the local FAERS CSV snapshot...")
        try:
            configure_faers_mode(use_live_api=False)
            stats_json = STATISTICAL_TOOLS["calculate_stats"].invoke(
                {"drug_name": drug_name.upper(), "adverse_event": adverse_event.upper()}
            )
            tool_usage.append("calculate_drug_event_statistics[local_csv_fallback]")
            stats = _safe_json_loads(stats_json)
            data_access_warnings.append("live_api_unavailable_used_local_csv_fallback")
        except Exception as e:
            data_access_warnings.append(f"local_csv_fallback_failed: {type(e).__name__}: {e}")
        finally:
            pharma_tools.set_use_live_api(True)

    # 2) Validate the statistics
    validation_json = STATISTICAL_TOOLS["validate_stats"].invoke(
        {"statistical_results_json": stats_json}
    )
    tool_usage.append("validate_statistical_results")
    validation = _safe_json_loads(validation_json)

    criteria_check = _ema_criteria_check(stats)
    literature_context = {
        "n_articles": literature_evidence.get("n_articles"),
        "evidence_strength": literature_evidence.get("evidence_strength"),
        "year_range": literature_evidence.get("year_range"),
    }

    # 3) Prompt with tool outputs (the only allowed source of numbers)
    prompt = (
        f"Drug: {drug_name}\nAdverse event: {adverse_event}\n\n"
        f"FAERS STATISTICS (calculate_drug_event_statistics JSON):\n{stats_json}\n\n"
        f"VALIDATION REPORT (validate_statistical_results JSON):\n{validation_json}\n\n"
        f"LITERATURE CONTEXT (PubMed summary, context only):\n{json.dumps(literature_context)}\n\n"
        f"EMA CRITERIA CHECK (tool values compared with the thresholds; report each one exactly as "
        f"MET / NOT MET in your Signal Decision section, and explain the decision from it — "
        f"a signal requires case count, ROR and at least one significance criterion to be met):\n"
        f"{json.dumps(criteria_check, indent=2)}\n"
    )
    if data_access_warnings:
        prompt += f"\nDATA ACCESS WARNINGS: {data_access_warnings}\n"
    prompt += _feedback_block(feedback_context, revision_number, previous_response)
    prompt += "\nWrite your statistical signal assessment now, following your output format."

    # 4) LLM interpretation
    narrative, llm_error = _call_llm(llm, STATISTICAL_SYSTEM_PROMPT, prompt)
    if llm_error:
        summary = _statistical_summary(stats)
        narrative = (
            f"[LLM unavailable: {llm_error}] Tool-only summary: signal_detected="
            f"{summary['signal_detected']}, ROR={summary['ror']} (95% CI {summary['ror_95ci'][0]}-"
            f"{summary['ror_95ci'][1]}), PRR={summary['prr']}, cases={summary['case_count']}, "
            f"p={summary['p_value']}; validation={validation.get('validation_status')}."
        )

    # 5) Result for QC / the loop
    return {
        "agent_response": narrative,
        "tool_usage": tool_usage,
        "raw_data": stats_json,                 # FAERS stats JSON string (used by QC + regulatory)
        "validation": validation,
        "statistics_summary": _statistical_summary(stats),
        "ema_criteria_check": criteria_check,
        "data_access_warnings": data_access_warnings,
        "numeric_grounding": _check_numeric_grounding(
            narrative, stats_json, validation_json, json.dumps(literature_context),
            json.dumps(criteria_check), feedback_context or "",
        ),
        "llm_error": llm_error,
        "status": "revised" if revision_number > 0 else "completed",
        "revision_number": revision_number,
        "feedback_addressed": feedback_context,
    }


def run_clinical_agent(
    llm: ChatOpenAI,
    drug_name: str,
    adverse_event: str,
    literature_evidence: Dict[str, Any],
    statistical_context: str,
    feedback_context: Optional[str] = None,
    revision_number: int = 0,
    previous_response: Optional[str] = None,
) -> Dict[str, Any]:
    """
    TODO 1B: Implement the clinical agent.

    Requirements (suggested steps):
      1. Call CLINICAL_TOOLS["search_literature"].invoke(...) → JSON string.
      2. Call CLINICAL_TOOLS["assess_plausibility"].invoke(...) → JSON string.
         (You can pass literature_evidence as a JSON string in literature_context.)
      3. Optionally merge those JSONs into ONE combined dict for QC and regulatory.
      4. Build a prompt that includes:
           - Drug / adverse event
           - The statistical_context (truncated if huge)
           - The tool outputs (literature + plausibility)
           - Any QC feedback_context if this is a revision
      5. Call llm.invoke(...) with CLINICAL_SYSTEM_PROMPT + your human prompt.
      6. Return a dict describing what you did (see header comments).
         Make sure QC & regulatory can access a combined JSON string, e.g. under
         something like raw_data["combined_for_qc"].
    """
    tool_usage: List[str] = []

    # 1) PubMed literature search
    literature_json = CLINICAL_TOOLS["search_literature"].invoke(
        {"drug_name": drug_name, "adverse_event": adverse_event}
    )
    tool_usage.append("search_clinical_literature")
    literature = _safe_json_loads(literature_json)

    # 2) Biological plausibility, enriched with the shared PubMed evidence
    plausibility_json = CLINICAL_TOOLS["assess_plausibility"].invoke(
        {
            "drug_name": drug_name,
            "adverse_event": adverse_event,
            "literature_context": json.dumps(literature_evidence),
        }
    )
    tool_usage.append("assess_biological_plausibility")
    plausibility = _safe_json_loads(plausibility_json)

    search_strength = literature.get("evidence_summary", {}).get("evidence_strength", "unknown")
    publication_count = literature.get("evidence_summary", {}).get("publication_count", 0)

    # The shared PubMed fetch (used by QC as ground truth) and this search can return slightly
    # different counts. Use the more conservative label so the narrative never over-claims.
    shared_strength = literature_evidence.get("evidence_strength", "unknown")
    tool_strength = search_strength
    if shared_strength in _EVIDENCE_RANK and (
        search_strength not in _EVIDENCE_RANK
        or _EVIDENCE_RANK[shared_strength] < _EVIDENCE_RANK[search_strength]
    ):
        tool_strength = shared_strength

    # 3) Prompt with statistical context + tool outputs
    prompt = (
        f"Drug: {drug_name}\nAdverse event: {adverse_event}\n\n"
        f"STATISTICAL CONTEXT (from the Statistical Agent):\n{_clip(statistical_context)}\n\n"
        f"LITERATURE (search_clinical_literature JSON):\n{literature_json}\n\n"
        f"BIOLOGICAL PLAUSIBILITY (assess_biological_plausibility JSON):\n{plausibility_json}\n\n"
        f"EVIDENCE STRENGTH CONSTRAINT: this literature search returned {publication_count} articles "
        f"('{search_strength}'); the shared PubMed evidence used for QC returned "
        f"{literature_evidence.get('n_articles', 0)} articles ('{shared_strength}'). Report the evidence "
        f"strength as '{tool_strength}' (the more conservative label) and do not describe it as stronger.\n"
    )
    prompt += _feedback_block(feedback_context, revision_number, previous_response)
    prompt += "\nWrite your clinical assessment now, following your output format."

    # 4) LLM synthesis
    narrative, llm_error = _call_llm(llm, CLINICAL_SYSTEM_PROMPT, prompt)
    if llm_error:
        narrative = (
            f"[LLM unavailable: {llm_error}] Tool-only summary: overall assessment "
            f"{plausibility.get('overall_assessment')}, evidence strength '{tool_strength}' "
            f"({publication_count} PubMed articles)."
        )

    # 5) Combined clinical JSON for QC + regulatory. QC reads:
    #    mechanism_assessment, evidence_summary, narrative_response and _revision_number.
    combined = {**literature, **plausibility}
    combined["narrative_response"] = narrative
    combined["_revision_number"] = revision_number

    llm_classification = _extract_classification(narrative)
    return {
        "agent_response": narrative,
        "tool_usage": tool_usage,
        "raw_data": {
            "literature_json": literature_json,
            "plausibility_json": plausibility_json,
            "combined_for_qc": json.dumps(combined),
        },
        "classification": llm_classification or plausibility.get("overall_assessment"),
        "tool_assessment": plausibility.get("overall_assessment"),
        "classification_consistent_with_tool": (
            llm_classification is None or llm_classification == plausibility.get("overall_assessment")
        ),
        "evidence_check": {
            "publication_count": publication_count,
            "shared_pubmed_articles": literature_evidence.get("n_articles", 0),
            "search_evidence_strength": search_strength,
            "shared_evidence_strength": shared_strength,
            "tool_evidence_strength": tool_strength,
            "overclaimed_strengths": _evidence_overclaims(narrative, tool_strength),
        },
        "llm_error": llm_error,
        "status": "revised" if revision_number > 0 else "completed",
        "revision_number": revision_number,
        "feedback_addressed": feedback_context,
    }


def run_regulatory_agent(
    drug_name: str,
    adverse_event: str,
    statistical_data_json: str,
    clinical_combined_json: str,
    literature_evidence: Dict[str, Any],
    feedback_loop_summary: Optional[Dict[str, Any]] = None,
    revision_number: int = 0,
    llm: Optional[ChatOpenAI] = None,
    feedback_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    TODO 1C: Implement the regulatory 'agent'.

    Extra (optional): when QC feedback targets this agent and an llm is given, the LLM writes a
    documented "Response to QC Feedback" (REGULATORY_SYSTEM_PROMPT) next to the tool report.
    The report itself is never edited, so it always reflects the latest inputs.

    This one can be purely tool-based (no extra LLM needed).

    Requirements:
      1. Build a dict of arguments expected by REGULATORY_TOOL (see pharma_tools.py
         generate_regulatory_report docstring). Roughly:
            {
              "drug_name": ...,
              "adverse_event": ...,
              "statistical_data": <JSON string>,
              "clinical_assessment": <JSON string>,
              "literature_evidence": <JSON string>,
              "feedback_loop_summary": <JSON string or None>,
            }
      2. Call REGULATORY_TOOL.invoke(args) → JSON string report.
      3. Return a dict describing what you did (see header comments).
    """
    args = {
        "drug_name": drug_name,
        "adverse_event": adverse_event,
        "statistical_data": _as_json_str(statistical_data_json),
        "clinical_assessment": _as_json_str(clinical_combined_json),
        "literature_evidence": json.dumps(literature_evidence),
        "feedback_loop_summary": json.dumps(feedback_loop_summary) if feedback_loop_summary else None,
    }
    report_json = REGULATORY_TOOL.invoke(args)
    report = _safe_json_loads(report_json)

    feedback_response, llm_error = None, None
    if feedback_context and llm is not None:
        clinical = _safe_json_loads(clinical_combined_json)
        prompt = (
            f"Drug: {drug_name}\nAdverse event: {adverse_event}\n\n"
            f"REGULATORY REPORT (generate_regulatory_report JSON):\n{report_json}\n\n"
            f"KEY STATISTICS: {json.dumps(_statistical_summary(_safe_json_loads(statistical_data_json)))}\n"
            f"CLINICAL: overall_assessment={clinical.get('overall_assessment')}, "
            f"mechanism={json.dumps(clinical.get('mechanism_assessment', {}))}\n"
            f"PUBMED: n_articles={literature_evidence.get('n_articles')}, "
            f"evidence_strength={literature_evidence.get('evidence_strength')}\n\n"
            f"QC FEEDBACK (you are writing revision {revision_number}):\n{feedback_context}\n"
        )
        feedback_response, llm_error = _call_llm(llm, REGULATORY_SYSTEM_PROMPT, prompt)

    return {
        "agent_response": report_json,          # the structured report is the agent's output
        "tool_usage": ["generate_regulatory_report"],
        "raw_data": report_json,
        "risk_level": report.get("executive_summary", {}).get("risk_level"),
        "signal_status": report.get("executive_summary", {}).get("signal_status"),
        "report_error": report.get("error"),
        "feedback_response": feedback_response,  # documented response to QC (re-runs only)
        "llm_error": llm_error,
        "status": "revised" if revision_number > 0 else "completed",
        "revision_number": revision_number,
        "feedback_addressed": feedback_context,
    }


# =============================================================================
# 4. QUALITY CONTROL & FEEDBACK HELPERS  (TODO 2)
# =============================================================================

def run_quality_control(
    statistical_data_json: str,
    clinical_combined_json: str,
    regulatory_report_json: str,
    literature_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    TODO 2A: Implement the QC runner.

    Requirements:
      1. Prepare arguments expected by QC_TOOL (quality_review_analysis).
         Look at pharma_tools.quality_review_analysis docstring. It expects 4 strings:
            - statistical_findings
            - clinical_findings
            - regulatory_report
            - literature_evidence
      2. Call QC_TOOL.invoke(args) → JSON string.
      3. Parse the JSON string to a dict.
      4. Return the parsed dict, but it is helpful to ALSO keep the raw JSON
         somewhere inside it, e.g. qc["raw_tool_output"] = <string>.

    The QC tool will typically return keys like:
        "quality_scores" (with "overall_score"),
        "target_agents",
        "issues",
        "approval_status",
    but confirm by reading pharma_tools.py.
    """
    args = {
        "statistical_findings": _as_json_str(statistical_data_json),
        "clinical_findings": _as_json_str(clinical_combined_json),
        "regulatory_report": _as_json_str(regulatory_report_json),
        "literature_evidence": json.dumps(literature_evidence),
    }
    qc_json = QC_TOOL.invoke(args)
    qc = _safe_json_loads(qc_json)
    if not qc:
        qc = {
            "error": "QC tool returned invalid JSON",
            "quality_scores": {"overall_score": 0},
            "approval_status": "revision_required",
            "target_agents": [],
            "issues": [],
        }
    qc.setdefault("quality_scores", {}).setdefault("overall_score", 0)
    qc.setdefault("target_agents", [])
    qc.setdefault("issues", [])
    qc.setdefault("approval_status", "revision_required")
    qc["raw_tool_output"] = qc_json
    return qc


def get_feedback_for_agent(
    agent_name: str, issues: List[Dict[str, Any]]
) -> Optional[str]:
    """
    TODO 2B: Extract QC feedback relevant to a specific agent.

    Requirements:
      1. Filter the issues list to those where issue["responsible_agent"] == agent_name.
      2. If none, return None.
      3. Otherwise, build a concise feedback string summarizing:
           - type
           - severity
           - description
           - specific_action
           - evidence_needed
         for each relevant issue.
      4. Return that feedback string.
    """
    relevant = [i for i in issues if i.get("responsible_agent") == agent_name]
    if not relevant:
        return None

    lines = [f"QC identified {len(relevant)} issue(s) for the {agent_name} agent:"]
    for n, issue in enumerate(relevant, start=1):
        lines.append(
            f"{n}. [{str(issue.get('severity', 'unknown')).upper()}] {issue.get('type', 'issue')}: "
            f"{issue.get('description', '')}"
        )
        if issue.get("specific_action"):
            lines.append(f"   Required action: {issue['specific_action']}")
        if issue.get("evidence_needed"):
            lines.append(f"   Evidence needed: {', '.join(map(str, issue['evidence_needed']))}")
    return "\n".join(lines)


# =============================================================================
# 5. FEEDBACK-DRIVEN RE-RUNNING  (TODO 3)
# =============================================================================

# Which key in current_results holds each agent's latest result.
AGENT_RESULT_KEYS = {
    "statistical": "statistical_analysis",
    "clinical": "clinical_assessment",
    "regulatory": "regulatory_report",
}


def _as_json_str(data: Any) -> str:
    """Tools take JSON strings; accept either a JSON string or a dict."""
    if isinstance(data, str):
        return data
    return json.dumps(data if data is not None else {})


def _statistical_json(stat_result: Dict[str, Any]) -> str:
    """Raw FAERS stats JSON (calculate_drug_event_statistics output) from a statistical result."""
    raw = stat_result.get("raw_data")
    if isinstance(raw, dict) and "statistics_json" in raw:
        raw = raw["statistics_json"]
    return _as_json_str(raw)


def _clinical_combined_json(clinical_result: Dict[str, Any]) -> str:
    """Combined clinical JSON (literature + plausibility + narrative) used by QC and regulatory."""
    raw = clinical_result.get("raw_data")
    if isinstance(raw, dict) and "combined_for_qc" in raw:
        raw = raw["combined_for_qc"]
    return _as_json_str(raw)


def rerun_agent_with_feedback(
    llm: ChatOpenAI,
    agent_name: str,
    feedback: str,
    drug_name: str,
    adverse_event: str,
    literature_evidence: Dict[str, Any],
    current_results: Dict[str, Any],
    audit: Optional[AuditLogger] = None,
    feedback_loop_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    TODO 3: Re-run a specific agent, incorporating QC feedback.

    Inputs:
      - agent_name: "statistical", "clinical", or "regulatory"
      - feedback:   text from get_feedback_for_agent(...)
      - current_results: a dict which should contain:
            "statistical_analysis": {...},
            "clinical_assessment": {...},
            "regulatory_report": {...}

    Implement:
      1. Look up the previous result for this agent in current_results:
           - "statistical" → current_results["statistical_analysis"]
           - "clinical"    → current_results["clinical_assessment"]
           - "regulatory"  → current_results["regulatory_report"]
         Read its revision_number (default 0 if missing) and set new_revision = old + 1.

      2. For each agent_name:
           - "statistical":
                 call run_statistical_agent(..., feedback_context=feedback,
                                            revision_number=new_revision)
           - "clinical":
                 call run_clinical_agent(..., feedback_context=feedback,
                                         revision_number=new_revision),
                 BUT remember to pass the latest statistical agent_response as
                 statistical_context.
           - "regulatory":
                 call run_regulatory_agent(..., revision_number=new_revision),
                 using the latest statistical raw_data JSON and clinical combined JSON.

      3. Return the revised result dict.

    Extra (optional) arguments:
      - audit: if given, each re-run is recorded with audit.log_agent_execution(...)
        (revision_number > 0 plus a feedback snippet). Failures are logged too.
      - feedback_loop_summary: loop metadata passed to the regulatory report tool.

    If the runner fails (e.g. API/LLM error), the failure is logged and the previous
    result is returned with status "revision_failed", so the loop does not crash.
    """
    if agent_name not in AGENT_RESULT_KEYS:
        raise ValueError(
            f"Unknown agent '{agent_name}'. Expected one of {sorted(AGENT_RESULT_KEYS)}."
        )

    # 1) Latest result for this agent -> next revision number
    previous = current_results.get(AGENT_RESULT_KEYS[agent_name]) or {}
    new_revision = int(previous.get("revision_number") or 0) + 1

    print(f"  🔁 Re-running {agent_name} agent (revision {new_revision - 1} -> {new_revision}) "
          f"with QC feedback:")
    for line in (feedback or "").splitlines()[:4]:
        print(f"     | {line}")

    # 2) Call the matching runner with the feedback and the latest upstream results
    try:
        if agent_name == "statistical":
            revised = run_statistical_agent(
                llm,
                drug_name,
                adverse_event,
                literature_evidence,
                feedback_context=feedback,
                revision_number=new_revision,
                previous_response=previous.get("agent_response"),
            )

        elif agent_name == "clinical":
            statistical_context = (
                current_results.get("statistical_analysis", {}).get("agent_response", "")
            )
            revised = run_clinical_agent(
                llm,
                drug_name,
                adverse_event,
                literature_evidence,
                statistical_context=statistical_context,
                feedback_context=feedback,
                revision_number=new_revision,
                previous_response=previous.get("agent_response"),
            )

        else:  # "regulatory" (tool-based; the report is rebuilt from the latest inputs)
            revised = run_regulatory_agent(
                drug_name,
                adverse_event,
                statistical_data_json=_statistical_json(
                    current_results.get("statistical_analysis", {})
                ),
                clinical_combined_json=_clinical_combined_json(
                    current_results.get("clinical_assessment", {})
                ),
                literature_evidence=literature_evidence,
                feedback_loop_summary=feedback_loop_summary,
                revision_number=new_revision,
                llm=llm,
                feedback_context=feedback,
            )

    except Exception as e:
        print(f"  ⚠️ Re-run of {agent_name} agent failed: {type(e).__name__}: {e}")
        if audit is not None:
            audit.log_error(
                "agent_rerun_failed",
                f"{type(e).__name__}: {e}",
                {"agent": agent_name, "attempted_revision": new_revision},
            )
            audit.log_agent_execution(
                agent_name=agent_name,
                tool_calls=[],
                success=False,
                revision_number=new_revision,
                feedback_context=feedback,
            )
        failed = dict(previous)
        failed["status"] = "revision_failed"
        failed["revision_error"] = f"{type(e).__name__}: {e}"
        return failed

    # Make sure the revision metadata is always present on the returned result
    revised["revision_number"] = new_revision
    revised["status"] = "revised"
    if not revised.get("feedback_addressed"):
        revised["feedback_addressed"] = feedback

    # 3) Audit trail for the re-run
    if audit is not None:
        audit.log_agent_execution(
            agent_name=agent_name,
            tool_calls=revised.get("tool_usage", []),
            success=True,
            revision_number=new_revision,
            feedback_context=feedback,
        )

    return revised


# =============================================================================
# 6. MAIN ORCHESTRATION: FEEDBACK LOOP  (TODO 4)
# =============================================================================

def _log_agent_outcome(audit: AuditLogger, agent_name: str, result: Dict[str, Any]) -> None:
    """Record each agent's decision (not just that it ran) so the audit trail explains the outcome."""
    rev = result.get("revision_number", 0)
    if agent_name == "statistical":
        audit.log_event("statistical_signal_assessment", {
            "revision_number": rev,
            **result.get("statistics_summary", {}),
            "validation_status": result.get("validation", {}).get("validation_status"),
            "data_access_warnings": result.get("data_access_warnings", []),
            "ungrounded_numbers": result.get("numeric_grounding", {}).get("ungrounded_numbers", []),
        }, level="warning" if result.get("data_access_warnings") or result.get("llm_error") else "info")
    elif agent_name == "clinical":
        audit.log_event("clinical_plausibility_assessment", {
            "revision_number": rev,
            "classification": result.get("classification"),
            "tool_assessment": result.get("tool_assessment"),
            "classification_consistent_with_tool": result.get("classification_consistent_with_tool"),
            **result.get("evidence_check", {}),
        }, level="warning" if result.get("evidence_check", {}).get("overclaimed_strengths")
                                or result.get("llm_error") else "info")
    elif agent_name == "regulatory":
        report = _safe_json_loads(result.get("raw_data"))
        audit.log_event("regulatory_report_generated", {
            "revision_number": rev,
            "risk_level": result.get("risk_level"),
            "signal_status": result.get("signal_status"),
            "immediate_actions": report.get("regulatory_actions", {}).get("immediate_actions", []),
            "report_error": result.get("report_error"),
            "feedback_response_snippet": (result.get("feedback_response") or "")[:300] or None,
        }, level="error" if result.get("report_error") else "info")


def _action_record(agent_name: str, action: str, result: Dict[str, Any],
                   feedback: Optional[str] = None) -> Dict[str, Any]:
    """One entry in an iteration's list of actions (for metadata + audit log)."""
    return {
        "agent": agent_name,
        "action": action,
        "revision_number": result.get("revision_number", 0),
        "tool_usage": result.get("tool_usage", []),
        "feedback_snippet": feedback[:200] if feedback else None,
    }


def analyze_with_feedback_loop(
    drug_name: str,
    adverse_event: str,
    max_iterations: int = 3,
    quality_threshold: float = 90.0,
    llm: Optional[ChatOpenAI] = None,
) -> Dict[str, Any]:
    """
    TODO 4: Implement the full multi-agent pipeline WITH a feedback loop.

    Overall structure (high level):

      1. Setup:
         - Create llm if not provided.
         - Create AuditLogger and call audit.start_run(...).

      2. Step 0: Fetch literature evidence ONCE using fetch_literature_evidence().
         - Save this in results so all agents share the same view of PubMed.

      3. Step 1: Initial pass (iteration = 1):
         - Run statistical agent once.
         - Run clinical agent once (with statistical agent_response as context).
         - Run regulatory agent once.
         - Log each agent execution with audit.log_agent_execution(...).

      4. Step 2: First QC pass:
         - Call run_quality_control(...) using:
              - statistical_data_json  (stats agent raw_data)
              - clinical_combined_json (your combined clinical JSON)
              - regulatory_report_json (regulatory raw_data)
              - literature_evidence
         - Extract:
              quality_score  = qc["quality_scores"]["overall_score"]
              target_agents  = qc["target_agents"]
         - Log QC with audit.log_quality_review(...).

      5. Step 3: Feedback loop:
         - Maintain:
              iteration = 1  (already done one pass)
              feedback_loop_metadata = {
                  "iterations_used": ...,
                  "final_quality_score": ...,
                  "converged": bool,
                  "analysis_history": [ ... ],
              }
         - While:
              iteration < max_iterations AND quality_score < quality_threshold:
             a) If target_agents is empty:
                    mark converged=True and break.
             b) For each agent_name in target_agents:
                    - Construct feedback string with get_feedback_for_agent(...)
                    - Call rerun_agent_with_feedback(...)
                    - Update current_results[...] for that agent.
                    - Call audit.log_agent_execution(...) for each revision
                      (minimal audit logging is a realistic regulatory requirement).
             c) If stats OR clinical changed, re-run regulatory agent so it sees
                updated data, and update current_results["regulatory_report"].
             d) Re-run QC with updated results.
             e) Update quality_score, target_agents.
             f) Append an iteration record to feedback_loop_metadata["analysis_history"].
             g) Log QC again with audit.log_quality_review(...).

         - After loop ends, set:
             feedback_loop_metadata["iterations_used"] = iteration
             feedback_loop_metadata["final_quality_score"] = quality_score
             feedback_loop_metadata["converged"] = (quality_score >= quality_threshold
                                                    or target_agents == []).

      6. Construct and return a final results dict with keys like:
           "drug_name", "adverse_event", "timestamp",
           "literature_evidence", "statistical_analysis",
           "clinical_assessment", "regulatory_report",
           "quality_review", "feedback_loop_metadata",
           "audit_log_path".
    """
    # ---------------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------------
    if llm is None:
        llm = setup_environment()

    audit = AuditLogger()
    audit.start_run(drug_name, adverse_event, metadata={
        "max_iterations": max_iterations,
        "quality_threshold": quality_threshold,
        "faers_live_api": pharma_tools._use_live_api,
        "model": getattr(llm, "model_name", None),
    })

    print("=" * 72)
    print(f"PHARMACOVIGILANCE SIGNAL DETECTOR: {drug_name} + {adverse_event}")
    print(f"max_iterations={max_iterations}, quality_threshold={quality_threshold}")
    print("=" * 72)

    current_results: Dict[str, Any] = {}
    feedback_loop_metadata: Dict[str, Any] = {
        "iterations_used": 0,
        "final_quality_score": None,
        "converged": False,
        "stop_reason": None,
        "max_iterations": max_iterations,
        "quality_threshold": quality_threshold,
        "analysis_history": [],
    }

    try:
        # ---------------------------------------------------------------
        # 2. Step 0: literature evidence, fetched once and shared by all agents
        # ---------------------------------------------------------------
        print("\n📚 Fetching PubMed literature evidence...")
        literature_evidence = fetch_literature_evidence(drug_name, adverse_event)
        audit.log_literature_fetch(literature_evidence)
        if not literature_evidence.get("available"):
            audit.log_event("literature_unavailable",
                            {"reason": literature_evidence.get("reason")}, level="warning")
        print(f"  {literature_evidence.get('n_articles', 0)} articles, "
              f"evidence strength: {literature_evidence.get('evidence_strength')}")

        def qc_pass(iteration: int, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Run QC on the current results, log it, and record the iteration."""
            qc = run_quality_control(
                _statistical_json(current_results["statistical_analysis"]),
                _clinical_combined_json(current_results["clinical_assessment"]),
                _as_json_str(current_results["regulatory_report"]["raw_data"]),
                literature_evidence,
            )
            scores = qc["quality_scores"]
            issues = qc["issues"]
            audit.log_quality_review(scores, issues, qc["approval_status"], qc["target_agents"])
            audit.log_iteration(iteration, scores["overall_score"], issues, actions)
            feedback_loop_metadata["iterations_used"] = iteration

            unknown = [a for a in qc["target_agents"] if a not in AGENT_RESULT_KEYS]
            if unknown:
                audit.log_event("unroutable_target_agents", {"agents": unknown}, level="warning")

            feedback_loop_metadata["analysis_history"].append({
                "iteration": iteration,
                "quality_score": scores["overall_score"],
                "quality_scores": scores,
                "approval_status": qc["approval_status"],
                "target_agents": qc["target_agents"],
                "issues": [
                    {k: i.get(k) for k in ("type", "severity", "responsible_agent", "description",
                                           "issue_status", "applied_penalty")}
                    for i in issues
                ],
                "actions": actions,
            })

            print(f"\n🔍 QC (iteration {iteration}): overall={scores['overall_score']} "
                  f"[{qc['approval_status']}] targets={qc['target_agents'] or 'none'}")
            for i in issues:
                print(f"   - [{i.get('severity')}] {i.get('type')} -> {i.get('responsible_agent')}")
            return qc

        # ---------------------------------------------------------------
        # 3. Step 1: initial pass (iteration 1)
        # ---------------------------------------------------------------
        iteration = 1
        actions: List[Dict[str, Any]] = []
        print(f"\n{'─' * 72}\nITERATION 1: initial pass\n{'─' * 72}")

        print("📊 Statistical agent...")
        stat = run_statistical_agent(llm, drug_name, adverse_event, literature_evidence)
        current_results["statistical_analysis"] = stat
        audit.log_agent_execution("statistical", stat["tool_usage"],
                                  success=not stat.get("llm_error"), revision_number=0)
        _log_agent_outcome(audit, "statistical", stat)
        actions.append(_action_record("statistical", "initial_run", stat))
        s = stat["statistics_summary"]
        print(f"  signal_detected={s['signal_detected']}, ROR={s['ror']} "
              f"(95% CI {s['ror_95ci'][0]}-{s['ror_95ci'][1]}), cases={s['case_count']}")

        print("🩺 Clinical agent...")
        clin = run_clinical_agent(llm, drug_name, adverse_event, literature_evidence,
                                  statistical_context=stat["agent_response"])
        current_results["clinical_assessment"] = clin
        audit.log_agent_execution("clinical", clin["tool_usage"],
                                  success=not clin.get("llm_error"), revision_number=0)
        _log_agent_outcome(audit, "clinical", clin)
        actions.append(_action_record("clinical", "initial_run", clin))
        print(f"  classification={clin['classification']} "
              f"(tool: {clin['tool_assessment']}), evidence={clin['evidence_check']['tool_evidence_strength']}")

        print("📋 Regulatory agent...")
        reg = run_regulatory_agent(drug_name, adverse_event, _statistical_json(stat),
                                   _clinical_combined_json(clin), literature_evidence)
        current_results["regulatory_report"] = reg
        audit.log_agent_execution("regulatory", reg["tool_usage"],
                                  success=not reg.get("report_error"), revision_number=0)
        _log_agent_outcome(audit, "regulatory", reg)
        actions.append(_action_record("regulatory", "initial_run", reg))
        print(f"  risk_level={reg['risk_level']}, signal_status={reg['signal_status']}")

        # ---------------------------------------------------------------
        # 4. Step 2: first QC pass
        # ---------------------------------------------------------------
        qc = qc_pass(iteration, actions)
        quality_score = float(qc["quality_scores"]["overall_score"])
        routable = [a for a in AGENT_RESULT_KEYS if a in qc["target_agents"]]

        # ---------------------------------------------------------------
        # 5. Step 3: feedback loop
        # ---------------------------------------------------------------
        while iteration < max_iterations and quality_score < quality_threshold:
            if not routable:
                feedback_loop_metadata["stop_reason"] = "no_target_agents"
                break

            iteration += 1
            actions = []
            changed: List[str] = []
            loop_summary = {"iterations_used": iteration, "final_quality_score": quality_score,
                            "converged": False}
            print(f"\n{'─' * 72}\nITERATION {iteration}: revising {routable}\n{'─' * 72}")

            # a/b) Re-run only the QC-targeted upstream agents (statistical before clinical,
            #      so a revised clinical agent sees the latest statistical response)
            for agent_name in routable:
                if agent_name == "regulatory":
                    continue  # handled below, after upstream agents
                feedback = get_feedback_for_agent(agent_name, qc["issues"])
                if not feedback:
                    actions.append(_action_record(agent_name, "skipped_no_feedback",
                                                  current_results[AGENT_RESULT_KEYS[agent_name]]))
                    continue
                revised = rerun_agent_with_feedback(
                    llm, agent_name, feedback, drug_name, adverse_event, literature_evidence,
                    current_results, audit=audit, feedback_loop_summary=loop_summary,
                )
                current_results[AGENT_RESULT_KEYS[agent_name]] = revised
                actions.append(_action_record(agent_name, revised["status"], revised, feedback))
                if revised["status"] == "revised":
                    changed.append(agent_name)
                    _log_agent_outcome(audit, agent_name, revised)

            # c) Regulatory: re-run with feedback if targeted, otherwise regenerate it
            #    whenever statistical/clinical changed so the report reflects the latest inputs.
            reg_feedback = (get_feedback_for_agent("regulatory", qc["issues"])
                            if "regulatory" in routable else None)
            if reg_feedback:
                revised = rerun_agent_with_feedback(
                    llm, "regulatory", reg_feedback, drug_name, adverse_event, literature_evidence,
                    current_results, audit=audit, feedback_loop_summary=loop_summary,
                )
                reg_action = revised["status"]
            elif changed:
                previous_reg = current_results["regulatory_report"]
                new_revision = int(previous_reg.get("revision_number") or 0) + 1
                note = f"Regenerated after upstream revision(s): {', '.join(changed)}"
                print(f"  📋 {note}")
                revised = run_regulatory_agent(
                    drug_name, adverse_event,
                    _statistical_json(current_results["statistical_analysis"]),
                    _clinical_combined_json(current_results["clinical_assessment"]),
                    literature_evidence, feedback_loop_summary=loop_summary,
                    revision_number=new_revision,
                )
                revised["feedback_addressed"] = note
                audit.log_agent_execution("regulatory", revised["tool_usage"],
                                          success=not revised.get("report_error"),
                                          revision_number=new_revision, feedback_context=note)
                reg_action, reg_feedback = "regenerated", note
            else:
                revised = None
            if revised is not None:
                current_results["regulatory_report"] = revised
                actions.append(_action_record("regulatory", reg_action, revised, reg_feedback))
                _log_agent_outcome(audit, "regulatory", revised)

            # d/e/f/g) Re-run QC on the updated results and route again
            qc = qc_pass(iteration, actions)
            quality_score = float(qc["quality_scores"]["overall_score"])
            routable = [a for a in AGENT_RESULT_KEYS if a in qc["target_agents"]]

        # ---------------------------------------------------------------
        # Loop outcome
        # ---------------------------------------------------------------
        if quality_score >= quality_threshold:
            feedback_loop_metadata["stop_reason"] = "quality_threshold_met"
        elif feedback_loop_metadata["stop_reason"] is None:
            feedback_loop_metadata["stop_reason"] = (
                "no_target_agents" if not routable else "max_iterations_reached"
            )
        feedback_loop_metadata["iterations_used"] = iteration
        feedback_loop_metadata["final_quality_score"] = quality_score
        feedback_loop_metadata["converged"] = quality_score >= quality_threshold or not routable

        # Final report carries the feedback-loop summary in its quality_assurance section
        final_reg = run_regulatory_agent(
            drug_name, adverse_event,
            _statistical_json(current_results["statistical_analysis"]),
            _clinical_combined_json(current_results["clinical_assessment"]),
            literature_evidence,
            feedback_loop_summary={k: feedback_loop_metadata[k]
                                   for k in ("iterations_used", "final_quality_score", "converged")},
            revision_number=current_results["regulatory_report"].get("revision_number", 0),
        )
        final_reg["status"] = current_results["regulatory_report"].get("status", "completed")
        final_reg["feedback_addressed"] = current_results["regulatory_report"].get("feedback_addressed")
        final_reg["feedback_response"] = current_results["regulatory_report"].get("feedback_response")
        current_results["regulatory_report"] = final_reg
        audit.log_event("final_report_generated", {
            "risk_level": final_reg.get("risk_level"),
            "signal_status": final_reg.get("signal_status"),
            "revision_number": final_reg.get("revision_number"),
        })
        audit.log_event("feedback_loop_completed", {
            k: v for k, v in feedback_loop_metadata.items() if k != "analysis_history"
        })

        print(f"\n✅ Loop finished: {feedback_loop_metadata['stop_reason']} after {iteration} "
              f"iteration(s); final quality score {quality_score} "
              f"(converged={feedback_loop_metadata['converged']})")

        # ---------------------------------------------------------------
        # 6. Final results + audit end_run
        # ---------------------------------------------------------------
        results: Dict[str, Any] = {
            "drug_name": drug_name,
            "adverse_event": adverse_event,
            "timestamp": datetime.now().isoformat(),
            "literature_evidence": literature_evidence,
            "statistical_analysis": current_results["statistical_analysis"],
            "clinical_assessment": current_results["clinical_assessment"],
            "regulatory_report": current_results["regulatory_report"],
            "quality_review": qc,
            "feedback_loop_metadata": feedback_loop_metadata,
        }
        results["audit_log_path"] = audit.end_run(final_results={
            "statistical_summary": current_results["statistical_analysis"].get("statistics_summary"),
            "clinical_classification": current_results["clinical_assessment"].get("classification"),
            "risk_level": final_reg.get("risk_level"),
            "approval_status": qc.get("approval_status"),
            "quality_scores": qc.get("quality_scores"),
            "feedback_loop": {k: v for k, v in feedback_loop_metadata.items()
                              if k != "analysis_history"},
        }, status="completed")
        return results

    except Exception as e:
        print(f"❌ Analysis failed: {type(e).__name__}: {e}")
        audit.log_error("pipeline_failed", f"{type(e).__name__}: {e}",
                        {"iterations_used": feedback_loop_metadata.get("iterations_used")})
        log_path = audit.end_run(final_results={"error": str(e)}, status="failed")
        print(f"   Audit log: {log_path}")
        raise


# =============================================================================
# 7. SIMPLE EXAMPLE ENTRY POINT  (GIVEN)
# =============================================================================

def print_analysis_report(results: Dict[str, Any]) -> None:
    """Print each agent's final output and the per-iteration feedback-loop history."""
    stat = results["statistical_analysis"]
    clin = results["clinical_assessment"]
    reg = results["regulatory_report"]
    report = _safe_json_loads(reg.get("raw_data"))

    sections = [
        (f"STATISTICAL AGENT (revision {stat['revision_number']}, tools: {', '.join(stat['tool_usage'])})",
         stat["agent_response"]),
        (f"CLINICAL AGENT (revision {clin['revision_number']}, tools: {', '.join(clin['tool_usage'])})",
         clin["agent_response"]),
        (f"REGULATORY REPORT (revision {reg['revision_number']}, tool: generate_regulatory_report)",
         json.dumps({k: report.get(k) for k in ("executive_summary", "statistical_findings",
                                                 "clinical_evaluation", "regulatory_actions",
                                                 "quality_assurance")}, indent=2)),
    ]
    if reg.get("feedback_response"):
        sections.append(("REGULATORY RESPONSE TO QC FEEDBACK", reg["feedback_response"]))

    for title, body in sections:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}\n{body}")

    print(f"\n{'=' * 72}\nANTI-HALLUCINATION CHECKS\n{'=' * 72}")
    print(f"Statistical numbers not found in tool output: "
          f"{stat.get('numeric_grounding', {}).get('ungrounded_numbers') or 'none'}")
    ev = clin.get("evidence_check", {})
    print(f"Clinical evidence strength: '{ev.get('tool_evidence_strength')}' "
          f"({ev.get('shared_pubmed_articles')} PubMed articles); over-claims: "
          f"{ev.get('overclaimed_strengths') or 'none'}")
    print(f"Clinical classification: {clin.get('classification')} "
          f"(plausibility tool: {clin.get('tool_assessment')})")

    meta = results["feedback_loop_metadata"]
    print(f"\n{'=' * 72}\nFEEDBACK LOOP HISTORY\n{'=' * 72}")
    for h in meta["analysis_history"]:
        print(f"Iteration {h['iteration']}: QC score {h['quality_score']} "
              f"[{h['approval_status']}], targets: {h['target_agents'] or 'none'}")
        for a in h["actions"]:
            print(f"   - {a['agent']}: {a['action']} (revision {a['revision_number']})")
        for i in h["issues"]:
            print(f"   ! [{i['severity']}] {i['type']} -> {i['responsible_agent']}")
    print(f"Stop reason: {meta['stop_reason']}")


def run_example(
    drug_name: str = "MONTELUKAST",
    adverse_event: str = "AGGRESSION",
    max_iterations: int = 3,
    quality_threshold: float = 95.0,
) -> Dict[str, Any]:
    """
    Run a single example analysis.

    You can modify the drug / adverse_event to experiment.

    Try:
      - Clear-cut:    METFORMIN + LACTIC ACIDOSIS
      - Ambiguous:    MONTELUKAST + AGGRESSION   (good for seeing iterations)

    Full results are also saved as JSON in outputs/ (next to audit_logs/).
    """
    configure_faers_mode(use_live_api=True)
    llm = setup_environment()

    results = analyze_with_feedback_loop(
        drug_name=drug_name,
        adverse_event=adverse_event,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        llm=llm,
    )

    print_analysis_report(results)

    os.makedirs("outputs", exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", f"{drug_name}_{adverse_event}").strip("_")
    results_path = os.path.join(
        "outputs", f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    meta = results.get("feedback_loop_metadata", {})
    print("\n=== SUMMARY ===")
    print(f"Drug: {results['drug_name']}")
    print(f"Adverse event: {results['adverse_event']}")
    print(f"Final quality score: {meta.get('final_quality_score')}")
    print(f"Iterations used: {meta.get('iterations_used')}")
    print(f"Converged: {meta.get('converged')}")
    print(f"Stop reason: {meta.get('stop_reason')}")
    print(f"Audit log: {results.get('audit_log_path')}")
    print(f"Full results: {results_path}")
    return results


if __name__ == "__main__":
    # You can comment this out while you are working on TODOs.
    # Optional: python starter_code_pharmacovigilance_signal_detector.py DRUG "ADVERSE EVENT" [threshold]
    import sys

    if len(sys.argv) >= 3:
        run_example(sys.argv[1], sys.argv[2],
                    quality_threshold=float(sys.argv[3]) if len(sys.argv) > 3 else 95.0)
    else:
        run_example()

