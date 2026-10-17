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
from datetime import datetime
from typing import Any, Dict, List, Optional

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


def configure_faers_mode(use_live_api: bool = True) -> None:
    """
    Switch between live FAERS API and any local fallback data.

    For this course we recommend using the LIVE API (default=True).
    """
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
3. Signal decision: apply EMA-style criteria (case count >= 3, ROR >= 2, lower 95% CI bound > 1,
   or p < 0.05 when the chi-square test is valid). State SIGNAL DETECTED or SIGNAL NOT DETECTED
   and confirm it matches the tool's signal_detection.signal_detected field.
4. Validation: summarise the validation status and any issues it raised.
5. Limitations: explain data-quality limits relevant to THIS result (low counts, wide CI,
   data_quality flags) plus inherent FAERS limits: under-reporting, stimulated/notoriety
   reporting, duplicate reports, confounding by indication, no exposure denominator
   (disproportionality is not incidence or risk).

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
   stimulated reporting after safety communications).
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
    "calculate_stats": None,   # TODO: pharma_tools.calculate_drug_event_statistics
    "validate_stats": None,    # TODO: pharma_tools.validate_statistical_results
}

CLINICAL_TOOLS = {
    "search_literature": None,     # TODO: pharma_tools.search_clinical_literature
    "assess_plausibility": None,   # TODO: pharma_tools.assess_biological_plausibility
}

REGULATORY_TOOL = None   # TODO: pharma_tools.generate_regulatory_report
QC_TOOL = None           # TODO: pharma_tools.quality_review_analysis


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


def run_statistical_agent(
    llm: ChatOpenAI,
    drug_name: str,
    adverse_event: str,
    literature_evidence: Dict[str, Any],
    feedback_context: Optional[str] = None,
    revision_number: int = 0,
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
    raise NotImplementedError("Implement run_statistical_agent()")


def run_clinical_agent(
    llm: ChatOpenAI,
    drug_name: str,
    adverse_event: str,
    literature_evidence: Dict[str, Any],
    statistical_context: str,
    feedback_context: Optional[str] = None,
    revision_number: int = 0,
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
    raise NotImplementedError("Implement run_clinical_agent()")


def run_regulatory_agent(
    drug_name: str,
    adverse_event: str,
    statistical_data_json: str,
    clinical_combined_json: str,
    literature_evidence: Dict[str, Any],
    feedback_loop_summary: Optional[Dict[str, Any]] = None,
    revision_number: int = 0,
) -> Dict[str, Any]:
    """
    TODO 1C: Implement the regulatory 'agent'.

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
    raise NotImplementedError("Implement run_regulatory_agent()")


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
    raise NotImplementedError("Implement run_quality_control()")


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
    raise NotImplementedError("Implement get_feedback_for_agent()")


# =============================================================================
# 5. FEEDBACK-DRIVEN RE-RUNNING  (TODO 3)
# =============================================================================

def rerun_agent_with_feedback(
    llm: ChatOpenAI,
    agent_name: str,
    feedback: str,
    drug_name: str,
    adverse_event: str,
    literature_evidence: Dict[str, Any],
    current_results: Dict[str, Any],
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
    """
    raise NotImplementedError("Implement rerun_agent_with_feedback()")


# =============================================================================
# 6. MAIN ORCHESTRATION: FEEDBACK LOOP  (TODO 4)
# =============================================================================

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
    raise NotImplementedError("Implement analyze_with_feedback_loop()")


# =============================================================================
# 7. SIMPLE EXAMPLE ENTRY POINT  (GIVEN)
# =============================================================================

def run_example() -> None:
    """
    Run a single example analysis.

    You can modify the drug / adverse_event to experiment.

    Try:
      - Clear-cut:    METFORMIN + LACTIC ACIDOSIS
      - Ambiguous:    MONTELUKAST + AGGRESSION   (good for seeing iterations)
    """
    configure_faers_mode(use_live_api=True)
    llm = setup_environment()

    results = analyze_with_feedback_loop(
        drug_name="MONTELUKAST",
        adverse_event="AGGRESSION",
        max_iterations=3,
        quality_threshold=95.0,
        llm=llm,
    )

    meta = results.get("feedback_loop_metadata", {})
    print("\n=== SUMMARY ===")
    print(f"Drug: {results['drug_name']}")
    print(f"Adverse event: {results['adverse_event']}")
    print(f"Final quality score: {meta.get('final_quality_score')}")
    print(f"Iterations used: {meta.get('iterations_used')}")
    print(f"Converged: {meta.get('converged')}")
    print(f"Audit log: {results.get('audit_log_path')}")


if __name__ == "__main__":
    # You can comment this out while you are working on TODOs.
    run_example()

