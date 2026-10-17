# Submission Evidence — Pharmacovigilance Signal Detector

All code changes are in [`starter/starter_code_pharmacovigilance_signal_detector.py`](../starter/starter_code_pharmacovigilance_signal_detector.py).
`pharma_tools.py` and `audit_logger.py` are unmodified.

Every file in this folder comes from a real run: `gpt-4o-mini` via the OpenAI API, live openFDA FAERS and live PubMed.

| Run | Command (from `starter/`) | QC score by iteration | Stop condition |
|---|---|---|---|
| **MONTELUKAST + AGGRESSION** (provided example) | `python starter_code_pharmacovigilance_signal_detector.py` | 92.5 → 97.0 | quality ≥ threshold (95) after 1 clinical revision |
| METFORMIN + LACTIC ACIDOSIS (clear-cut) | `... METFORMIN "LACTIC ACIDOSIS"` | 100 | quality ≥ threshold on first pass |
| ACETAMINOPHEN + HEADACHE (not convergent) | `... ACETAMINOPHEN HEADACHE` | 91.2 → 91.2 → 91.2 | `max_iterations` (3) reached |

Each run has three files:
- `console_*.txt`: full console output, with each agent's final narrative, the regulatory report, anti-hallucination checks and the loop history.
- `audit_*.json`: the `AuditLogger` trail.
- `results_*.json`: the full `results` dict, including `feedback_loop_metadata`.

## Completion checklist

| Checklist item | Where it is met | Evidence |
|---|---|---|
| System prompts clearly describe the Statistical and Clinical agents | `STATISTICAL_SYSTEM_PROMPT`, `CLINICAL_SYSTEM_PROMPT` (plus an optional `REGULATORY_SYSTEM_PROMPT`) | Narratives in `console_*.txt` follow the prompts' section format |
| Agents use the tools in `pharma_tools.py` (no manual number-crunching) | `STATISTICAL_TOOLS`, `CLINICAL_TOOLS`, `REGULATORY_TOOL`, `QC_TOOL` and the `run_*_agent` runners | `tool_usage` in `results_*.json`; "numbers not found in tool output: none" in every console |
| `rerun_agent_with_feedback(...)` re-runs the right agent and increments revisions | `rerun_agent_with_feedback` | Console: `Re-running clinical agent (revision 0 -> 1) with QC feedback:` followed by the feedback text |
| `analyze_with_feedback_loop(...)` iterates when needed and stops sensibly | `analyze_with_feedback_loop` | The three runs above cover all three stop conditions (threshold met, first-pass threshold, max iterations) |
| `feedback_loop_metadata` shows history and final metrics | `analyze_with_feedback_loop` | `results_*.json`: `iterations_used`, `final_quality_score`, `converged`, `stop_reason`, `analysis_history` (scores, issues, actions per iteration) |
| Audit log JSON has agent executions, QC passes, iterations and the final outcome | `audit.start_run`, `log_agent_execution`, `log_quality_review`, `log_iteration`, `end_run` | `audit_*.json`: `events`, `iterations`, `final_results`, `summary` |
| Example: QC triggers a revision, the score changes, a clear stop | Provided example | `console_montelukast_aggression.txt`: score 92.5 → 97.0, `Stop reason: quality_threshold_met` |

## Rubric mapping

### Data handling and ROR evaluation
- **FAERS data access:** `run_statistical_agent` calls `calculate_drug_event_statistics` on the live openFDA API. The documented local-CSV fallback works in two ways:
  - Explicitly, through the provided switches: `configure_faers_mode(use_live_api=False)` calls `load_faers_data()`, then `set_faers_data()`, then `set_use_live_api()`.
  - Automatically, when the live API returns no data.
- **Function that loads and parses report data:** `load_faers_data()` parses `data/faers_signal_detection_dataset.csv`. `run_statistical_agent` parses the tool JSON into `statistics_summary` and `ema_criteria_check`, which feed the signal decision, QC and the regulatory report.
- **API failures handled gracefully:**
  - If the API is down, the agent falls back to the local CSV and adds the flag `live_api_unavailable_used_local_csv_fallback`.
  - A drug that isn't in FAERS gets `drug_not_found_in_faers_or_query_failed`.
  - LLM errors produce a tool-only summary instead of an exception.
  - A failed re-run is logged and keeps the previous result.
  - All of these are flagged in the output and the audit log; nothing crashes.
- **ROR comes from the tools, not invented:** `calculate_drug_event_statistics` and `validate_statistical_results` are always both called. The narrative copies tool values, and `_ema_criteria_check` only compares tool values with the EMA thresholds.

### Prompts and tool wiring
- **Two distinct prompts:**
  - The statistical prompt forbids clinical recommendations and causality claims, and requires citing the tool numbers with their source.
  - The clinical prompt requires a plausibility classification and evidence strength that matches the PubMed count (0 = none, 1–4 = limited, 5–19 = moderate, ≥20 = strong).
  - Both require a **"Response to QC Feedback — What I changed"** section on revisions.
  - Both include step-by-step workflows and refer to FDA / EMA guidance (EMA signal criteria, Bradford Hill, FAERS limitations).
- **Tool mapping:** statistical → `calculate_drug_event_statistics`, `validate_statistical_results`; clinical → `search_clinical_literature`, `assess_biological_plausibility`; regulatory → `generate_regulatory_report`; QC → `quality_review_analysis`.

### Agent runners and QC
- **Runner outputs:** every runner returns `agent_response`, `tool_usage`, `raw_data`, `revision_number`, `status` and `feedback_addressed`.
  - The clinical runner receives `statistical_context`.
  - It builds `combined_for_qc` with the literature data, plausibility data, `narrative_response` and `_revision_number`, which are the keys `quality_review_analysis` reads.
- **QC:** `run_quality_control` passes all four inputs to the QC tool and parses its JSON. `get_feedback_for_agent` filters issues by `responsible_agent` and formats severity, description, required action and evidence needed. The loop routes re-runs from `qc["target_agents"]`.
- **Anti-hallucination checks** (reported in each console and `results_*.json`):
  - `numeric_grounding`: every number in the statistical narrative must appear in the tool output. Result: 0 ungrounded numbers in all runs.
  - `evidence_check`: evidence-strength claims are compared with the PubMed label (using the more conservative of the two PubMed searches). Result: no over-claims.
  - Classification is checked for consistency with the plausibility tool.
  - The regulatory report is always rebuilt from the latest statistical and clinical raw outputs.

### Feedback loop
- **Stopping:** `while iteration < max_iterations and quality_score < quality_threshold`, with a break when no routable `target_agents` remain. `stop_reason` records which condition ended the loop.
- **Targeted re-execution:** only QC-targeted agents are re-run. Each re-run gets the QC feedback (and its previous version) in its prompt, and its updated result replaces that agent's entry in `current_results`.
- **Regulatory regeneration:** when statistical or clinical output changes, the regulatory report is regenerated (`regulatory: regenerated (revision 1)` in the MONTELUKAST history).
- **Regulatory-targeted feedback** (ACETAMINOPHEN): the report tool computes the risk level deterministically, so the regulatory agent adds an evidence-based "Response to QC Feedback" justifying the LOW risk instead of editing the report. QC keeps flagging the same issue, so the loop stops at `max_iterations` with `converged=False`. This case shows the third stop condition.

### Audit logging and traceability
- **Lifecycle:** `start_run` at the beginning and `end_run` at completion (status `completed`, or `failed` on error). `log_agent_execution` runs for the initial runs (revision 0) and every revision (revision > 0, with a feedback snippet). `log_quality_review` and `log_iteration` run on every QC pass.
- **Decision-making record:** the log also has `statistical_signal_assessment`, `clinical_plausibility_assessment`, `regulatory_report_generated`, `final_report_generated` and `feedback_loop_completed` events (signal decision, classification, risk level, stop reason).
- **Chronology in `audit_*_MONTELUKAST.json`:** `analysis_started` → `literature_fetched` → statistical / clinical / regulatory runs → `quality_review` (92.5) → clinical revision 1 → regulatory revision 1 → `quality_review` (97.0) → `feedback_loop_completed` → `analysis_completed`.

## How to reproduce
```bash
cd project
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# project/.env: OPENAI_API_KEY=... (and BASE_URL=https://openai.vocareum.com/v1 for Vocareum keys)
cd starter
python starter_code_pharmacovigilance_signal_detector.py
```
New runs write to `starter/audit_logs/` and `starter/outputs/`. Both are git-ignored; the curated copies are in this folder.

*Educational project only. Not for clinical decision-making.*
