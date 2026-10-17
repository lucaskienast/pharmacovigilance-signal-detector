# Pharmacovigilance Signal Detector

## Multi-Agent AI System for Drug Safety Analysis

You will build a **4‑agent AI system** that analyzes real FDA adverse event data (FAERS, via the openFDA API) and real PubMed literature to detect potential drug safety signals and generate a structured safety report.

Your main job is to **orchestrate the agents and implement the feedback loop**. The heavy lifting for data (FAERS, PubMed, plausibility, QC, regulatory report) is already implemented for you in `pharma_tools.py`.

This is an educational project only; do **not** use it for real clinical decisions.

---

## 1. What You Need to Build

Your system should include:

1. **Statistical Agent**
   - Uses FAERS tools to compute ROR/PRR and related statistics.
   - Interprets whether a safety signal is present and notes data‑quality limits.

2. **Clinical Agent**
   - Uses PubMed + biological plausibility tools.
   - Decides if the association is **plausible / implausible / uncertain**.
   - Aligns evidence strength with the **actual** number of PubMed articles.

3. **Regulatory Agent**
   - Uses a report‑generator tool to produce a structured safety report.
   - Reflects the statistical and clinical findings and feedback‑loop metadata.

4. **Quality Control (QC) Agent**
   - Uses a QC tool to:
     - Score the analysis (alignment, completeness, risk alignment, actionability).
     - Identify issues and **which agent** is responsible (e.g. `["clinical"]`).
     - Drive the feedback loop.

5. **Feedback Loop**
   - QC → feedback → targeted agent re‑runs → QC → … until:
     - Quality score ≥ threshold, **or**
     - No agents are targeted for revision, **or**
     - You hit `max_iterations`.

6. **Audit Log (Minimal)**
   - Uses `AuditLogger` to log at least:
     - Initial agent executions.
     - Agent revisions inside the loop (`log_agent_execution`).
     - Each QC pass (`log_quality_review`).
     - Final `end_run(...)` status.

---

## 2. Files You Work With

**Do NOT modify**:

- `pharma_tools.py`  
  - Real tools for:
    - FAERS (openFDA) statistics
    - PubMed evidence
    - Mechanistic plausibility
    - Regulatory report generation
    - Quality control scoring and issues

- `audit_logger.py`  
  - Helpers for:
    - `start_run(...)`
    - `log_agent_execution(...)`
    - `log_quality_review(...)`
    - `log_iteration(...)`
    - `end_run(...)`

**You WILL modify**:

- `starter_code_pharmacovigilance_signal_detector.py` (scaffold)
  - Contains:
    - Environment / LLM setup.
    - Empty or partially implemented helpers with **TODOs**.
  - You will implement:
    - System prompts for the agents.
    - Tool assignment (which pharma_tools functions each agent uses).
    - Agent runners for statistical, clinical, regulatory agents.
    - QC runner and feedback extraction.
    - `rerun_agent_with_feedback(...)` (feedback‑driven re‑runs).
    - Full feedback loop in `analyze_with_feedback_loop(...)`.
    - Calls to `audit.log_agent_execution(...)` and `audit.log_quality_review(...)` inside the loop.

---

## 3. Environment Setup

Create a `.env` file in the project root.

### Vocareum (Udacity hosted)

```bash
OPENAI_API_KEY=voc-your-vocareum-key-here
BASE_URL=https://openai.vocareum.com/v1
```

### Direct OpenAI

```bash
OPENAI_API_KEY=your-openai-api-key-here
# BASE_URL=  # leave unset to use the default OpenAI endpoint
```

The scaffold already calls `load_dotenv()` for you. Once `.env` is in place, you can run:

```bash
python starter_code_pharmacovigilance_signal_detector.py
```

to execute the built‑in example.

---

## 4. Core TODOs in `starter_code_pharmacovigilance_signal_detector.py`

### ✅ TODO 0 – Prompts & Tool Assignment

- Write effective **system prompts** for:
  - Statistical agent.
  - Clinical agent.
- Assign the correct functions from `pharma_tools.py` to:
  - `STATISTICAL_TOOLS`
  - `CLINICAL_TOOLS`
  - `REGULATORY_TOOL`
  - `QC_TOOL`
- You’ll need to inspect `pharma_tools.py` to see function names and arguments.

---

### ✅ TODO 1 – Agent Runners

Implement:

- `run_statistical_agent(...)`
  - Call FAERS tools to compute and validate stats.
  - Ask the LLM (with your statistical system prompt) to interpret the JSON.
- `run_clinical_agent(...)`
  - Call PubMed + biological plausibility tools.
  - Ask the LLM (with your clinical system prompt) to synthesize a narrative.
  - Provide a **combined clinical JSON** string for QC & regulatory.
- `run_regulatory_agent(...)`
  - Call the regulatory report generator tool with the latest stats + clinical JSON + literature.

Each runner should return a dict with at least:

```python
{
    "agent_response": "...",          # narrative or JSON
    "tool_usage": [...],
    "raw_data": <JSON string or dict>,
    "status": "completed" or "revised",
    "revision_number": int,
    "feedback_addressed": Optional[str],
}
```

---

### ✅ TODO 2 – QC Runner & Feedback Extraction

- Implement `run_quality_control(...)`:
  - Call `QC_TOOL` (`quality_review_analysis`) with:
    - Statistical findings (JSON string).
    - Combined clinical findings (JSON string).
    - Regulatory report (JSON string).
    - Literature evidence (JSON string).
  - Parse the JSON string into a dict and return it (keep the raw string too).
- Implement `get_feedback_for_agent(agent_name, issues)`:
  - Filter issues for the given `agent_name`.
  - Concisely summarize type, severity, description, specific_action, and evidence_needed.

---

### ✅ TODO 3 – `rerun_agent_with_feedback(...)`

Implement this helper so that it:

1. Looks up the previous result for `agent_name` in `current_results`:
   - `"statistical"` → `"statistical_analysis"`
   - `"clinical"` → `"clinical_assessment"`
   - `"regulatory"` → `"regulatory_report"`
2. Reads its `revision_number` (default 0), increments by 1.
3. Calls the correct runner:
   - `"statistical"` → `run_statistical_agent(...)`
   - `"clinical"` → `run_clinical_agent(...)`
   - `"regulatory"` → `run_regulatory_agent(...)`
4. Passes:
   - `feedback` as `feedback_context`.
   - The new `revision_number`.
   - For the clinical agent: the latest statistical `agent_response` as context.
   - For the regulatory agent: the latest stats JSON and combined clinical JSON.
5. Returns the revised result dict.

This is what turns QC feedback into **targeted agent re‑runs**.

---

### ✅ TODO 4 – Feedback Loop in `analyze_with_feedback_loop(...)`

Implement a full pipeline with a **real feedback loop**:

1. **Setup**
   - Create the LLM (if not provided).
   - Create `AuditLogger` and call `audit.start_run(...)`.

2. **Iteration 1 (initial pass)**
   - Fetch literature evidence once with `fetch_literature_evidence(...)`.
   - Run statistical, clinical, and regulatory agents once.
   - Store them in `current_results`.
   - Call `audit.log_agent_execution(...)` for each agent.

3. **QC Pass**
   - Call `run_quality_control(...)` to get:
     - `quality_scores` (including `overall_score`).
     - `target_agents`.
     - `issues`.
   - Call `audit.log_quality_review(...)`.

4. **Feedback Loop**
   - Maintain `iteration`, `quality_score`, and a `feedback_loop_metadata` dict.
   - While `iteration < max_iterations` **and** `quality_score < quality_threshold`:
     - If `target_agents` is empty → mark `converged=True` and break.
     - For each `agent_name` in `target_agents`:
       - Build feedback with `get_feedback_for_agent(...)`.
       - Call `rerun_agent_with_feedback(...)`.
       - Update `current_results[...]` for that agent.
       - Call `audit.log_agent_execution(...)` for each revised agent.
     - If **statistical or clinical** changed, regenerate the regulatory report and update `current_results["regulatory_report"]`.
     - Re‑run QC with updated results.
     - Update `quality_score`, `target_agents`.
     - Append an iteration record to `feedback_loop_metadata["analysis_history"]`.
     - Call `audit.log_quality_review(...)` for each QC pass.

   - At the end, set:
     - `feedback_loop_metadata["iterations_used"]`
     - `feedback_loop_metadata["final_quality_score"]`
     - `feedback_loop_metadata["converged"]`

5. **Return Results**
   - Build a final `results` dict with:
     - `drug_name`, `adverse_event`, `timestamp`
     - `literature_evidence`
     - `statistical_analysis`, `clinical_assessment`, `regulatory_report`
     - `quality_review`, `feedback_loop_metadata`
     - `audit_log_path` (from `audit.end_run(...)`)

---

## 5. How to Run & What to Look For

From the command line:

```bash
python starter_code_pharmacovigilance_signal_detector.py
```

By default, `run_example()` analyzes:

- `MONTELUKAST + AGGRESSION`

This is a good case to see your feedback loop in action once implemented.

Check the output for:

- QC quality scores per iteration.
- Which agents were re‑run and why.
- Whether quality eventually reaches the threshold.
- The path to the generated audit log file.

Then try your own pairs, e.g.:

- Clear‑cut: `METFORMIN + LACTIC ACIDOSIS`
- Other combinations you’re curious about.

---

## 6. Submission Checklist

Before you’re done, verify:

- [ ] System prompts clearly define roles & responsibilities of statistical and clinical agents.
- [ ] Correct pharma_tools functions are assigned to `STATISTICAL_TOOLS`, `CLINICAL_TOOLS`, `REGULATORY_TOOL`, and `QC_TOOL`.
- [ ] `run_statistical_agent(...)`, `run_clinical_agent(...)`, and `run_regulatory_agent(...)` call tools correctly and return structured results.
- [ ] `run_quality_control(...)` calls the QC tool and parses its JSON.
- [ ] `get_feedback_for_agent(...)` returns focused, actionable feedback per agent.
- [ ] `rerun_agent_with_feedback(...)` re‑runs the correct agent with feedback and increments `revision_number`.
- [ ] `analyze_with_feedback_loop(...)` uses a loop with:
  - [ ] `max_iterations`
  - [ ] `quality_threshold`
  - [ ] “no target agents” as a stopping condition.
- [ ] `feedback_loop_metadata` contains per‑iteration history and final metrics.
- [ ] `AuditLogger` is used to log at least:
  - [ ] Initial agent executions.
  - [ ] Agent revisions inside the loop (`log_agent_execution`).
  - [ ] Each QC pass (`log_quality_review`).
  - [ ] Final `end_run(...)` call.

If you can run `python starter_code_pharmacovigilance_signal_detector.py` and see at least one case where QC triggers a revision and quality improves or is clearly documented, you’ve achieved the core learning goals.
