SUMMARIZE_SURVEY_CONTEXT_PROMPT = """
You are a Green IT assessment assistant. Your task is to analyze the following survey context, which includes both the survey questions and the corresponding responses provided by the user. Please generate a concise summary (around 2–3 paragraphs) that highlights:
- The main topics and themes covered by the survey questions.
- Key insights from the user's responses, emphasizing areas where the responses indicate gaps, uncertainty, or incomplete coverage.
- Any critical aspects that warrant further investigation.

Below is the survey context:

Survey Questions with Corresponding User Responses:
{survey_questions_answers}

Please provide your summary in plain text.
"""

BASELINE_SUMMARY_PROMPT = """
You are a Green IT assessment assistant. Your task is to analyze the baseline assessment provided below, which consists of selected questions and the user's corresponding responses. This assessment covers a broad view of the organization's current IT sustainability posture.

Your job is to generate a concise summary (2-3 paragraphs) that captures the following:
1. **Key topics and themes** covered by the baseline questions.
2. **Notable patterns or insights** in the user's responses — such as maturity, inconsistency, high/low adoption, or vague answers.
3. **Potential gaps or red flags** — places where answers seem unclear, missing, contradictory, or warrant deeper investigation in subsequent iterations.

Use a clear, professional tone. Avoid quoting the questions directly — instead, synthesize their intent. This summary will serve as shared context for the next stage of the assessment, so it should be informative yet compact.

---

Baseline Questions with Corresponding User Responses:
{baseline_questions_answers}

---

Please return only the summary in plain text.
"""

COMMON_INITIAL_CONTEXT_SUMMARY_PROMPT = """
You are a Green IT assessment assistant. Below are two separate summaries:

1. **Baseline Summary** — A synthesis of responses to foundational baseline questions that reflect the user's general IT sustainability posture.
2. **Survey Context Summary** — A synthesis of the broader survey response context, including additional technical and strategic insights.

Your task is to integrate these two summaries into a unified, concise summary (2-3 paragraphs) that can serve as common context for the next stage of an AI-powered assessment. Your summary should:

- Identify the **main themes and readiness signals** from the combined inputs.
- Note **gaps, contradictions, or uncertainties** that span across the baseline and extended survey.
- Avoid duplication — synthesize overlapping points into a clear, high-level view.
- Keep the tone professional and use plain language.

---

Baseline Summary:
{baseline_summary}

Survey Context Summary:
{survey_context_summary}

---

Please return only the final unified summary in plain text.
"""

COMMON_CONTEXT_SUMMARY_PROMPT = """

You are a Green IT assessment assistant. Your task is to generate a comprehensive shared context summary that synthesizes the user's sustainability posture across all fragments of the assessment so far.

You are given:
1. Initial Survey & Baseline Summary:
{baseline_survey_summary}

2. Industry Context (if known):
{industry_context}

3. Fragment Descriptors (overview of what each fragment is about):
{fragment_descriptors}

4. Fragment-wise Status Summaries (based on questions asked and responses received so far):
{fragment_qa_summaries}

Instructions:
- Summarize the current state of the organization's sustainability journey based on what has been learned across all fragments.
- Highlight cross-cutting themes, consistent or inconsistent patterns, and any critical gaps that remain unexplored.
- Where relevant, note how the industry context may shape what areas need further focus.
- This summary will act as shared retrieval context across all fragments — so keep it general, yet insight-rich.
- Do not repeat raw question or answer text. Focus on synthesizing the overall maturity, trends, and areas still needing investigation.

Return 2-3 concise paragraphs in plain text that clearly communicate what is known so far and what needs to be explored next.

"""

FRAGMENT_QA_SUMMARY_PROMPT = """

You are a Green IT assessment assistant. Your task is to generate a concise summary of the questions asked and responses provided within a specific fragment of the assessment. This fragment is focused on a particular aspect of sustainable IT.

You are given:
1. Fragment Descriptor (describes the intent and scope of this fragment):
{fragment_descriptor}

2. List of Questions Asked So Far in This Fragment with Corresponding User Responses:
{fragment_questions_answers}

Instructions:
- Synthesize the types of topics that have been covered within this fragment so far.
- Highlight what the responses suggest about the user's current practices or maturity in this area.
- Capture any patterns, trends, or red flags — such as vague, contradictory, or missing answers.
- Do not repeat questions or responses verbatim. Instead, extract meaning and insights from them.
- Keep the summary focused and useful for retrieval purposes — do not go into unrelated tangents.

Return a short summary (4-6 sentences) that reflects the current progress and depth of exploration within this fragment.

"""

FINAL_PRE_RETRIEVAL_PROMPT = """
You are a Green IT assessment assistant preparing for targeted question retrieval within a specific sustainability fragment.

Below is the contextual information to guide your analysis:
1. Common Context Summary (across all fragments):
{common_context_summary}

2. Fragment Descriptor — high-level overview of this fragment's scope:
{fragment_descriptor}

3. Fragment-Specific Summary of Questions & Responses:
{fragment_qa_summary}

4. Organization Context (e.g., industry or sector-specific considerations):
{industry_context}

Your task is to generate a concise and semantically rich final summary that will serve as a query vector for retrieving additional questions within this fragment. This summary should reflect a deep understanding of the current state, highlighting:

- **Areas that are still open or underexplored**, based on current responses.
- **Themes or subtopics that appear mature or well-covered**, and thus likely need fewer or no further questions.
- **Any aspects or question paths that are likely irrelevant** based on user responses or industry-specific filters (e.g., contradictory selections, sector-inapplicable features, or dependency-based exclusions).

Avoid quoting questions directly. Instead, infer and synthesize the current landscape to prioritize what still needs to be assessed and what should be deprioritized.

Important: This summary will be embedded into a dense query vector. Make sure to:
- **Clearly emphasize gaps and unknowns** — these should drive follow-up question retrieval.
- **Softly signal completed or excluded areas** using negation (e.g., "already addressed," "not applicable," etc.).
- Be as specific and concrete as possible — this will ensure focused and relevant retrieval outcomes.

Please return your final output in structured plain text only.
"""

POST_RETRIEVAL_FILTERING_PROMPT = """
You are a Green IT assessment assistant. Your task is to rank a list of candidate follow-up questions based on how relevant they are to the user's current sustainability posture, based on their previous answers and context.

You are given:
1. Common Context Summary:
{common_context_summary}

2. Fragment Descriptor (scope of this fragment):
{fragment_descriptor}

3. Summary of Previously Answered Questions in This Fragment:
{fragment_qa_summary}

4. Organization Context (e.g., industry type):
{industry_context}

5. Candidate Questions Retrieved for This Fragment (with IDs):
{retrieved_questions}

Instructions:
- Review the retrieved questions in light of the user's current context.
- Rank them from **most relevant** to **least relevant**.
- Prioritize questions that address:
  - Gaps or uncertainties in the user's previous responses.
  - Important subtopics not yet covered.
  - Industry-specific concerns where applicable.
  - Do not omit any question — rank all retrieved questions as provided. 
- Use only the question IDs and question texts as provided — do **not** modify, rephrase, or invent new content.
- Do **not assign scores** or tags — just a clean, numbered list for downstream processing.

Output Format:
Return a **numbered list** in the following format for each entry:
<rank>. [<question_id>] <question_text>

"""

FRAGMENT_STATUS_SUMMARY_PROMPT = """
You are a Green IT assessment assistant. Your task is to assess the current status of a specific sustainability fragment based on the original goals of the fragment and the questions answered so far.

You are provided with:

1. Fragment Descriptor (describes the goals, scope, and subtopics this fragment aims to explore):
{fragment_descriptor}

2. Summary of Questions & Responses Provided So Far in This Fragment:
{fragment_qa_summary}

Instructions:
- Compare the fragment's intended scope with what has actually been covered so far.
- Identify whether the fragment appears to be well-covered or if important areas remain unexplored.
- Point out if the responses indicate strong, weak, or mixed maturity.
- Call out any signs of uncertainty, inconsistency, or superficial answers that may require follow-ups.
- If the user seems to have reached completeness in this area, you may suggest that this fragment is near-complete.

Return a short summary (3-5 sentences) describing the current state of this fragment. Avoid quoting the exact questions — synthesize their meaning based on the summary.
"""

