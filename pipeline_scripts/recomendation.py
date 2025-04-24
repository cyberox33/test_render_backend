import os
import json
import time
import re
from collections import defaultdict, deque # deque needed if call_llm is in this file
import traceback
from typing import List, Dict, Any, Optional, Tuple
import sys, os
import google.generativeai as genai
import google.api_core.exceptions # For potential error handling in main script if needed
from supabase import create_client, Client
from dotenv import load_dotenv
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.dml.color import RGBColor
import matplotlib.pyplot as plt
import argparse
import io # To save matplotlib plot to memory

# --- Configuration ---
load_dotenv()

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.utils import config

SUPABASE_URL = config.SUPABASE_URL
SUPABASE_KEY = config.SUPABASE_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=config.GOOGLE_API_KEY)

LLM_MODEL_NAME = "gemma-3-27b-it"
# --- Rate Limiting Globals ---
LLM_CALL_TIMESTAMPS = deque()
LLM_TOKEN_TIMESTAMPS = deque() # For TPM tracking
MAX_RPM = 30
TARGET_RPM = MAX_RPM - 3
TPM_LIMIT = 15000 # Tokens Per Minute limit (adjust if needed for model)
TARGET_TPM = TPM_LIMIT * 0.9 # Use 90% of TPM limit as target
RPM_WINDOW_SECONDS = 60
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2

# --- Constants ---
MAX_CHUNK_TOKENS = 7500 # Max estimated tokens per chunk (input only)
HOURS_PER_DAY = 8 # Define conversion rate for graph
DEFAULT_LAYOUT_TITLE_CONTENT = 1 # Standard index for "Title and Content"
DEFAULT_LAYOUT_SECTION_HEADER = 2 # Standard index for "Section Header"
DEFAULT_LAYOUT_TITLE_ONLY = 5 # Standard index for "Title Only"
DEFAULT_LAYOUT_BLANK = 6 # Standard index for "Blank"


# --- LLM Call Function (Modified for TPM) ---
def call_llm(prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.4) -> Optional[str]:
    """
    Calls the configured Gemini model with rate limiting (RPM & TPM) and retry logic.
    Uses model.count_tokens for accurate input token estimation.
    """
    global LLM_CALL_TIMESTAMPS, LLM_TOKEN_TIMESTAMPS # Use global deques

    current_time = time.monotonic()

    # --- RPM Rate Limiting ---
    while LLM_CALL_TIMESTAMPS and LLM_CALL_TIMESTAMPS[0] <= current_time - RPM_WINDOW_SECONDS:
        LLM_CALL_TIMESTAMPS.popleft()

    if len(LLM_CALL_TIMESTAMPS) >= TARGET_RPM:
        oldest_call_in_window = LLM_CALL_TIMESTAMPS[0]
        wait_time = (oldest_call_in_window + RPM_WINDOW_SECONDS) - current_time
        wait_time = max(0, wait_time)
        print(f"RPM limit approaching ({len(LLM_CALL_TIMESTAMPS)}/{TARGET_RPM}). Waiting for {wait_time:.2f} seconds...")
        time.sleep(wait_time + 0.1)
        current_time = time.monotonic()
        while LLM_CALL_TIMESTAMPS and LLM_CALL_TIMESTAMPS[0] <= current_time - RPM_WINDOW_SECONDS:
            LLM_CALL_TIMESTAMPS.popleft()

    # --- TPM Rate Limiting ---
    estimated_output_tokens = max_tokens if max_tokens is not None else 1024 # Default max output
    estimated_input_tokens = 0
    try:
        # Initialize model instance for counting if needed (can be optimized)
        model_for_counting = genai.GenerativeModel(LLM_MODEL_NAME)
        count_response = model_for_counting.count_tokens(prompt)
        estimated_input_tokens = count_response.total_tokens
        # print(f"Accurate input token estimate: {estimated_input_tokens}") # Verbose logging
    except Exception as e:
        print(f"Warning: Token counting failed ({e}). Using character estimation fallback.")
        estimated_input_tokens = len(prompt) // 4 # Rough fallback

    current_call_estimated_tokens = estimated_input_tokens + estimated_output_tokens

    while LLM_TOKEN_TIMESTAMPS and LLM_TOKEN_TIMESTAMPS[0][0] <= current_time - RPM_WINDOW_SECONDS:
        LLM_TOKEN_TIMESTAMPS.popleft()
    tokens_in_window = sum(t[1] for t in LLM_TOKEN_TIMESTAMPS)

    while tokens_in_window + current_call_estimated_tokens > TARGET_TPM and LLM_TOKEN_TIMESTAMPS:
        oldest_token_time, oldest_token_count = LLM_TOKEN_TIMESTAMPS[0]
        wait_time = (oldest_token_time + RPM_WINDOW_SECONDS) - current_time
        wait_time = max(0, wait_time)
        print(f"TPM limit approaching ({(tokens_in_window + current_call_estimated_tokens):.0f}/{TARGET_TPM:.0f} estimated). Waiting for {wait_time:.2f} seconds...")
        time.sleep(wait_time + 0.1)
        current_time = time.monotonic()
        while LLM_TOKEN_TIMESTAMPS and LLM_TOKEN_TIMESTAMPS[0][0] <= current_time - RPM_WINDOW_SECONDS:
             LLM_TOKEN_TIMESTAMPS.popleft()
        tokens_in_window = sum(t[1] for t in LLM_TOKEN_TIMESTAMPS)

    # Record RPM attempt time *before* the call
    # We store the *intended* start time for RPM calculation
    intended_start_time = current_time
    LLM_CALL_TIMESTAMPS.append(intended_start_time)

    # --- API Call with Retry Logic ---
    backoff_time = INITIAL_BACKOFF_SECONDS
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Calling LLM (Input est: ~{estimated_input_tokens} tokens)...")
            llm = genai.GenerativeModel(LLM_MODEL_NAME)
            gen_config = genai.types.GenerationConfig(
                temperature=temperature,
                # Use max_output_tokens if provided, else use model's default (often large)
                # Be mindful this affects TPM estimation.
                max_output_tokens=max_tokens if max_tokens is not None else None
            )
            response = llm.generate_content(prompt, generation_config=gen_config)

            if response.parts:
                total_tokens_used = 0
                try:
                    prompt_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                    total_tokens_used = response.usage_metadata.total_token_count
                    print(f"LLM Call Success. Tokens: In={prompt_tokens}, Out={output_tokens}, Total={total_tokens_used}")
                except (AttributeError, TypeError):
                    actual_output_tokens_est = len(response.text) // 4
                    total_tokens_used = estimated_input_tokens + actual_output_tokens_est
                    print(f"LLM Call Success. Using Estimated Tokens: In={estimated_input_tokens:.0f}, Out={actual_output_tokens_est:.0f}, Total={total_tokens_used:.0f}")

                # Append actual tokens used to TPM deque
                LLM_TOKEN_TIMESTAMPS.append((time.monotonic(), total_tokens_used))
                return response.text.strip()
            elif response.prompt_feedback.block_reason:
                 print(f"Warning: LLM call blocked. Reason: {response.prompt_feedback.block_reason}")
                 if LLM_CALL_TIMESTAMPS and LLM_CALL_TIMESTAMPS[-1] == intended_start_time: LLM_CALL_TIMESTAMPS.pop()
                 return None
            else:
                 print("Warning: LLM returned empty response.")
                 if LLM_CALL_TIMESTAMPS and LLM_CALL_TIMESTAMPS[-1] == intended_start_time: LLM_CALL_TIMESTAMPS.pop()
                 return None

        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"Error: ResourceExhausted (likely rate limit) on attempt {attempt + 1}: {e}")
            if LLM_CALL_TIMESTAMPS and LLM_CALL_TIMESTAMPS[-1] == intended_start_time: LLM_CALL_TIMESTAMPS.pop()
            if attempt < MAX_RETRIES - 1:
                print(f"Waiting {backoff_time:.2f} seconds before retry...")
                time.sleep(backoff_time)
                backoff_time *= 2
                intended_start_time = time.monotonic() # Update time for next attempt
                LLM_CALL_TIMESTAMPS.append(intended_start_time) # Add timestamp for next attempt's RPM check
            else:
                print("Max retries reached after rate limit error.")
                return None
        except Exception as e:
             print(f"Error during LLM call on attempt {attempt + 1}: {e}")
             if LLM_CALL_TIMESTAMPS and LLM_CALL_TIMESTAMPS[-1] == intended_start_time: LLM_CALL_TIMESTAMPS.pop()
             if "API key not valid" in str(e):
                 print("Critical: Invalid Google API Key. Aborting retry.")
                 return None
             if attempt < MAX_RETRIES - 1:
                 print(f"Waiting {backoff_time:.2f} seconds before retry...")
                 time.sleep(backoff_time)
                 backoff_time *= 2
                 intended_start_time = time.monotonic() # Update time for next attempt
                 LLM_CALL_TIMESTAMPS.append(intended_start_time) # Add timestamp for next attempt's RPM check
             else:
                 print("Max retries reached after general error.")
                 return None

    # Safety cleanup if loop exits unexpectedly
    if LLM_CALL_TIMESTAMPS and LLM_CALL_TIMESTAMPS[-1] == intended_start_time: LLM_CALL_TIMESTAMPS.pop()
    return None


# --- Database Interaction Functions (Mostly Unchanged) ---

def get_supabase_client() -> Client:
    """Returns the initialized Supabase client."""
    global supabase
    # Re-initialize if needed (e.g., in long-running processes, though unlikely here)
    if not supabase:
         print("Re-initializing Supabase client...")
         supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

def fetch_followup_data(supabase_client: Client, session_id: str) -> List[Dict[str, Any]]:
    """Fetches followup responses for a given session_id."""
    try:
        response = supabase_client.table("followup_responses").select("*").eq("session_id", session_id).execute()
        if response.data:
            return response.data
        else:
            print(f"No followup responses found for session_id: {session_id}")
            return []
    except Exception as e:
        print(f"Error fetching followup responses: {e}")
        traceback.print_exc()
        return []

def fetch_question_details(supabase_client: Client, question_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Fetches question details for a list of question IDs."""
    if not question_ids:
        return {}
    try:
        # Ensure IDs are integers
        valid_question_ids = [int(qid) for qid in question_ids if isinstance(qid, (int, str)) and str(qid).isdigit()]
        if not valid_question_ids:
            print("No valid integer question IDs provided.")
            return {}

        response = supabase_client.table("questions").select(
            "id, category, subcategory, sr_no, fragments, question, additional_fields, text_representation"
        ).in_("id", valid_question_ids).execute()

        if response.data:
            return {q['id']: q for q in response.data}
        else:
            print(f"No questions found matching IDs: {valid_question_ids}")
            return {}
    except Exception as e:
        print(f"Error fetching question details: {e}")
        traceback.print_exc()
        return {}

def fetch_fragment_info(supabase_client: Client, fragment_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetches fragment descriptions for a list of CLEANED fragment IDs."""
    if not fragment_ids:
        return {}
    try:
        unique_fragment_ids = list(set(filter(None, fragment_ids)))
        if not unique_fragment_ids:
             print("Warning: No valid unique fragment IDs provided to fetch_fragment_info.")
             return {}
        # print(f"Fetching fragment info for IDs: {unique_fragment_ids}") # Debug
        response = supabase_client.table("fragment_info").select(
            "fragment, description, category"
        ).in_("fragment", unique_fragment_ids).execute()

        if response.data:
            # print(f"Successfully fetched info for fragments: {[f['fragment'] for f in response.data]}") # Debug
            return {f['fragment']: f for f in response.data}
        else:
            print(f"Warning: No fragment info found matching fragments: {unique_fragment_ids} in the database.")
            return {}
    except Exception as e:
        print(f"Error fetching fragment info: {e}")
        traceback.print_exc()
        return {}

# --- Data Processing Functions (Mostly Unchanged) ---

def process_additional_fields(question: Dict[str, Any], answer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters additional_fields based on the answer provided, according to the rules.
    Returns the filtered additional_fields dictionary. Uses CORRECTED LOGIC.
    """
    additional_fields = question.get("additional_fields")
    if not additional_fields: return {}

    if isinstance(additional_fields, str):
        try: additional_fields = json.loads(additional_fields)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse additional_fields JSON for QID {question.get('id')}")
            return {}
    if not isinstance(additional_fields, dict):
        print(f"Warning: additional_fields is not dict for QID {question.get('id')}. Type: {type(additional_fields)}")
        return {}

    filtered_fields = {}
    user_answer = answer_data.get("value") if isinstance(answer_data, dict) else None

    always_include_keys = ["guidelines", "answer_options", "sub_sub_category", "subjective_answer", "multiple_correct_answer_options", "why_use_these_tools"]
    for key in always_include_keys:
        if key in additional_fields: filtered_fields[key] = additional_fields[key]

    retrieve_all_fields = False
    processed_specifically = False

    # --- Apply Conditional Logic ---
    if user_answer == "Implemented":
        processed_specifically = True
        filtered_fields.pop("recommendation", None); filtered_fields.pop("recommendations", None); filtered_fields.pop("tools", None)
    elif user_answer == "Not Implemented":
        retrieve_all_fields = True; processed_specifically = True
    elif user_answer == "dont know" or "subjective_answer" in additional_fields or "multiple_correct_answer_options" in additional_fields:
        retrieve_all_fields = True; processed_specifically = True
    elif "recommendations" in additional_fields and isinstance(additional_fields["recommendations"], dict):
        matched_recommendation = None; match_found_in_b = False
        normalized_answer = user_answer.strip().lower() if user_answer else ""
        for rec_key, rec_value in additional_fields["recommendations"].items():
            normalized_key = rec_key.strip().lower()
            key_base = re.sub(r'\s*\(level\s*\d+\)$', '', normalized_key).strip()
            if normalized_answer == normalized_key or \
               (key_base and normalized_answer == key_base) or \
               (key_base and normalized_answer.startswith(key_base)):
                matched_recommendation = {rec_key: rec_value}; match_found_in_b = True; break
        if match_found_in_b:
            processed_specifically = True
            filtered_fields["recommendations"] = matched_recommendation
            if "tools" in additional_fields: filtered_fields["tools"] = additional_fields["tools"]
        else:
            # print(f"Info: Answer '{user_answer}' did not match specific key in 'recommendations' for QID {question.get('id')}. Retrieving all.") # Verbose
            retrieve_all_fields = True
    if not processed_specifically: retrieve_all_fields = True

    if retrieve_all_fields:
        temp_copy = additional_fields.copy()
        temp_copy.update(filtered_fields) # Preserve already added base fields
        filtered_fields = temp_copy

    return {k: v for k, v in filtered_fields.items() if v is not None}


def group_data_by_fragment(
    followup_responses: List[Dict[str, Any]],
    questions_details: Dict[int, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Groups processed question data by fragment, using cleaned fragment IDs."""
    grouped_data = defaultdict(lambda: {"fragment_info": None, "questions_data": []})
    processed_fragments = set() # Store cleaned fragment IDs

    for resp in followup_responses:
        q_id = resp.get("question_id")
        question_detail = questions_details.get(q_id)
        if not question_detail:
            # print(f"Warning: Details not found for question_id {q_id}. Skipping.") # Verbose
            continue

        fragment_list = question_detail.get("fragments")
        raw_fragment_id = None; cleaned_fragment_id = None
        if isinstance(fragment_list, (list, tuple)) and fragment_list:
            raw_fragment_id = str(fragment_list[0]).strip()
        elif isinstance(fragment_list, str):
            raw_fragment_id = fragment_list.strip()
        else: continue # Skip invalid format

        if raw_fragment_id:
            match = re.match(r'^[\[\'\"]*(.*?)[\'\"\]]*$', raw_fragment_id)
            cleaned_fragment_id = match.group(1) if match else raw_fragment_id
        if not cleaned_fragment_id: continue # Skip empty/invalid ID

        processed_fragments.add(cleaned_fragment_id)

        answer_json = resp.get("answer")
        if isinstance(answer_json, str):
            try: answer_json = json.loads(answer_json)
            except json.JSONDecodeError: answer_json = {}
        elif not isinstance(answer_json, dict): answer_json = {}

        processed_add_fields = process_additional_fields(question_detail, answer_json)

        question_data = {
            "question_id": q_id,
            "question_text": question_detail.get("question"),
            "category": question_detail.get("category"),
            "subcategory": question_detail.get("subcategory"),
            "user_answer": answer_json, # Store the parsed dictionary
            "processed_additional_fields": processed_add_fields,
            # "raw_additional_fields": question_detail.get("additional_fields") # Optional: Keep if needed for debugging
        }
        grouped_data[cleaned_fragment_id]["questions_data"].append(question_data)

    if processed_fragments:
        all_fragment_info = fetch_fragment_info(get_supabase_client(), list(processed_fragments))
        for frag_id, data in grouped_data.items():
            data["fragment_info"] = all_fragment_info.get(frag_id, {"fragment": frag_id, "description": "N/A", "category": "N/A"})
    else: print("No valid fragments found to fetch info for.")

    return dict(grouped_data)

# --- LLM Prompt Generation Functions (REFINED) ---
def format_chunk_for_prompt(fragment_info: Dict[str, Any], questions_chunk: List[Dict[str, Any]]) -> str:
    """Formats a chunk of questions for a fragment into a string for LLM prompts."""
    output = []
    output.append(f"Fragment ID: {fragment_info.get('fragment', 'N/A')}")
    output.append(f"Fragment Name/Category: {fragment_info.get('category', 'N/A')}")
    output.append(f"Fragment Description: {fragment_info.get('description', 'N/A')}")
    output.append("\n--- User Assessment Data (Chunk) ---")
    for q_data in questions_chunk:
        output.append(f"\nQuestion ID: {q_data['question_id']} (Topic: {q_data.get('subcategory', 'N/A')})") # Include subcategory for theme context
        output.append(f"Question: {q_data['question_text']}")
        try: user_answer_str = json.dumps(q_data['user_answer'])
        except TypeError: user_answer_str = str(q_data['user_answer'])
        output.append(f"User Answer: {user_answer_str}")
        proc_fields = q_data.get('processed_additional_fields', {})
        if proc_fields:
            output.append("  Relevant Context/Metadata:")
            if 'guidelines' in proc_fields: output.append(f"    - Guideline: {proc_fields['guidelines']}")
            if 'sub_sub_category' in proc_fields: output.append(f"    - Sub-Topic: {proc_fields['sub_sub_category']}")
            try:
                if 'tools' in proc_fields: output.append(f"    - Available Tools Info: {json.dumps(proc_fields['tools'], indent=2)}")
                if 'recommendations' in proc_fields: output.append(f"    - Specific Recommendation Context: {json.dumps(proc_fields['recommendations'], indent=2)}")
                elif 'recommendation' in proc_fields: output.append(f"    - General Recommendation Context: {json.dumps(proc_fields['recommendation'], indent=2)}")
            except (TypeError, ValueError) as json_err: output.append(f"    - (Error formatting context field: {json_err})")
        output.append("-" * 10)
    return "\n".join(output)

def generate_insights_prompt(fragment_context: str) -> str:
    """
    Creates the prompt for generating Top 3 Insights (used for chunks).
    REFINED for specificity and no IDs.
    """
    prompt = f"""
Based on the following assessment data for a specific fragment chunk, identify up to 3 key insights regarding the user's current state, challenges, or strengths related to the topics covered in this chunk.

Assessment Data Chunk:
{fragment_context}

Instructions:
- Analyze the questions, user answers, and associated metadata within this chunk.
- Synthesize the information to derive meaningful insights specific to this chunk.
- **Insights must be specific, actionable, and avoid vague jargon.**
- **Do NOT refer to specific Question IDs.** Instead, refer to the themes or topics assessed (e.g., "Regarding version control practices...", "The assessment of monitoring tools indicates...").
- Present the insights as distinct points or paragraphs (up to 3).
- Focus on the most impactful observations based *only* on the provided chunk data.
- Ensure the output is formatted clearly, starting each insight on a new line.
- Do not include introductory phrases like "Here are the insights:".
- Do not use markdown formatting like '***', '---', or numbering like '1.'.

Top Insights for this Chunk:
[Insight 1 text]
[Insight 2 text]
[Insight 3 text]
"""
    return prompt

def generate_roadmap_prompt(fragment_context: str) -> str:
    """
    Creates the prompt for generating the Roadmap Summary as JSON (used for chunks).
    REFINED for 'how_methodology' format, specificity, and no IDs.
    """
    prompt = f"""
Based *only* on the following assessment data chunk for a specific fragment, generate a 3-phase implementation roadmap (Early, Intermediate, Advanced Steps) **strictly as a JSON object**.

Assessment Data Chunk:
{fragment_context}

Instructions:
- Analyze the questions, user answers, and available metadata *within this chunk*.
- Synthesize this information to create a coherent *partial* roadmap based *only* on this chunk's data.
- **Aggregate existing step-wise recommendations within this chunk**: Merge Step 1s into 'Early Steps', Step 2s into 'Intermediate Steps', etc.
- **Format `how_methodology`:** For each phase, create a single string containing a sequentially numbered list of specific, actionable steps (e.g., "1. Implement basic monitoring for critical services. 2. Define initial SLOs based on current performance. 3. Setup automated alerts for SLO breaches."). **Do NOT repeat 'Step X:' labels for each item.**
- **Estimate man-hours realistically** based on data *in this chunk*. Ensure 'total_hours' per phase is a numeric integer.
- **Consolidate roles and tools** mentioned *in this chunk* for each phase.
- **Content Specificity:** Ensure the `summary_context`, phase `description`s, and `how_methodology` steps are specific and avoid vague jargon.
- **No Question IDs:** Do NOT refer to specific Question IDs in any field. Refer to topics/themes if necessary.
- **Output ONLY the JSON object**, starting with `{{` and ending with `}}`. No extra text or markdown.

Required JSON Structure (for this partial roadmap):
{{
  "fragment_id": "[Fragment ID from Assessment Data]",
  "fragment_name": "[Fragment Name/Category from Assessment Data]",
  "summary_context": "[Specific ~1-2 sentence narrative overview *based on this chunk*]",
  "total_estimated_hours": "[Numeric integer sum of hours *from this chunk*]",
  "key_roles_involved": "[Consolidated list/string of unique roles *from this chunk*]",
  "phases": [
    {{
      "phase_name": "Early Steps",
      "step_title": "[Specific title, e.g., Foundational Monitoring Setup]",
      "description": "[Specific overview based on this chunk and Step 1s]",
      "how_methodology": "[Single string with numbered list of actions, e.g., '1. Action A. 2. Action B.']",
      "who_roles_involved": "[List/string of roles *from this chunk*]",
      "tools_platforms_used": "[List/string of tools *from this chunk*]",
      "estimate_per_subtask": "[Labelled subtasks/hours string *from this chunk*]",
      "total_hours": "[Numeric integer total hours for this phase *from this chunk*]"
    }},
    {{
      "phase_name": "Intermediate Steps",
      "step_title": "[Specific title, e.g., Process Integration & Automation]",
      "description": "[Specific goal based on this chunk and Step 2s]",
      "how_methodology": "[Single string with numbered list of actions, e.g., '1. Action C. 2. Action D.']",
      "who_roles_involved": "[List/string of roles *from this chunk*]",
      "tools_platforms_used": "[List/string of tools *from this chunk*]",
      "estimate_per_subtask": "[Labelled subtasks/hours string *from this chunk*]",
      "total_hours": "[Numeric integer total hours for this phase *from this chunk*]"
    }},
    {{
      "phase_name": "Advanced Steps",
      "step_title": "[Specific title, e.g., Performance Optimization & Governance]",
      "description": "[Specific plan based on this chunk and Step 3s]",
      "how_methodology": "[Single string with numbered list of actions, e.g., '1. Action E. 2. Action F.']",
      "who_roles_involved": "[List/string of roles *from this chunk*]",
      "tools_platforms_used": "[List/string of tools *from this chunk*]",
      "estimate_per_subtask": "[Labelled subtasks/hours string *from this chunk*]",
      "total_hours": "[Numeric integer total hours for this phase *from this chunk*]"
    }}
  ]
}}
"""
    return prompt

def generate_synthesize_insights_prompt(partial_insights_list: List[str]) -> str:
    """
    Creates the prompt for synthesizing partial insights into the final Top 3.
    REFINED for specificity and no IDs.
    """
    combined_insights_text = "\n\n---\n\n".join(
        f"Partial Insights Set {i+1}:\n{insight}"
        for i, insight in enumerate(partial_insights_list) if insight and insight.strip()
    )
    if not combined_insights_text: return ""

    prompt = f"""
Review the following sets of partial insights generated from analyzing different chunks of assessment data for the *same* overall fragment. Your task is to synthesize these partial observations into the definitive Top 3 key insights for the entire fragment.

Combined Partial Insights:
{combined_insights_text}

Instructions:
- Identify recurring themes, major challenges, and significant strengths across all partial insights.
- Consolidate overlapping points.
- Prioritize the most impactful and overarching observations for the final Top 3.
- **Final insights must be specific, actionable, and avoid vague jargon.**
- **Do NOT refer to specific Question IDs.** Instead, refer to the themes or topics assessed (e.g., "Overall, the approach to automated testing lacks...", "A key strength identified is the existing CI/CD pipeline...").
- Present the final insights as 3 distinct points or paragraphs.
- Focus on clarity and conciseness.
- Do not include introductory phrases like "Here are the final insights:". Just provide the insights directly.
- Do not use markdown formatting like '***', '---', or numbering like '1.'. Start each final insight on a new line.

Final Top 3 Insights:
[Final Insight 1 text]
[Final Insight 2 text]
[Final Insight 3 text]
"""
    return prompt

def generate_synthesize_roadmaps_prompt(partial_roadmaps_list: List[Dict[str, Any]], fragment_id: str, fragment_name: str) -> str:
    """
    Creates the prompt for synthesizing partial roadmaps into a final JSON roadmap.
    REFINED for 'how_methodology' format, specificity, and no IDs.
    """
    valid_partial_roadmaps = [r for r in partial_roadmaps_list if r and isinstance(r, dict) and r.get("phases")]
    if not valid_partial_roadmaps: return ""

    try:
        partial_roadmaps_json_str = json.dumps(valid_partial_roadmaps, indent=2) # Use indent for LLM readability
    except TypeError as e:
        print(f"Error serializing partial roadmaps for synthesis prompt: {e}")
        try: partial_roadmaps_json_str = json.dumps([str(r) for r in valid_partial_roadmaps], indent=2)
        except Exception as e2: print(f"Fallback serialization also failed: {e2}"); return ""

    prompt = f"""
You are given a list of partial implementation roadmaps (as a JSON array), each generated from analyzing a different chunk of assessment data for the *same* overall fragment (ID: {fragment_id}, Name: {fragment_name}). Your task is to synthesize these partial roadmaps into a single, coherent, final 3-phase implementation roadmap (Early, Intermediate, Advanced Steps) **strictly as a JSON object**.

List of Partial Roadmaps (JSON Array):
{partial_roadmaps_json_str}

Instructions for Synthesis and Final JSON Output:
1.  **Analyze All Partial Roadmaps:** Review the 'phases' within each partial roadmap.
2.  **Merge Phases:**
    * **Consolidate Activities & Goals:** For each final phase (Early, Intermediate, Advanced), combine related activities, goals, and methodologies from the corresponding partial phases. Identify logical progressions.
    * **Synthesize `how_methodology`:** Critically review the `how_methodology` strings from all partial roadmaps for the corresponding phase. **Merge these into a single, sequentially re-numbered list of specific, actionable steps within ONE string.** Ensure correct numbering (1., 2., 3., ...). Remove redundancy. **Crucially, do NOT repeat 'Step X:' labels.** Example format: "1. Define initial monitoring scope. 2. Install monitoring agents on critical servers. 3. Configure basic dashboards in Grafana."
    * **Combine `who_roles_involved` & `tools_platforms_used`:** Create consolidated, unique lists (as comma-separated strings or lists of strings).
    * **Aggregate `estimate_per_subtask`:** Combine subtask estimates. Sum hours for similar tasks or create representative combined entries. Present as a labelled string.
    * **Sum `total_hours`:** Calculate the total hours for each final phase by summing the `total_hours` from corresponding partial phases. Ensure numeric integer output.
    * **Create Overarching `step_title` and `description`:** Formulate concise, specific titles and descriptions for each final phase based on the merged content. Avoid jargon.
3.  **Generate Final Top-Level Fields:**
    * **`fragment_id`:** "{fragment_id}"
    * **`fragment_name`:** "{fragment_name}"
    * **`summary_context`:** Write a new, specific, brief (~2-3 sentence) narrative overview summarizing the overall state based on *all* partial roadmaps. Avoid jargon.
    * **`total_estimated_hours`:** Numeric integer sum of the final three phase `total_hours`.
    * **`key_roles_involved`:** Final consolidated list/string of unique roles across all phases.
4.  **Content Quality:** Ensure all generated text fields (`summary_context`, `description`, `step_title`, `how_methodology`) are specific, actionable, and avoid vague generalities.
5.  **No Question IDs:** **Do NOT refer to specific Question IDs** anywhere in the output JSON. Refer to underlying topics or themes if necessary (e.g., "addressing the gaps identified in automated testing processes").
6.  **Output Format:** Output **ONLY the final synthesized JSON object**, starting with `{{` and ending with `}}`. No extra text, markdown, or explanations.

Required Final JSON Structure:
{{
  "fragment_id": "{fragment_id}",
  "fragment_name": "{fragment_name}",
  "summary_context": "[Synthesized, specific ~2-3 sentence narrative overview]",
  "total_estimated_hours": "[Numeric integer sum of final phase hours]",
  "key_roles_involved": "[Final consolidated list/string of unique roles]",
  "phases": [
    {{
      "phase_name": "Early Steps",
      "step_title": "[Specific, synthesized title for Early Steps]",
      "description": "[Specific, synthesized description for Early Steps]",
      "how_methodology": "[Single string with merged, re-numbered list of specific actions, e.g., '1. Action A. 2. Action B. ...']",
      "who_roles_involved": "[Consolidated list/string of roles for Early Steps]",
      "tools_platforms_used": "[Consolidated list/string of tools for Early Steps]",
      "estimate_per_subtask": "[Consolidated/Aggregated subtask estimates string]",
      "total_hours": "[Numeric integer sum of hours for Early Steps]"
    }},
    {{
      "phase_name": "Intermediate Steps",
      "step_title": "[Specific, synthesized title for Intermediate Steps]",
      "description": "[Specific, synthesized description for Intermediate Steps]",
      "how_methodology": "[Single string with merged, re-numbered list of specific actions]",
      "who_roles_involved": "[Consolidated list/string of roles for Intermediate Steps]",
      "tools_platforms_used": "[Consolidated list/string of tools for Intermediate Steps]",
      "estimate_per_subtask": "[Consolidated/Aggregated subtask estimates string]",
      "total_hours": "[Numeric integer sum of hours for Intermediate Steps]"
    }},
    {{
      "phase_name": "Advanced Steps",
      "step_title": "[Specific, synthesized title for Advanced Steps]",
      "description": "[Specific, synthesized description for Advanced Steps]",
      "how_methodology": "[Single string with merged, re-numbered list of specific actions]",
      "who_roles_involved": "[Consolidated list/string of roles for Advanced Steps]",
      "tools_platforms_used": "[Consolidated list/string of tools for Advanced Steps]",
      "estimate_per_subtask": "[Consolidated/Aggregated subtask estimates string]",
      "total_hours": "[Numeric integer sum of hours for Advanced Steps]"
    }}
  ]
}}
"""
    return prompt

# --- Roadmap Parsing and Graphing (Unchanged from previous version) ---
def parse_roadmap_output_json(roadmap_json_text: str) -> Optional[Dict[str, Any]]:
    """
    Parses the roadmap JSON text (partial or synthesized).
    Returns a dictionary with roadmap details or None if parsing fails.
    Includes key renaming for PPT compatibility.
    """
    if not roadmap_json_text: print("Warning: Roadmap JSON text is empty."); return None
    cleaned_text = roadmap_json_text.strip()
    cleaned_text = re.sub(r'^```(?:json)?\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\s*```$', '', cleaned_text); cleaned_text = cleaned_text.strip()
    cleaned_text = re.sub(r',\s*([\}\]])', r'\1', cleaned_text) # Fix trailing commas

    try:
        if not cleaned_text.startswith("{") or not cleaned_text.endswith("}"):
            # print("Warning: Roadmap text doesn't start/end with {}. Searching for JSON object.") # Less verbose
            json_match = re.search(r'\{.*\S.*\}', cleaned_text, re.DOTALL)
            if json_match: cleaned_text = json_match.group(0)
            else: print("Error: Could not find valid JSON object structure in the text."); raise json.JSONDecodeError("Invalid JSON structure", cleaned_text, 0)

        roadmap_data = json.loads(cleaned_text)
        if not isinstance(roadmap_data, dict): print("Error: Parsed roadmap JSON is not a dictionary."); return None

        # Ensure 'phases' exists, even if empty initially
        if "phases" not in roadmap_data: roadmap_data["phases"] = []
        elif not isinstance(roadmap_data["phases"], list): print("Error: Parsed roadmap JSON 'phases' is not a list."); return None

        total_hours_calc = 0; valid_phases = []
        for i, phase in enumerate(roadmap_data.get("phases", [])):
            if not isinstance(phase, dict): print(f"Warning: Phase item at index {i} is not a dict. Skipping."); continue
            try: phase['total_hours'] = int(float(phase.get('total_hours', 0))); total_hours_calc += phase['total_hours']
            except (ValueError, TypeError) as e: print(f"Warning: Could not parse total_hours for phase {i}: {e}. Setting to 0."); phase['total_hours'] = 0

            # Rename keys for PPT compatibility
            phase["name"] = phase.pop("phase_name", f"Phase {i+1}")
            phase["title"] = phase.pop("step_title", "N/A")
            phase["how"] = phase.pop("how_methodology", "N/A") # Renamed from prompt
            phase["who"] = phase.pop("who_roles_involved", "N/A")
            if isinstance(phase["who"], list): phase["who"] = ", ".join(map(str, phase["who"]))
            phase["tools"] = phase.pop("tools_platforms_used", "N/A")
            if isinstance(phase["tools"], list): phase["tools"] = ", ".join(map(str, phase["tools"]))
            phase["subtasks"] = phase.pop("estimate_per_subtask", "N/A")
            valid_phases.append(phase)
        roadmap_data["phases"] = valid_phases

        # Validate/update top-level total hours
        try: roadmap_data['total_estimated_hours'] = int(float(roadmap_data.get('total_estimated_hours', 0)))
        except (ValueError, TypeError): roadmap_data['total_estimated_hours'] = total_hours_calc
        if roadmap_data.get('total_estimated_hours', 0) != total_hours_calc:
            if 'total_estimated_hours' in roadmap_data: print(f"Warning: Top-level hours ({roadmap_data.get('total_estimated_hours')}) differs from phase sum ({total_hours_calc}). Using sum.")
            roadmap_data['total_estimated_hours'] = total_hours_calc

        # Ensure top-level roles is string
        if "key_roles_involved" in roadmap_data and isinstance(roadmap_data["key_roles_involved"], list): roadmap_data["key_roles_involved"] = ", ".join(map(str, roadmap_data["key_roles_involved"]))
        elif "key_roles_involved" not in roadmap_data: roadmap_data["key_roles_involved"] = "N/A"
        return roadmap_data

    except json.JSONDecodeError as e: print(f"Error: Failed to decode roadmap JSON: {e}"); return None
    except Exception as e: print(f"Error processing parsed roadmap data: {e}"); traceback.print_exc(); return None

def create_roadmap_graph(roadmap_data: Dict[str, Any]) -> Optional[io.BytesIO]:
    """ Generates roadmap bar chart, returns in-memory buffer. """
    if not roadmap_data or not isinstance(roadmap_data.get("phases"), list) or not roadmap_data["phases"]: return None
    phases = roadmap_data["phases"]; phase_names = [p.get("name", f"P{i+1}") for i, p in enumerate(phases)]
    total_hours = [p.get("total_hours", 0) for p in phases]; total_days = [h / HOURS_PER_DAY for h in total_hours]
    if not any(d > 0 for d in total_days): return None
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(phase_names, total_days, color=['#4A90E2', '#F5A623', '#50E3C2', '#BD10E0', '#7ED321'][:len(phase_names)])
        ax.set_ylabel(f'Est. Duration (Days)', fontsize=9); ax.set_title(f'Phase Durations: {roadmap_data.get("fragment_name", "Fragment")}', fontsize=10)
        ax.set_xticks(range(len(phase_names))); ax.set_xticklabels(phase_names, rotation=0, ha='center', fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.yaxis.grid(True, linestyle='--', alpha=0.6); ax.tick_params(axis='y', labelsize=9)
        for bar in bars:
            yval = bar.get_height();
            if yval > 0: plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom', ha='center', fontsize=8)
        plt.tight_layout(pad=0.5); img_buffer = io.BytesIO(); plt.savefig(img_buffer, format='png', dpi=120); img_buffer.seek(0); plt.close(fig); return img_buffer
    except Exception as e: print(f"Error generating graph: {e}"); traceback.print_exc(); return None

# --- PPT Generation Functions (Unchanged from previous version) ---
def add_text_to_shape(shape, text):
    """Helper to add text to a shape, handling None values."""
    if shape and hasattr(shape, "text_frame"): tf = shape.text_frame; tf.text = str(text) if text is not None else ""; tf.word_wrap = True
    elif shape and hasattr(shape, "text"): shape.text = str(text) if text is not None else ""

def add_text_slide(prs: Presentation, title_text: str, content_text: str, layout_index: int = DEFAULT_LAYOUT_TITLE_CONTENT):
    """ Adds slide with title/content, handles overflow, uses safe placeholder access. """
    try: slide_layout = prs.slide_layouts[layout_index]
    except IndexError: print(f"Error: Layout index {layout_index} OOR. Using 0."); slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout); title_shape = None; content_shape = None; title_ph_idx = -1; content_ph_idx = -1
    for i, ph in enumerate(slide.placeholders):
         if 'Title' in ph.name or ph.placeholder_format.idx == 0 or ph.placeholder_format.idx == 10:
             if title_shape is None: title_shape = ph; title_ph_idx = i
         elif 'Content' in ph.name or 'Body' in ph.name or 'Text' in ph.name or ph.placeholder_format.idx == 1 or ph.placeholder_format.idx == 11:
              if content_shape is None: content_shape = ph; content_ph_idx = i
    if title_shape is None and len(slide.placeholders) > 0:
        try: title_shape = slide.placeholders[0]; title_ph_idx = 0; # print(f"Info: Using placeholder idx 0 as title fallback for L{layout_index}.") # Verbose
        except (KeyError, IndexError): pass
    if content_shape is None and len(slide.placeholders) > 1 and title_ph_idx != 1:
        try: content_shape = slide.placeholders[1]; content_ph_idx = 1; # print(f"Info: Using placeholder idx 1 as content fallback for L{layout_index}.") # Verbose
        except (KeyError, IndexError): pass

    if title_shape: add_text_to_shape(title_shape, title_text)
    else: print(f"Warning: Could not find title placeholder for '{title_text}' on layout {layout_index}.")
    if content_shape: tf = content_shape.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP; tf.margin_bottom = Inches(0.1); tf.margin_top = Inches(0.1); add_text_to_shape(tf, content_text)
    else:
        print(f"Warning: Could not find content placeholder for '{title_text}' on layout {layout_index}. Adding to notes.")
        try: notes_tf = slide.notes_slide.notes_text_frame; notes_tf.text = f"Title: {title_text}\n\nContent:\n{content_text}"
        except Exception as e_notes: print(f"Error adding content to notes slide: {e_notes}")

def build_ppt_report(session_id: str, report_data: Dict[str, Dict[str, Any]], template_path_relative: Optional[str] = "template.pptx") -> Optional[io.BytesIO]:
    """Builds the PowerPoint presentation in memory."""
    prs = None
    if template_path_relative:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            absolute_template_path = os.path.join(script_dir, template_path_relative)
            if os.path.exists(absolute_template_path): prs = Presentation(absolute_template_path); print(f"Loaded template: {absolute_template_path}")
            else: print(f"Warning: Template not found at {absolute_template_path}")
        except NameError:
             if os.path.exists(template_path_relative):
                 try: prs = Presentation(template_path_relative); print(f"Loaded template from CWD: {template_path_relative}")
                 except Exception as e_cwd: print(f"Error loading template from CWD: {e_cwd}")
             else: print(f"Warning: Template not found in CWD.")
        except Exception as e: print(f"Error loading template '{template_path_relative}': {e}")
    if prs is None: print("Using default pptx template."); prs = Presentation()

    # Slide 1: Title
    try:
        title_layout = prs.slide_layouts[0]; slide = prs.slides.add_slide(title_layout); title_shape = None; subtitle_shape = None
        try: title_shape = slide.shapes.title
        except AttributeError: 
            try: title_shape = slide.placeholders[0]
            except (KeyError, IndexError): pass
        try: subtitle_shape = slide.placeholders[1]
        except (KeyError, IndexError): pass
        if title_shape: add_text_to_shape(title_shape, "Recommendation Report")
        else: print("Warning: Title slide missing title placeholder.")
        if subtitle_shape: add_text_to_shape(subtitle_shape, f"Session ID: {session_id}\nGenerated: {time.strftime('%Y-%m-%d')}")
        else: print("Warning: Title slide missing subtitle placeholder.")
    except Exception as e: print(f"Error creating title slide: {e}")

    # Process fragments
    sorted_fragment_ids = sorted(report_data.keys())
    for fragment_id in sorted_fragment_ids:
        data = report_data[fragment_id]; frag_info = data.get("fragment_info", {}); frag_name = frag_info.get('category') or frag_info.get('description') or fragment_id or "Unknown"; frag_name = str(frag_name).strip()
        print(f"Generating slides for Fragment: {fragment_id} ({frag_name})...")
        # Slide: Fragment Title
        try:
            header_layout = prs.slide_layouts[DEFAULT_LAYOUT_SECTION_HEADER]; slide = prs.slides.add_slide(header_layout); title_shape = None
            try: title_shape = slide.shapes.title
            except AttributeError: 
                try: title_shape = slide.placeholders[0]
                except (KeyError, IndexError): pass
            if title_shape: add_text_to_shape(title_shape, f"Fragment: {frag_name}")
            else: print(f"Warning: Layout {DEFAULT_LAYOUT_SECTION_HEADER} missing title.")
        except IndexError: print(f"Error: Layout {DEFAULT_LAYOUT_SECTION_HEADER} OOR.")
        except Exception as e: print(f"Error creating fragment title slide: {e}")

        # Slide: Insights
        insights = data.get("insights", "Insights could not be generated."); insights_title = f"{frag_name}: Top 3 Insights"; insights_cleaned = insights
        if isinstance(insights, str): insights_cleaned = re.sub(r'^\s*(\d+\.|-|\*)\s*', '', insights, flags=re.MULTILINE).strip().replace('***', '').replace('---', '')
        add_text_slide(prs, insights_title, insights_cleaned if insights_cleaned else "Insights empty/failed.")

        # Slides: Roadmap
        roadmap = data.get("roadmap"); roadmap_base_title = f"{frag_name}: Roadmap"
        is_valid_roadmap = isinstance(roadmap, dict) and "error" not in roadmap
        if is_valid_roadmap and roadmap.get("phases"):
            try: # Overview Slide
                overview_title = f"{frag_name}: Roadmap Overview"; total_hrs = roadmap.get('total_estimated_hours', 0)
                overview_text = f"Summary: {roadmap.get('summary_context', 'N/A')}\n\nTotal Est: {total_hrs} hrs (~{total_hrs/HOURS_PER_DAY:.1f} days)\n\nKey Roles: {roadmap.get('key_roles_involved', 'N/A')}"
                add_text_slide(prs, overview_title, overview_text, DEFAULT_LAYOUT_TITLE_CONTENT)
            except Exception as e: print(f"Error creating roadmap overview slide: {e}")
            for i, phase in enumerate(roadmap.get("phases", [])): # Detail Slides
                if not isinstance(phase, dict): continue
                phase_title = f"{frag_name}: {phase.get('name', f'Phase {i+1}')}"; phase_hrs = phase.get('total_hours', 0)
                phase_content = (f"Goal: {phase.get('title', 'N/A')}\n\nDesc: {phase.get('description', 'N/A')}\n\nMethodology:\n{phase.get('how', 'N/A')}\n\n"
                                 f"Roles: {phase.get('who', 'N/A')}\nTools: {phase.get('tools', 'N/A')}\n\nSubtasks Est: {phase.get('subtasks', 'N/A')}\n\n"
                                 f"Duration: {phase_hrs} hrs (~{phase_hrs/HOURS_PER_DAY:.1f} days)")
                add_text_slide(prs, phase_title, phase_content, DEFAULT_LAYOUT_TITLE_CONTENT)
        else: # Roadmap failed slide
            roadmap_error_msg = "Roadmap could not be generated or synthesized.";
            if isinstance(roadmap, dict) and "error" in roadmap: roadmap_error_msg += f"\n\nDetails: {roadmap.get('error')}"
            add_text_slide(prs, roadmap_base_title, roadmap_error_msg)

        # Slide: Graph
        graph_buffer = data.get("graph_buffer"); graph_title = f"{frag_name}: Roadmap Duration Overview"
        if graph_buffer:
            try:
                graph_layout = None
                try: graph_layout = prs.slide_layouts[DEFAULT_LAYOUT_TITLE_ONLY]
                except IndexError: 
                    try: graph_layout = prs.slide_layouts[DEFAULT_LAYOUT_TITLE_CONTENT]
                    except IndexError: print(f"Error: Layouts {DEFAULT_LAYOUT_TITLE_ONLY}/{DEFAULT_LAYOUT_TITLE_CONTENT} missing."); graph_layout = None
                if graph_layout:
                    slide = prs.slides.add_slide(graph_layout); title_shape_g = None
                    try: title_shape_g = slide.shapes.title
                    except AttributeError: 
                        try: title_shape_g = slide.placeholders[0]
                        except (KeyError, IndexError): pass
                    if title_shape_g: add_text_to_shape(title_shape_g, graph_title)
                    else: print(f"Warning: Graph slide missing title placeholder.")
                    left, top, height = Inches(0.5), Inches(1.5), Inches(5.5) # Adjust as needed
                    try: pic = slide.shapes.add_picture(graph_buffer, left, top, height=height)
                    except Exception as pic_err: print(f"Error adding picture to graph slide: {pic_err}")
                else: print("Skipping graph slide - suitable layout not found.")
            except Exception as e: print(f"Error adding graph slide: {e}"); traceback.print_exc()
            finally:
                 if graph_buffer and not graph_buffer.closed: graph_buffer.close()
        elif is_valid_roadmap and roadmap.get("phases"): add_text_slide(prs, graph_title, "Graph could not be generated.")

    # Save to buffer
    try: ppt_buffer = io.BytesIO(); prs.save(ppt_buffer); ppt_buffer.seek(0); print("\nPPTX saved to memory buffer."); return ppt_buffer
    except Exception as e: print(f"\nError saving presentation to buffer: {e}"); traceback.print_exc(); return None

def upload_report_to_supabase(session_id: str, ppt_buffer: io.BytesIO, bucket_name: str = "recommendations") -> bool:
    """Uploads the generated PPTX buffer's content (bytes) to Supabase Storage."""
    if not ppt_buffer: print("Upload failed: No PPT buffer provided."); return False
    remote_path = f"{session_id}/recommendation_report.pptx"
    try:
        print(f"Attempting upload: bucket='{bucket_name}', path='{remote_path}'")
        supabase_client = get_supabase_client(); ppt_buffer.seek(0); ppt_bytes = ppt_buffer.read()
        upload_response = supabase_client.storage.from_(bucket_name).upload(
            path=remote_path, file=ppt_bytes,
            file_options={"content-type": "application/vnd.openxmlformats-officedocument.presentationml.presentation", "upsert": "true"}
        )
        print(f"Supabase upload finished for {remote_path}. Check dashboard.")
        return True
    except Exception as e: print(f"Error uploading to Supabase: {e}"); traceback.print_exc(); return False
    finally:
        if ppt_buffer and not ppt_buffer.closed: ppt_buffer.close()

# --- Main Generation Function (Refactored for Chunking & Synthesis) ---
def generate_recommendations(session_id: str) -> Dict[str, Dict[str, Any]]:
    """Generates recommendations using context-aware chunking and synthesis."""
    print(f"Starting recommendation generation for session_id: {session_id}")
    supabase_client = get_supabase_client()
    print("Fetching data...")
    followup_resp = fetch_followup_data(supabase_client, session_id)
    if not followup_resp: print("No followup responses found."); return {}
    q_ids = list(set(r['question_id'] for r in followup_resp if r.get('question_id')))
    q_details = fetch_question_details(supabase_client, q_ids)
    if not q_details: print("No question details found."); return {}

    print("Processing and grouping data...")
    grouped_data = group_data_by_fragment(followup_resp, q_details)
    if not grouped_data: print("No data grouped by fragment."); return {}

    final_report_data = {}
    sorted_fragment_ids = sorted(grouped_data.keys())
    token_counter_model = None
    try: token_counter_model = genai.GenerativeModel(LLM_MODEL_NAME); print("Token counter model initialized.")
    except Exception as e: print(f"Warning: Could not initialize token counter model: {e}.")

    for fragment_id in sorted_fragment_ids:
        fragment_data = grouped_data[fragment_id]; frag_info = fragment_data.get("fragment_info", {})
        frag_name = frag_info.get('category') or frag_info.get('description') or fragment_id or "Unknown"; frag_name = str(frag_name).strip()
        print(f"\n--- Processing Fragment: {fragment_id} ({frag_name}) ---")
        all_questions = fragment_data.get("questions_data", [])
        if not all_questions:
            print("No questions found. Skipping fragment."); final_report_data[fragment_id] = {"fragment_info": frag_info, "insights": "No questions.", "roadmap": None, "graph_buffer": None}; continue

        final_report_data[fragment_id] = {"fragment_info": frag_info}; partial_insights_list = []; partial_roadmaps_list = []; current_chunk_questions = []; current_chunk_tokens = 0; chunk_num = 1

        # --- Context-Aware Chunking Loop ---
        for i, question_data in enumerate(all_questions):
            temp_chunk_for_estimation = current_chunk_questions + [question_data]; potential_chunk_str = format_chunk_for_prompt(frag_info, temp_chunk_for_estimation); estimated_next_chunk_tokens = 0
            try:
                if token_counter_model: estimated_next_chunk_tokens = token_counter_model.count_tokens(potential_chunk_str).total_tokens
                else: estimated_next_chunk_tokens = len(potential_chunk_str) // 4
            except Exception as e: print(f"Warning: Token counting failed during chunking ({e}). Using fallback."); estimated_next_chunk_tokens = len(potential_chunk_str) // 4

            is_last_question = (i == len(all_questions) - 1)
            # Process chunk if limit reached (and it's not the last question, as that's handled after loop)
            if current_chunk_questions and estimated_next_chunk_tokens > MAX_CHUNK_TOKENS and not is_last_question:
                print(f"\nChunk {chunk_num}: Token limit ({MAX_CHUNK_TOKENS}) reached. Processing {len(current_chunk_questions)} questions.")
                chunk_context_str = format_chunk_for_prompt(frag_info, current_chunk_questions)
                # Generate Partial Insights & Roadmap
                insights_prompt = generate_insights_prompt(chunk_context_str); insights_result = call_llm(insights_prompt, max_tokens=800, temperature=0.5)
                if insights_result: partial_insights_list.append(insights_result); print(f"  Chunk {chunk_num} Insights: Generated.")
                else: print(f"  Chunk {chunk_num} Insights: FAILED.")
                roadmap_prompt = generate_roadmap_prompt(chunk_context_str); roadmap_txt = call_llm(roadmap_prompt, max_tokens=3500, temperature=0.3)
                if roadmap_txt:
                    roadmap_parsed = parse_roadmap_output_json(roadmap_txt)
                    if roadmap_parsed: partial_roadmaps_list.append(roadmap_parsed); print(f"  Chunk {chunk_num} Roadmap: Parsed.")
                    else: print(f"  Chunk {chunk_num} Roadmap: Parse FAILED.")
                else: print(f"  Chunk {chunk_num} Roadmap: Generation FAILED.")
                # Reset for next chunk
                current_chunk_questions = [question_data]; chunk_num += 1
                new_chunk_str = format_chunk_for_prompt(frag_info, current_chunk_questions)
                try:
                     if token_counter_model: current_chunk_tokens = token_counter_model.count_tokens(new_chunk_str).total_tokens
                     else: current_chunk_tokens = len(new_chunk_str) // 4
                except Exception as e: print(f"Warning: Token counting failed starting chunk {chunk_num} ({e})."); current_chunk_tokens = len(new_chunk_str) // 4
            else:
                # Add question to chunk
                current_chunk_questions.append(question_data); current_chunk_tokens = estimated_next_chunk_tokens
                # Process the very last chunk after the loop finishes
                if is_last_question:
                     print(f"\nChunk {chunk_num} (Final): Processing {len(current_chunk_questions)} questions.")
                     chunk_context_str = format_chunk_for_prompt(frag_info, current_chunk_questions)
                     insights_prompt = generate_insights_prompt(chunk_context_str); insights_result = call_llm(insights_prompt, max_tokens=800, temperature=0.5)
                     if insights_result: partial_insights_list.append(insights_result); print(f"  Chunk {chunk_num} Insights: Generated.")
                     else: print(f"  Chunk {chunk_num} Insights: FAILED.")
                     roadmap_prompt = generate_roadmap_prompt(chunk_context_str); roadmap_txt = call_llm(roadmap_prompt, max_tokens=3500, temperature=0.3)
                     if roadmap_txt:
                         roadmap_parsed = parse_roadmap_output_json(roadmap_txt)
                         if roadmap_parsed: partial_roadmaps_list.append(roadmap_parsed); print(f"  Chunk {chunk_num} Roadmap: Parsed.")
                         else: print(f"  Chunk {chunk_num} Roadmap: Parse FAILED.")
                     else: print(f"  Chunk {chunk_num} Roadmap: Generation FAILED.")
        # --- End Chunking Loop ---

        # --- Synthesis Step ---
        print("\n--- Synthesizing Results for Fragment ---")
        # Synthesize Insights
        final_insights = "Insights synthesis skipped (no partial insights generated)."
        if partial_insights_list:
            print("Synthesizing insights..."); synthesis_insights_prompt = generate_synthesize_insights_prompt(partial_insights_list)
            if synthesis_insights_prompt:
                 final_insights_result = call_llm(synthesis_insights_prompt, max_tokens=1000, temperature=0.6)
                 if final_insights_result: final_insights = final_insights_result; print("Insights synthesized successfully.")
                 else: final_insights = "Insights synthesis failed."; print("Insights synthesis failed.")
            else: final_insights = "Insights synthesis skipped."; print("Insights synthesis skipped.")
        final_report_data[fragment_id]["insights"] = final_insights

        # Synthesize Roadmaps
        final_roadmap = None
        if partial_roadmaps_list:
            print("Synthesizing roadmap..."); synthesis_roadmaps_prompt = generate_synthesize_roadmaps_prompt(partial_roadmaps_list, fragment_id, frag_name)
            if synthesis_roadmaps_prompt:
                 final_roadmap_txt = call_llm(synthesis_roadmaps_prompt, max_tokens=4096, temperature=0.2)
                 if final_roadmap_txt:
                     print("Parsing final synthesized roadmap JSON..."); final_roadmap_parsed = parse_roadmap_output_json(final_roadmap_txt)
                     if final_roadmap_parsed:
                         if "fragment_name" not in final_roadmap_parsed or not final_roadmap_parsed["fragment_name"]: final_roadmap_parsed["fragment_name"] = frag_name
                         final_roadmap = final_roadmap_parsed; print("Final roadmap parsed successfully.")
                     else: print("Failed to parse synthesized roadmap JSON."); final_roadmap = {"error": "Failed to parse synthesized roadmap JSON", "raw_output": final_roadmap_txt[:1000]+"..."}
                 else: print("Failed to generate synthesized roadmap JSON text."); final_roadmap = {"error": "Failed to generate synthesized roadmap JSON text"}
            else: print("Roadmap synthesis skipped (prompt generation failed)."); final_roadmap = {"error": "Roadmap synthesis prompt generation failed"}
        else: print("Roadmap synthesis skipped (no partial roadmaps generated)."); final_roadmap = {"error": "No partial roadmaps generated"}
        final_report_data[fragment_id]["roadmap"] = final_roadmap

        # Generate Graph
        print("Generating graph..."); graph_buffer = None
        if isinstance(final_roadmap, dict) and "error" not in final_roadmap: graph_buffer = create_roadmap_graph(final_roadmap)
        final_report_data[fragment_id]["graph_buffer"] = graph_buffer
        # --- End Fragment Processing ---

    print("\nRecommendation generation process completed.")
    return final_report_data


# --- Main Orchestration Function (Unchanged) ---
def generate_and_upload_recommendations(session_id: str, template_path_relative: Optional[str] = "template.pptx") -> bool:
     """Generates recommendations, builds PPT, and uploads."""
     print(f"Starting generation and upload for session {session_id}...")
     try:
         report_content = generate_recommendations(session_id)
         if not report_content: print(f"Failed to generate content for session {session_id}."); return False
         print(f"Building PPT report for session {session_id}...")
         ppt_buffer = build_ppt_report(session_id, report_content, template_path_relative)
         if not ppt_buffer:
             print(f"Failed to build PPT buffer for session {session_id}.")
             # Cleanup graph buffers
             for frag_id in report_content:
                 if report_content[frag_id].get('graph_buffer'):
                     try: report_content[frag_id]['graph_buffer'].close()
                     except Exception: pass
             return False
         print(f"Uploading report for session {session_id}...")
         upload_success = upload_report_to_supabase(session_id, ppt_buffer)
         if upload_success: print(f"Successfully generated and uploaded report for session {session_id}.")
         else: print(f"Upload failed for session {session_id}.")
         return upload_success
     except Exception as e: print(f"Error during generate_and_upload for session {session_id}: {e}"); traceback.print_exc(); return False


# --- Main Execution Block (Hardcoded Local Save) ---
if __name__ == "__main__":
    # --- Hardcoded Settings ---
    session_id_to_process = "1e195af3-b327-483a-ab8f-8c5f3940705c" # <-- EDIT THIS LINE
    template_file_name = "template.pptx"
    hardcoded_output_filename = "Generated_Recommendation_Report.pptx"
    should_upload = False # Set True to also upload
    # ---

    parser = argparse.ArgumentParser(description="Generate recommendations PPT from session_id.")
    parser.add_argument("-s", "--session", type=str, default=session_id_to_process, help="Session ID to process.")
    parser.add_argument("-t", "--template", type=str, default=template_file_name, help="Relative path to PPTX template.")
    parser.add_argument("--upload", action='store_true', default=should_upload, help="Upload report to Supabase.")

    args = parser.parse_args()
    session_id_to_process = args.session
    template_file_name = args.template
    should_upload = args.upload

    if not session_id_to_process or session_id_to_process == "YOUR_SESSION_ID_HERE": print("Error: Provide valid session ID via --session or edit script."); sys.exit(1)

    print(f"Processing session_id: {session_id_to_process}")
    print(f"Using template: {template_file_name}")
    print(f"Hardcoded local output: {hardcoded_output_filename}")
    print(f"Upload enabled: {should_upload}")

    report_content = generate_recommendations(session_id_to_process)

    if report_content:
        print("\nBuilding PowerPoint presentation...")
        ppt_buffer = build_ppt_report(session_id_to_process, report_content, template_path_relative=template_file_name)
        if ppt_buffer:
            print("PPT Buffer created.")
            try: # Save locally
                with open(hardcoded_output_filename, 'wb') as f: f.write(ppt_buffer.getvalue())
                print(f"Presentation saved locally: {hardcoded_output_filename}")
            except Exception as e: print(f"\nError saving locally to '{hardcoded_output_filename}': {e}")
            if should_upload: # Upload if requested
                 print("\nUploading report to Supabase...")
                 ppt_buffer.seek(0); upload_success = upload_report_to_supabase(session_id_to_process, ppt_buffer)
                 print("Upload successful." if upload_success else "Upload failed.")
            elif not ppt_buffer.closed: ppt_buffer.close() # Close buffer if not uploaded
        else:
            print("\nFailed to build PPT buffer.")
            # Cleanup graph buffers
            for frag_id in report_content:
                 if report_content[frag_id].get('graph_buffer'):
                     try: report_content[frag_id]['graph_buffer'].close()
                     except Exception: pass
    else: print("\nNo report data generated. PPT creation skipped.")
    print("\nScript finished.")

