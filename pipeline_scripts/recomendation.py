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
MAX_RPM = 30 # As specified
# Add a buffer to be safe (e.g., aim for slightly fewer calls)
TARGET_RPM = MAX_RPM - 3 # Aim for 27 RPM
RPM_WINDOW_SECONDS = 60
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2

def call_llm(prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.4) -> Optional[str]:
    """
    Calls the configured Gemini model with rate limiting (RPM) and retry logic.
    """
    global LLM_CALL_TIMESTAMPS # Use the global deque

    current_time = time.monotonic()

    # --- RPM Rate Limiting ---
    # Remove timestamps older than the window
    while LLM_CALL_TIMESTAMPS and LLM_CALL_TIMESTAMPS[0] <= current_time - RPM_WINDOW_SECONDS:
        LLM_CALL_TIMESTAMPS.popleft()

    # Check if limit reached
    if len(LLM_CALL_TIMESTAMPS) >= TARGET_RPM:
        oldest_call_in_window = LLM_CALL_TIMESTAMPS[0]
        wait_time = (oldest_call_in_window + RPM_WINDOW_SECONDS) - current_time
        wait_time = max(0, wait_time) # Ensure non-negative wait
        print(f"RPM limit approaching ({len(LLM_CALL_TIMESTAMPS)}/{TARGET_RPM}). Waiting for {wait_time:.2f} seconds...")
        time.sleep(wait_time + 0.1) # Add small buffer to sleep time

        # Recalculate current time and prune again after sleeping
        current_time = time.monotonic()
        while LLM_CALL_TIMESTAMPS and LLM_CALL_TIMESTAMPS[0] <= current_time - RPM_WINDOW_SECONDS:
            LLM_CALL_TIMESTAMPS.popleft()

    # Record the upcoming call attempt time BEFORE the call
    LLM_CALL_TIMESTAMPS.append(current_time)

    # --- API Call with Retry Logic ---
    backoff_time = INITIAL_BACKOFF_SECONDS
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Calling LLM...") # Add attempt info
            llm = genai.GenerativeModel(LLM_MODEL_NAME)
            gen_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens if max_tokens is not None else 1024
            )
            response = llm.generate_content(prompt, generation_config=gen_config)

            # Check response validity (same as before)
            if response.parts:
                # --- Token Usage (Optional Logging) ---
                # Note: Actual token count requires specific API access or tokenizer
                # This is a very rough estimate based on characters
                estimated_input_tokens = len(prompt) / 4
                estimated_output_tokens = len(response.text) / 4
                print(f"LLM Call Success. Estimated Tokens: In={estimated_input_tokens:.0f}, Out={estimated_output_tokens:.0f}")
                return response.text.strip()
            elif response.prompt_feedback.block_reason:
                print(f"Warning: LLM call blocked. Reason: {response.prompt_feedback.block_reason}")
                return None # Don't retry safety blocks
            else:
                print("Warning: LLM returned empty response.")
                # Consider retrying empty responses? For now, return None.
                return None

        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"Error: ResourceExhausted (likely rate limit) on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Waiting {backoff_time:.2f} seconds before retry...")
                time.sleep(backoff_time)
                backoff_time *= 2 # Exponential backoff
                # No need to manage LLM_CALL_TIMESTAMPS here, as the call didn't technically "complete" successfully for RPM counting
            else:
                print("Max retries reached after rate limit error.")
                return None # Give up after max retries
        except Exception as e:
             # Handle other potential errors (like API key, network issues)
             print(f"Error during LLM call on attempt {attempt + 1}: {e}")
             if "API key not valid" in str(e):
                 print("Critical: Invalid Google API Key. Aborting retry.")
                 return None # Don't retry auth errors
             # Add more specific non-retryable errors if needed

             # For potentially transient errors, apply backoff and retry
             if attempt < MAX_RETRIES - 1:
                  print(f"Waiting {backoff_time:.2f} seconds before retry...")
                  time.sleep(backoff_time)
                  backoff_time *= 2
             else:
                  print("Max retries reached after general error.")
                  return None # Give up after max retries

    # Should not be reached if loop finishes correctly, but safety return
    return None

HOURS_PER_DAY = 8 # Define conversion rate for graph
DEFAULT_LAYOUT_TITLE_CONTENT = 1 # Standard index for "Title and Content"
DEFAULT_LAYOUT_SECTION_HEADER = 2 # Standard index for "Section Header"
DEFAULT_LAYOUT_TITLE_ONLY = 5 # Standard index for "Title Only"
DEFAULT_LAYOUT_BLANK = 6 # Standard index for "Blank"

# --- Database Interaction ---

def get_supabase_client() -> Client:
    """Returns the initialized Supabase client."""
    # Client is already initialized globally in this version
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
        return []

def fetch_question_details(supabase_client: Client, question_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Fetches question details for a list of question IDs."""
    if not question_ids:
        return {}
    try:
        response = supabase_client.table("questions").select("id, category, subcategory, sr_no, fragments, question, additional_fields, text_representation").in_("id", question_ids).execute()
        if response.data:
            return {q['id']: q for q in response.data}
        else:
            print(f"No questions found for IDs: {question_ids}")
            return {}
    except Exception as e:
        print(f"Error fetching question details: {e}")
        return {}

def fetch_fragment_info(supabase_client: Client, fragment_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetches fragment descriptions for a list of CLEANED fragment IDs."""
    if not fragment_ids:
        return {}
    try:
        # Ensure fragment_ids are unique strings and not empty
        unique_fragment_ids = list(set(filter(None, fragment_ids)))
        if not unique_fragment_ids:
             print("Warning: No valid unique fragment IDs provided to fetch_fragment_info.")
             return {}
        print(f"Fetching fragment info for IDs: {unique_fragment_ids}") # Debug print
        response = supabase_client.table("fragment_info").select("fragment, description, category").in_("fragment", unique_fragment_ids).execute()
        if response.data:
            print(f"Successfully fetched info for fragments: {[f['fragment'] for f in response.data]}") # Debug print
            return {f['fragment']: f for f in response.data}
        else:
            # This warning is now expected if the fragment IDs weren't cleaned properly before calling
            print(f"Warning: No fragment info found matching fragments: {unique_fragment_ids} in the database.")
            return {}
    except Exception as e:
        print(f"Error fetching fragment info: {e}")
        return {}

# --- Data Processing ---

def process_additional_fields(question: Dict[str, Any], answer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters additional_fields based on the answer provided, according to the rules.
    Returns the filtered additional_fields dictionary.
    Uses CORRECTED LOGIC.
    """
    additional_fields = question.get("additional_fields")
    if not additional_fields:
        return {}

    # Ensure additional_fields is a dictionary
    if isinstance(additional_fields, str):
        try:
            additional_fields = json.loads(additional_fields)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse additional_fields JSON for question ID {question.get('id')}")
            return {}
    if not isinstance(additional_fields, dict):
         print(f"Warning: additional_fields is not a dictionary for question ID {question.get('id')}. Type: {type(additional_fields)}")
         return {}

    filtered_fields = {}
    user_answer = answer_data.get("value") if isinstance(answer_data, dict) else None

    # Always try to include these base fields if they exist initially.
    always_include_keys = ["guidelines", "answer_options", "sub_sub_category", "subjective_answer", "multiple_correct_answer_options", "why_use_these_tools"]
    for key in always_include_keys:
        if key in additional_fields:
            filtered_fields[key] = additional_fields[key]

    retrieve_all_fields = False # Flag to simplify logic
    processed_specifically = False # Flag to track if a specific condition was met

    # --- Apply Conditional Logic --- Order matters! ---

    # Condition a) Implemented
    if user_answer == "Implemented":
        processed_specifically = True
        # Ensure recommendation(s) and tools are NOT included
        filtered_fields.pop("recommendation", None)
        filtered_fields.pop("recommendations", None)
        filtered_fields.pop("tools", None)

    # Condition a) Not Implemented
    elif user_answer == "Not Implemented":
        retrieve_all_fields = True
        processed_specifically = True

    # Condition d) Subjective or "don't know" -> Retrieve all
    elif user_answer == "dont know" or \
         "subjective_answer" in additional_fields or \
         "multiple_correct_answer_options" in additional_fields:
         retrieve_all_fields = True
         processed_specifically = True

    # Condition b) Specific option with corresponding recommendation dictionary
    elif "recommendations" in additional_fields and isinstance(additional_fields["recommendations"], dict):
        matched_recommendation = None
        normalized_answer = user_answer.strip().lower() if user_answer else ""
        match_found_in_b = False

        for rec_key, rec_value in additional_fields["recommendations"].items():
             normalized_key = rec_key.strip().lower()
             key_base = re.sub(r'\s*\(level\s*\d+\)$', '', normalized_key).strip()
             if normalized_answer == normalized_key or \
                (key_base and normalized_answer == key_base) or \
                (key_base and normalized_answer.startswith(key_base)):
                 matched_recommendation = {rec_key: rec_value}
                 match_found_in_b = True
                 break

        if match_found_in_b:
            processed_specifically = True
            filtered_fields["recommendations"] = matched_recommendation
            if "tools" in additional_fields:
                 filtered_fields["tools"] = additional_fields["tools"]
        else:
            # No specific match within recommendations dict -> retrieve all fields.
            print(f"Info: Answer '{user_answer}' did not match specific key in 'recommendations' for QID {question.get('id')}. Retrieving all fields.")
            retrieve_all_fields = True
            # processed_specifically remains False here

    # Default Case: If none of the specific conditions above were met, retrieve all.
    if not processed_specifically:
         retrieve_all_fields = True


    # --- Final Step: Add all fields if flagged ---
    if retrieve_all_fields:
        # Update ensures base fields are kept, and all others are added.
        temp_copy = additional_fields.copy()
        temp_copy.update(filtered_fields) # Preserve already added base fields if keys conflict
        filtered_fields = temp_copy


    # Final cleanup
    return {k: v for k, v in filtered_fields.items() if v is not None}


def group_data_by_fragment(
    followup_responses: List[Dict[str, Any]],
    questions_details: Dict[int, Dict[str, Any]],
    # fragment_info removed as argument, fetched inside using cleaned IDs
) -> Dict[str, Dict[str, Any]]:
    """Groups processed question data by fragment, using cleaned fragment IDs."""
    grouped_data = defaultdict(lambda: {"fragment_info": None, "questions_data": []})
    processed_fragments = set() # Store cleaned fragment IDs encountered

    for resp in followup_responses:
        q_id = resp.get("question_id")
        question_detail = questions_details.get(q_id)

        if not question_detail:
            print(f"Warning: Details not found for question_id {q_id}. Skipping.")
            continue

        # Extract fragment ID (handle potential list format AND CLEAN IT)
        fragment_list = question_detail.get("fragments")
        raw_fragment_id = None
        cleaned_fragment_id = None

        if isinstance(fragment_list, (list, tuple)) and fragment_list:
            raw_fragment_id = str(fragment_list[0]).strip()
        elif isinstance(fragment_list, str):
             raw_fragment_id = fragment_list.strip()
        else:
            print(f"Warning: Invalid or missing fragment format for question_id {q_id}: {fragment_list}. Skipping.")
            continue

        # --- Clean the extracted fragment ID ---
        if raw_fragment_id:
            # Remove common list/string representations like ["..."] or '["..."]'
            # This regex attempts to capture the content inside optional brackets and quotes
            match = re.match(r'^[\[\'\"]*(.*?)[\'\"\]]*$', raw_fragment_id)
            if match:
                cleaned_fragment_id = match.group(1)
            else: # Fallback if regex fails unexpectedly
                 cleaned_fragment_id = raw_fragment_id # Use raw if regex didn't match

        if not cleaned_fragment_id:
             print(f"Warning: Empty or invalid fragment ID after cleaning for question_id {q_id} (Original: {raw_fragment_id}). Skipping.")
             continue

        processed_fragments.add(cleaned_fragment_id) # Add the CLEANED ID

        # Process additional fields based on the answer
        answer_json = resp.get("answer")
        if isinstance(answer_json, str):
            try:
                answer_json = json.loads(answer_json)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse answer JSON for response ID {resp.get('id')}. Treating as empty.")
                answer_json = {}
        elif not isinstance(answer_json, dict):
             print(f"Warning: Answer format is not a dict or parsable string for response ID {resp.get('id')}. Type: {type(answer_json)}. Treating as empty.")
             answer_json = {}

        processed_add_fields = process_additional_fields(question_detail, answer_json)

        question_data = {
            "question_id": q_id,
            "question_text": question_detail.get("question"),
            "category": question_detail.get("category"),
            "subcategory": question_detail.get("subcategory"),
            "user_answer": answer_json,
            "processed_additional_fields": processed_add_fields,
            "raw_additional_fields": question_detail.get("additional_fields")
        }
        # Use the CLEANED fragment ID as the key for grouping
        grouped_data[cleaned_fragment_id]["questions_data"].append(question_data)

    # Fetch and add fragment info using CLEANED IDs stored in processed_fragments
    if processed_fragments:
         all_fragment_info = fetch_fragment_info(get_supabase_client(), list(processed_fragments))
         for frag_id, data in grouped_data.items(): # frag_id here is already the cleaned one
             data["fragment_info"] = all_fragment_info.get(frag_id, {"fragment": frag_id, "description": "N/A", "category": "N/A"})
    else:
         print("No valid fragments found to fetch info for.")


    return dict(grouped_data)

# --- LLM Interaction & Content Generation ---

def format_data_for_prompt(fragment_data: Dict[str, Any]) -> str:
    """Formats the grouped data for a specific fragment into a string for LLM prompts."""
    output = []
    frag_info = fragment_data.get("fragment_info", {})
    # Use the cleaned fragment ID
    output.append(f"Fragment ID: {frag_info.get('fragment', 'N/A')}")
    output.append(f"Fragment Name/Category: {frag_info.get('category', 'N/A')}")
    output.append(f"Fragment Description: {frag_info.get('description', 'N/A')}")
    output.append("\n--- User Assessment Data ---")

    for q_data in fragment_data.get("questions_data", []):
        output.append(f"\nQuestion ID: {q_data['question_id']}")
        output.append(f"Question: {q_data['question_text']}")
        output.append(f"User Answer: {json.dumps(q_data['user_answer'])}")
        proc_fields = q_data['processed_additional_fields']
        if proc_fields:
             output.append("Relevant Context/Metadata:")
             # Selectively include fields that give context for insights/roadmap
             if 'guidelines' in proc_fields:
                 output.append(f"  - Guideline: {proc_fields['guidelines']}")
             if 'sub_sub_category' in proc_fields:
                 output.append(f"  - Sub-Topic: {proc_fields['sub_sub_category']}")
             if 'tools' in proc_fields: # Include tools if present
                  output.append(f"  - Available Tools Info: {json.dumps(proc_fields['tools'], indent=2)}")
             if 'recommendations' in proc_fields: # Specific matched recommendations
                 output.append(f"  - Specific Recommendation Context: {json.dumps(proc_fields['recommendations'], indent=2)}")
             elif 'recommendation' in proc_fields: # General step-wise recommendations
                  output.append(f"  - General Recommendation Context: {json.dumps(proc_fields['recommendation'], indent=2)}")
        output.append("-" * 10)

    return "\n".join(output)

def generate_insights_prompt(fragment_context: str) -> str:
    """Creates the prompt for generating Top 3 Insights."""
    prompt = f"""
Based on the following assessment data for a specific fragment, identify the Top 3 key insights regarding the user's current state, challenges, or strengths related to this area.

Assessment Data:
{fragment_context}

Instructions:
- Analyze the questions, user answers, and associated metadata.
- Synthesize the information to derive meaningful insights.
- Present the insights as 3 distinct points or paragraphs.
- Focus on the most impactful observations based on the provided data.
- Ensure the output is formatted clearly, with each insight presented separately.
- Do not include introductory phrases like "Here are the top 3 insights:". Just provide the insights directly.
- Do not use markdown formatting like '***', '---', or numbering like '1.'. Start each insight on a new line.

Top 3 Insights:
[Insight 1 text]
[Insight 2 text]
[Insight 3 text]
"""
    return prompt

def generate_roadmap_prompt(fragment_context: str) -> str:
    """Creates the prompt for generating the Roadmap Summary as JSON."""
    prompt = f"""
Based on the following assessment data for a specific fragment, generate a 3-phase implementation roadmap (Early, Intermediate, Advanced Steps) **strictly as a JSON object**.

Assessment Data:
{fragment_context}

Instructions:
- Analyze the questions, user answers, and available metadata (including guidelines, tools, and *any existing step-wise recommendations* found in the 'General Recommendation Context' or 'Specific Recommendation Context' fields above).
- Synthesize this information to create a coherent roadmap for improvement in the fragment's area.
- **Aggregate existing step-wise recommendations**: Merge Step 1s into the 'Early Steps' phase, Step 2s into 'Intermediate Steps', etc. Generate descriptions for phases based on questions *without* pre-defined steps as well.
- Estimate man-hours realistically. If estimates exist in the data (e.g., 'man_hour_total' within recommendation steps), use them as guidance and aggregate them per phase. If not, provide reasonable estimates. Ensure 'total_hours' per phase is a number.
- Consolidate roles and tools mentioned across relevant questions/recommendations for each phase.
- **Output ONLY the JSON object**, starting with `{{` and ending with `}}`. Do not include any introductory text, concluding remarks, markdown formatting, or the markers "--- ROADMAP START/END ---".

Required JSON Structure:
{{
  "fragment_id": "[Fragment ID from Assessment Data - should be simple like 'A', 'B', etc.]",
  "fragment_name": "[Fragment Name/Category from Assessment Data]",
  "summary_context": "[~2-3 sentence narrative overview of the current state based on the assessment data]",
  "total_estimated_hours": "[Sum of total_hours from all phases below - numeric integer]",
  "key_roles_involved": "[Consolidated list of unique roles across all phases as a comma-separated string or list of strings, e.g., 'IT Specialist, Project Manager, SRE']",
  "phases": [
    {{
      "phase_name": "Early Steps",
      "step_title": "[e.g., Initial Assessment & Planning]",
      "description": "[Overview of what this phase aims to do (aggregated goal based on data and Step 1s)]",
      "how_methodology": "[Step-by-step activities (combine generated steps and existing Step 1 descriptions)]",
      "who_roles_involved": "[List of roles for this phase as a comma-separated string or list of strings]",
      "tools_platforms_used": "[List of tools/platforms for this phase as a comma-separated string or list of strings]",
      "estimate_per_subtask": "[Labelled subtasks and hours as a string (e.g., 'Capability Mapping – 20h; Tool Selection - 15h')]",
      "total_hours": "[Numeric integer total hours for this phase]"
    }},
    {{
      "phase_name": "Intermediate Steps",
      "step_title": "[e.g., Practice Implementation & Configuration]",
      "description": "[Aggregated goal for mid-stage maturity (based on data and Step 2s)]",
      "how_methodology": "[Approach for implementation, configuration, validation (combine generated steps and existing Step 2 descriptions)]",
      "who_roles_involved": "[List of roles for this phase as a comma-separated string or list of strings]",
      "tools_platforms_used": "[List of tools/platforms for this phase as a comma-separated string or list of strings]",
      "estimate_per_subtask": "[Labelled subtasks and hours as a string (e.g., 'Tool Config – 30h; Initial Rollout - 25h')]",
      "total_hours": "[Numeric integer total hours for this phase]"
    }},
    {{
      "phase_name": "Advanced Steps",
      "step_title": "[e.g., Optimization, Monitoring & Governance]",
      "description": "[Consolidated plan for tuning, validation, operationalizing (based on data and Step 3s)]",
      "how_methodology": "[Process for performance tracking, SLA adherence, long-term alignment (combine generated steps and existing Step 3 descriptions)]",
      "who_roles_involved": "[List of roles for this phase as a comma-separated string or list of strings]",
      "tools_platforms_used": "[List of tools/platforms for this phase as a comma-separated string or list of strings]",
      "estimate_per_subtask": "[Labelled subtasks and hours as a string (e.g., 'KPI Setup – 15h; Performance Tuning - 20h')]",
      "total_hours": "[Numeric integer total hours for this phase]"
    }}
  ]
}}
"""
    return prompt

def parse_roadmap_output_json(roadmap_json_text: str) -> Optional[Dict[str, Any]]:
    """
    Parses the roadmap JSON text generated by the LLM.
    Returns a dictionary with roadmap details or None if parsing fails.
    """
    if not roadmap_json_text:
        print("Warning: Roadmap JSON text is empty.")
        return None

    # Clean potential markdown code blocks or leading/trailing whitespace
    cleaned_text = roadmap_json_text.strip()
    # Remove ```json ... ``` or ``` ... ``` markers
    cleaned_text = re.sub(r'^```(?:json)?\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
    cleaned_text = cleaned_text.strip()

    try:
        # Ensure the text looks like a JSON object before parsing
        if not cleaned_text.startswith("{") or not cleaned_text.endswith("}"):
             print("Warning: Roadmap text doesn't start/end with {} braces. Attempting parse anyway.")
             # Attempt to find JSON object within text if necessary (more complex)
             json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
             if json_match:
                  print("Info: Found potential JSON object within text.")
                  cleaned_text = json_match.group(0)
             else:
                  print("Error: Could not find valid JSON object structure in the text.")
                  raise json.JSONDecodeError("Invalid JSON structure", cleaned_text, 0)


        roadmap_data = json.loads(cleaned_text)

        # --- Basic Validation ---
        if not isinstance(roadmap_data, dict):
            print("Error: Parsed roadmap JSON is not a dictionary.")
            return None
        if "phases" not in roadmap_data or not isinstance(roadmap_data["phases"], list):
            print("Error: Parsed roadmap JSON missing 'phases' list.")
            return None
        # Allow for flexibility in phase count, but warn if not 3
        if len(roadmap_data["phases"]) != 3:
            print(f"Warning: Expected 3 phases, but parsed {len(roadmap_data['phases'])}.")
        if not roadmap_data["phases"]: # Check if phases list is empty
             print("Error: Parsed roadmap JSON has an empty 'phases' list.")
             return None


        # Ensure hours are numeric integers and calculate sum
        total_hours_calc = 0
        for i, phase in enumerate(roadmap_data["phases"]):
            if not isinstance(phase, dict):
                 print(f"Warning: Phase item at index {i} is not a dictionary: {phase}. Skipping.")
                 continue # Skip this phase for calculations/renaming if it's not a dict
            try:
                # Ensure total_hours exists and is numeric, default to 0
                phase['total_hours'] = int(float(phase.get('total_hours', 0)))
                total_hours_calc += phase['total_hours']
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse total_hours for phase '{phase.get('phase_name', f'Index {i}')}': {e}. Setting to 0.")
                phase['total_hours'] = 0

        # Validate or update top-level total hours
        try:
            roadmap_data['total_estimated_hours'] = int(float(roadmap_data.get('total_estimated_hours', 0)))
        except (ValueError, TypeError):
             print("Warning: Could not parse top-level 'total_estimated_hours'. Using sum of phases.")
             roadmap_data['total_estimated_hours'] = total_hours_calc

        if roadmap_data['total_estimated_hours'] != total_hours_calc:
             print(f"Warning: Top-level total hours ({roadmap_data.get('total_estimated_hours', 'N/A')}) differs from phase sum ({total_hours_calc}). Using phase sum.")
             roadmap_data['total_estimated_hours'] = total_hours_calc

        # Rename keys from JSON prompt structure to internal structure used by PPT code
        for i, phase in enumerate(roadmap_data["phases"]):
             if not isinstance(phase, dict): continue # Skip if phase is not a dict

             phase["name"] = phase.pop("phase_name", f"Phase {i+1}")
             phase["title"] = phase.pop("step_title", "N/A")
             # phase["description"] = phase.get("description", "N/A") # Name already matches
             phase["how"] = phase.pop("how_methodology", "N/A")
             phase["who"] = phase.pop("who_roles_involved", "N/A")
             # Ensure 'who' is a string for the PPT
             if isinstance(phase["who"], list): phase["who"] = ", ".join(map(str, phase["who"]))
             phase["tools"] = phase.pop("tools_platforms_used", "N/A")
              # Ensure 'tools' is a string for the PPT
             if isinstance(phase["tools"], list): phase["tools"] = ", ".join(map(str, phase["tools"]))
             phase["subtasks"] = phase.pop("estimate_per_subtask", "N/A")
             # phase["total_hours"] = phase.get("total_hours", 0) # Name matches, type already validated

        # Ensure top-level key_roles_involved is a string
        if "key_roles_involved" in roadmap_data and isinstance(roadmap_data["key_roles_involved"], list):
            roadmap_data["key_roles_involved"] = ", ".join(map(str, roadmap_data["key_roles_involved"]))
        elif "key_roles_involved" not in roadmap_data:
             roadmap_data["key_roles_involved"] = "N/A" # Ensure field exists

        return roadmap_data

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode roadmap JSON: {e}")
        print(f"Error occurred at line {e.lineno}, column {e.colno}")
        print("--- Roadmap Text causing JSON error ---")
        # Print more context around the error location
        error_context_start = max(0, e.pos - 150)
        error_context_end = min(len(cleaned_text), e.pos + 150)
        print(f"... Context around position {e.pos} ...")
        print(cleaned_text[error_context_start:error_context_end])
        print("--- End of Roadmap Text ---")
        return None
    except Exception as e:
        print(f"Error processing parsed roadmap data: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        return None

def create_roadmap_graph(roadmap_data: Dict[str, Any]) -> Optional[io.BytesIO]:
    """ Generates roadmap bar chart, returns in-memory buffer. """
    if not roadmap_data or not roadmap_data.get("phases"): print("Graph: No phase data."); return None
    phases = roadmap_data["phases"]; phase_names = [p.get("name", f"P{i+1}") for i, p in enumerate(phases)]
    total_hours = [p.get("total_hours", 0) for p in phases]; total_days = [h / HOURS_PER_DAY for h in total_hours]
    if not any(d > 0 for d in total_days): print("Graph: Durations zero/negative."); return None
    try:
        fig, ax = plt.subplots(figsize=(7, 4)) # Smaller size
        bars = ax.bar(phase_names, total_days, color=['#4A90E2', '#F5A623', '#50E3C2'])
        ax.set_ylabel(f'Est. Duration (Days)', fontsize=9); ax.set_title(f'Phase Durations: {roadmap_data.get("fragment_name", "Fragment")}', fontsize=10)
        ax.set_xticks(range(len(phase_names))); ax.set_xticklabels(phase_names, rotation=0, ha='center', fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6); ax.tick_params(axis='y', labelsize=9)
        for bar in bars:
            yval = bar.get_height();
            if yval > 0: plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom', ha='center', fontsize=8)
        plt.tight_layout(pad=0.5) # Adjust padding
        img_buffer = io.BytesIO(); plt.savefig(img_buffer, format='png', dpi=120); # Lower DPI
        img_buffer.seek(0); plt.close(fig); return img_buffer
    except Exception as e: print(f"Error generating graph: {e}"); return None

# --- PPT Generation Functions ---

def add_text_to_shape(shape, text):
    """Helper to add text to a shape, handling None values."""
    if shape and hasattr(shape, "text_frame"): tf = shape.text_frame; tf.text = str(text) if text is not None else ""; tf.word_wrap = True
    elif shape and hasattr(shape, "text"): shape.text = str(text) if text is not None else ""

def add_text_slide(prs: Presentation, title_text: str, content_text: str, layout_index: int = DEFAULT_LAYOUT_TITLE_CONTENT): # Use constant
    """ Adds slide with title/content, handles overflow, uses safe placeholder access. """
    try: slide_layout = prs.slide_layouts[layout_index]
    except IndexError: print(f"Error: Layout index {layout_index} OOR. Using 0."); slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    title_shape = None
    try: title_shape = slide.shapes.title
    except AttributeError:
        try: title_shape = slide.placeholders[0]; print(f"Info: Layout {layout_index} no .title, using idx 0.")
        except (KeyError, IndexError): print(f"Warning: Layout {layout_index} no title via .title or idx 0.")
    except Exception as e: print(f"Warning: Err accessing title L{layout_index}: {e}")
    if title_shape: add_text_to_shape(title_shape, title_text)
    else: print(f"Info: No title placeholder for '{title_text}'.")
    content_shape = None; body_ph_found = False
    try: content_shape = slide.placeholders[1]; body_ph_found = True # Try index 1 first
    except (KeyError, IndexError):
        for idx, ph in enumerate(slide.placeholders):
             if ph.is_placeholder and (ph.name.lower().startswith('content') or ph.name.lower().startswith('body')):
                 if title_shape and hasattr(title_shape, 'placeholder_format') and hasattr(ph, 'placeholder_format') and ph.placeholder_format.idx == title_shape.placeholder_format.idx: continue
                 content_shape = ph; body_ph_found = True; print(f"Info: Using placeholder idx {idx} ('{ph.name}') as content for L{layout_index}."); break
        if not body_ph_found: print(f"Warning: Layout {layout_index} missing content placeholder (idx 1 or name).")
    except Exception as e: print(f"Warning: Err accessing content placeholder L{layout_index}: {e}")
    if content_shape:
        tf = content_shape.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
        tf.margin_bottom = Inches(0.1); tf.margin_top = Inches(0.1)
        paragraphs = str(content_text).split('\n'); current_text = ""; MAX_LINES = 20 # Adjust as needed
        lines_on_slide = 0; first_slide = True; current_slide = slide
        for para in paragraphs:
            est_lines = max(1, (len(para) // 70) + para.count('\n') + 1)
            if not first_slide and (lines_on_slide + est_lines > MAX_LINES):
                add_text_to_shape(tf, current_text.strip()) # Add text to previous slide
                current_slide = prs.slides.add_slide(slide_layout) # New slide
                new_title_shape = None # Safely get title shape on new slide
                try: new_title_shape = current_slide.shapes.title
                except AttributeError: 
                    try: new_title_shape = current_slide.placeholders[0]
                    except (KeyError, IndexError): pass
                if new_title_shape: add_text_to_shape(new_title_shape, f"{title_text} (cont.)")
                try: # Safely get content shape on new slide
                    content_shape = current_slide.placeholders[1]
                    tf = content_shape.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
                except (KeyError, IndexError): print("Error: Overflow slide missing content placeholder 1."); content_shape = None; break
                if content_shape: current_text = para + "\n"; lines_on_slide = est_lines
                else: break # Stop if content shape fails on overflow slide
            else: current_text += para + "\n"; lines_on_slide += est_lines; first_slide = False
        if content_shape: add_text_to_shape(tf, current_text.strip()) # Add remaining text
    else: print(f"Info: Skipping content for '{title_text}', no placeholder found.")

def build_ppt_report(session_id: str, report_data: Dict[str, Dict[str, Any]], output_filename: str, template_path_relative: Optional[str] = "template.pptx"):
    """Builds the PowerPoint presentation, attempting to load a template.
        Returns a BytesIO buffer containing the PPTX data, or None on failure.
    """
    prs = None # Initialize prs to None

        # --- Construct Absolute Path and Load Template ---
    if template_path_relative:
            try:
                # Get the directory where the currently running script is located
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # Join the script directory with the relative template filename
                absolute_template_path = os.path.join(script_dir, template_path_relative)

                print(f"Attempting to load template from: {absolute_template_path}") # Debug print

                if os.path.exists(absolute_template_path):
                    prs = Presentation(absolute_template_path)
                    print(f"Successfully loaded template: {absolute_template_path}")
                else:
                    print(f"Warning: Template file '{template_path_relative}' not found at expected location: {absolute_template_path}")
            except NameError:
                # __file__ might not be defined if running interactively, handle this edge case
                print("Warning: Could not determine script directory (__file__ not defined). Looking for template in current working directory.")
                if os.path.exists(template_path_relative):
                    try:
                        prs = Presentation(template_path_relative)
                        print(f"Successfully loaded template from CWD: {template_path_relative}")
                    except Exception as e_cwd:
                        print(f"Error loading template '{template_path_relative}' from CWD: {e_cwd}")
                else:
                    print(f"Warning: Template file '{template_path_relative}' not found in CWD either.")
            except Exception as e:
                print(f"Error loading presentation template '{absolute_template_path}': {e}")

    # If template loading failed or wasn't specified, use default
    if prs is None:
         print("Using default python-pptx presentation template.")
         prs = Presentation()

    # Slide 1: Title Slide (Layout 0)
    try:
        title_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_layout)
        try: add_text_to_shape(slide.shapes.title, "Recommendation Report")
        except AttributeError: print("Warning: Title slide layout 0 missing .title shape.")
        try: add_text_to_shape(slide.placeholders[1], f"Session ID: {session_id}\nGenerated: {time.strftime('%Y-%m-%d')}")
        except (KeyError, IndexError): print("Warning: Title slide layout 0 missing subtitle idx 1.")
    except Exception as e: print(f"Error creating title slide: {e}")

    # Process fragments
    sorted_fragment_ids = sorted(report_data.keys())
    for fragment_id in sorted_fragment_ids:
        data = report_data[fragment_id]; frag_info = data.get("fragment_info", {})
        frag_name = frag_info.get('category') or frag_info.get('description') or fragment_id
        if not frag_name: frag_name = fragment_id
        frag_name = str(frag_name).strip()
        print(f"Generating slides for Fragment: {fragment_id} ({frag_name})...")

        # Slide: Fragment Title (Layout 2)
        try:
            header_layout = prs.slide_layouts[DEFAULT_LAYOUT_SECTION_HEADER] # Use constant
            slide = prs.slides.add_slide(header_layout)
            title_shape = None
            try: title_shape = slide.shapes.title
            except AttributeError: 
                try: title_shape = slide.placeholders[0]
                except (KeyError, IndexError): pass
            if title_shape: add_text_to_shape(title_shape, f"Fragment: {fragment_id}")
            else: print(f"Warning: Layout {DEFAULT_LAYOUT_SECTION_HEADER} missing title placeholder.")
        except IndexError: print(f"Error: Layout index {DEFAULT_LAYOUT_SECTION_HEADER} OOR for Section Header.")
        except Exception as e: print(f"Error creating fragment title slide for {frag_name}: {e}")

        # Slide: Top 3 Insights (Uses add_text_slide -> layout DEFAULT_LAYOUT_TITLE_CONTENT)
        insights = data.get("insights"); insights_title = f"{fragment_id}: Top 3 Insights"
        if insights:
             insights_cleaned = re.sub(r'^\s*(\d+\.|-|\*)\s*', '', insights, flags=re.MULTILINE).strip().replace('***', '').replace('---', '')
             add_text_slide(prs, insights_title, insights_cleaned if insights_cleaned else "Insights generated but empty.")
        else: add_text_slide(prs, insights_title, "Insights could not be generated.")

        # Slides: Roadmap Summary
        roadmap = data.get("roadmap"); roadmap_base_title = f"{fragment_id}: Roadmap"
        if roadmap and roadmap.get("phases"):
            # Roadmap Overview slide (Layout DEFAULT_LAYOUT_TITLE_CONTENT)
            try:
                overview_title = f"{fragment_id}: Roadmap Overview"
                overview_layout = prs.slide_layouts[DEFAULT_LAYOUT_TITLE_CONTENT]
                slide = prs.slides.add_slide(overview_layout)
                title_shape_ov = None # Safe title access
                try: title_shape_ov = slide.shapes.title
                except AttributeError: 
                    try: title_shape_ov = slide.placeholders[0]
                    except(KeyError, IndexError): pass
                if title_shape_ov: add_text_to_shape(title_shape_ov, overview_title)
                else: print(f"Warning: Layout {DEFAULT_LAYOUT_TITLE_CONTENT} missing title for roadmap overview.")
                # Safe content access
                overview_text = f"Summary: {roadmap.get('summary_context', 'N/A')}\nTotal Est: {roadmap.get('total_estimated_hours', 0)} hrs (~{roadmap.get('total_estimated_hours', 0)/HOURS_PER_DAY:.1f} days)\nRoles: {roadmap.get('key_roles_involved', 'N/A')}"
                content_shape_ov = None
                try: content_shape_ov = slide.placeholders[1]
                except (KeyError, IndexError): print(f"Warning: Layout {DEFAULT_LAYOUT_TITLE_CONTENT} missing content idx 1 for overview.")
                if content_shape_ov: add_text_to_shape(content_shape_ov.text_frame, overview_text)
                else: 
                    try: add_text_to_shape(slide.notes_slide.notes_text_frame, "Ov. Content:\n" + overview_text) # Fallback notes
                    except Exception: pass
            except IndexError: print(f"Error: Layout index {DEFAULT_LAYOUT_TITLE_CONTENT} OOR for Roadmap Overview.")
            except Exception as e: print(f"Error creating roadmap overview slide: {e}")
            # Roadmap Phase Detail slides (Uses add_text_slide -> layout DEFAULT_LAYOUT_TITLE_CONTENT)
            for i, phase in enumerate(roadmap["phases"]):
                if not isinstance(phase, dict): continue
                phase_title = f"{fragment_id}: {phase.get('name', f'Phase {i+1}')}"
                phase_content = f"Goal: {phase.get('title', 'N/A')}\n\nDesc: {phase.get('description', 'N/A')}\n\nHow: {phase.get('how', 'N/A')}\n\nWho: {phase.get('who', 'N/A')}\nTools: {phase.get('tools', 'N/A')}\n\nSubtasks: {phase.get('subtasks', 'N/A')}\n\nDuration: {phase.get('total_hours', 0)} hrs (~{phase.get('total_hours', 0)/HOURS_PER_DAY:.1f} days)"
                add_text_slide(prs, phase_title, phase_content)
        else: add_text_slide(prs, roadmap_base_title, "Roadmap could not be generated or parsed.")

        # Slide: Roadmap Graph (Layout DEFAULT_LAYOUT_TITLE_ONLY or TITLE_CONTENT)
        graph_buffer = data.get("graph_buffer"); graph_title = f"{fragment_id}: Roadmap Duration Overview"
        if graph_buffer:
             try:
                 # Try "Title Only" layout first, fallback to "Title and Content"
                 graph_layout = None
                 try: graph_layout = prs.slide_layouts[DEFAULT_LAYOUT_TITLE_ONLY]
                 except IndexError: print(f"Warning: Layout {DEFAULT_LAYOUT_TITLE_ONLY} (Title Only) not found, using {DEFAULT_LAYOUT_TITLE_CONTENT}."); graph_layout = prs.slide_layouts[DEFAULT_LAYOUT_TITLE_CONTENT]

                 slide = prs.slides.add_slide(graph_layout)
                 title_shape_g = None # Safe title access
                 try: title_shape_g = slide.shapes.title
                 except AttributeError: 
                    try: title_shape_g = slide.placeholders[0]
                    except(KeyError, IndexError): pass
                 if title_shape_g: add_text_to_shape(title_shape_g, graph_title)
                 else: print(f"Warning: Layout for graph slide missing title placeholder.")
                 # Add graph image
                 left = Inches(0.5); top = Inches(1.2); width = Inches(9.0) # Adjust position/size for layout
                 # Calculate height based on aspect ratio if possible, otherwise fixed height
                 # Example fixed height:
                 height = Inches(5.0)
                 pic = slide.shapes.add_picture(graph_buffer, left, top, height=height) # Use height or width
                 graph_buffer.close()
             except IndexError: print(f"Error: Layout index {DEFAULT_LAYOUT_TITLE_CONTENT} OOR for Graph fallback.")
             except Exception as e: print(f"Error adding graph slide: {e}");
             if graph_buffer and not graph_buffer.closed: graph_buffer.close()
        elif roadmap and roadmap.get("phases"):
            add_text_slide(prs, f"{fragment_id}: Roadmap Graph", "Graph not generated.")
    try:
        ppt_buffer = io.BytesIO()
        prs.save(ppt_buffer)
        ppt_buffer.seek(0) # Rewind buffer to the beginning for reading
        print("PPTX successfully saved to memory buffer.")
        return ppt_buffer
    except Exception as e:
        print(f"\nError saving presentation to buffer: {e}")
        return None
""" # Save final presentation
    try: prs.save(output_filename); print(f"\nPresentation saved: {output_filename}")
    except Exception as e: print(f"\nError saving presentation '{output_filename}': {e}")
"""

def upload_report_to_supabase(session_id: str, ppt_buffer: io.BytesIO, bucket_name: str = "recommendations") -> bool:
    """Uploads the generated PPTX buffer's content (bytes) to Supabase Storage."""
    if not ppt_buffer:
        print("Upload failed: No PPT buffer provided.")
        return False

    # Define the path within the bucket
    remote_path = f"{session_id}/recommendation_report.pptx"

    try:
        print(f"Attempting to upload report bytes to Supabase Storage: bucket='{bucket_name}', path='{remote_path}'")
        supabase_client = get_supabase_client()

        # --- FIX: Read bytes from buffer and pass the bytes ---
        ppt_buffer.seek(0) # Go to the beginning of the buffer
        ppt_bytes = ppt_buffer.read() # Read the entire content into a bytes object
        # --- End Fix ---

        # Now pass the bytes object as the 'file' argument
        upload_response = supabase_client.storage.from_(bucket_name).upload(
            path=remote_path,
            file=ppt_bytes, # Pass the raw bytes
            file_options={"content-type": "application/vnd.openxmlformats-officedocument.presentationml.presentation", "upsert": "true"}
        )

        # Basic check (often raises exception on failure)
        print(f"Supabase upload finished for {remote_path}")
        # You might want more robust checking depending on the exact client library behavior
        # For example, sometimes you check response status if available, or just assume success if no exception.

        print(f"Successfully uploaded report to {bucket_name}/{remote_path}")
        return True

    except Exception as e:
        print(f"Error uploading report to Supabase Storage: {e}")
        traceback.print_exc()
        return False
    finally:
        # Ensure buffer is closed even if upload fails
        if ppt_buffer:
             ppt_buffer.close()

def generate_recommendations(session_id: str) -> Dict[str, Dict[str, Any]]:
    """ Main function to generate recommendations data. """
    print(f"Starting recommendation generation for session_id: {session_id}")
    supabase_client = get_supabase_client()
    print("Fetching data...")
    followup_resp = fetch_followup_data(supabase_client, session_id);
    if not followup_resp: print("No followup responses found."); return {}
    q_ids = list(set(r['question_id'] for r in followup_resp if r.get('question_id')))
    q_details = fetch_question_details(supabase_client, q_ids);
    if not q_details: print("No question details found."); return {}
    print("Processing and grouping data...")
    grouped_data = group_data_by_fragment(followup_resp, q_details);
    if not grouped_data: print("No data grouped by fragment."); return {}

    final_report_data = {}
    sorted_fragment_ids = sorted(grouped_data.keys())
    for fragment_id in sorted_fragment_ids:
        fragment_data = grouped_data[fragment_id]; frag_info = fragment_data.get("fragment_info", {})
        frag_name = frag_info.get('category') or frag_info.get('description') or fragment_id
        print(f"\n--- Processing Fragment: {fragment_id} ({frag_name}) ---")
        context_str = format_data_for_prompt(fragment_data)
        final_report_data[fragment_id] = {"fragment_info": frag_info}

        print("Generating insights...") # Insights
        insights_prompt = generate_insights_prompt(context_str)
        insights_result = call_llm(insights_prompt, max_tokens=800, temperature=0.5)
        final_report_data[fragment_id]["insights"] = insights_result
        print("Insights generated." if insights_result else "Failed to generate insights.")

        print("Pausing before roadmap..."); time.sleep(2) # Rate limit pause

        print("Generating roadmap (JSON)...") # Roadmap
        roadmap_prompt = generate_roadmap_prompt(context_str)
        roadmap_txt = call_llm(roadmap_prompt, max_tokens=3500, temperature=0.3) # Increased tokens
        if roadmap_txt:
            print("Parsing roadmap JSON...")
            roadmap_parsed = parse_roadmap_output_json(roadmap_txt)
            if roadmap_parsed:
                if "fragment_name" not in roadmap_parsed or not roadmap_parsed["fragment_name"]: roadmap_parsed["fragment_name"] = frag_name # Ensure name
                final_report_data[fragment_id]["roadmap"] = roadmap_parsed
                print("Roadmap parsed successfully.")
            else: final_report_data[fragment_id]["roadmap"] = None; print("Failed to parse roadmap JSON.")
        else: final_report_data[fragment_id]["roadmap"] = None; print("Failed to generate roadmap JSON text.")

        print("Generating graph...") # Graph
        if final_report_data[fragment_id].get("roadmap"):
            graph_buffer = create_roadmap_graph(final_report_data[fragment_id]["roadmap"])
            final_report_data[fragment_id]["graph_buffer"] = graph_buffer
            print("Graph generated." if graph_buffer else "Failed to generate graph.")
        else: final_report_data[fragment_id]["graph_buffer"] = None; print("Skipping graph (no roadmap).")

    return final_report_data

# --- Main Orchestration Function ---
# --- NEW: Main orchestrator function for background task ---

def generate_and_upload_recommendations(session_id: str, template_path_relative: Optional[str] = "template.pptx") -> bool:
     """
     Generates recommendations, builds the PPT report in memory, and uploads it.
     Returns True on success, False on failure.
     Can be async if parts of generate_recommendations become async, otherwise sync.
     """
     print(f"Starting recommendation generation process for session {session_id}...")
     try:
          # 1. Generate the report content data (sync or async depends on generate_recommendations)
          # Assuming generate_recommendations is sync for now
          report_content = generate_recommendations(session_id)
          if not report_content:
               print(f"Failed to generate report content for session {session_id}.")
               return False
          print(f"Report content generated for session {session_id}.")

          # 2. Build the PPT report into a memory buffer
          print(f"Building PPT report for session {session_id}...")
          ppt_buffer = build_ppt_report(session_id, report_content, template_path_relative)
          if not ppt_buffer:
               print(f"Failed to build PPT report buffer for session {session_id}.")
               return False
          print(f"PPT report built in memory for session {session_id}.")

          # 3. Upload the buffer to Supabase Storage
          print(f"Uploading report for session {session_id}...")
          upload_success = upload_report_to_supabase(session_id, ppt_buffer) # Buffer is closed inside this function

          return upload_success

     except Exception as e:
          print(f"Error during generate_and_upload_recommendations for session {session_id}: {e}")
          traceback.print_exc()
          return False



# --- Main Execution Block ---

if __name__ == "__main__":
    # --- Hardcoded session ID ---
    session_id_to_process = "1e195af3-b327-483a-ab8f-8c5f3940705c" # <-- EDIT THIS LINE
    template_file_name = "template.pptx" # <--- NAME OF YOUR TEMPLATE FILE
    # ---

    parser = argparse.ArgumentParser(description="Generate recommendations PPT from hardcoded session_id.")
    parser.add_argument("-o", "--output", type=str, default="Recommendation_Report_Generated.pptx", help="Output PPT filename.")
    args = parser.parse_args()
    output_filename = args.output

    if not session_id_to_process or session_id_to_process == "YOUR_ACTUAL_SESSION_ID_HERE": # Basic check
        print("Error: Please edit the 'session_id_to_process' variable in the script."); sys.exit(1)

    print(f"Processing hardcoded session_id: {session_id_to_process}")
    print(f"Attempting to use template: {template_file_name}")
    print(f"Output file set to: {output_filename}")

    report_content = generate_recommendations(session_id_to_process)

    if report_content:
        print("\nBuilding PowerPoint presentation...")
        # Pass the template path to the build function
        build_ppt_report(session_id_to_process, report_content, output_filename, template_path_relative=template_file_name)
    else:
        print("\nNo report data generated. PPT creation skipped.")# --- NEW: Main orchestrator function for background task ---