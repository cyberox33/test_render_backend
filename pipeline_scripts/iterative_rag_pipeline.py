import json
import time
import re
import copy
import sys, os
from collections import deque
from typing import Dict, Any, List, Optional
from supabase import create_client, Client
import google.generativeai as genai
import google.api_core.exceptions

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.utils import config
from . import prompt_templates, iteration_logic

SUPABASE_URL = config.SUPABASE_URL
SUPABASE_KEY = config.SUPABASE_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=config.GOOGLE_API_KEY)

LLM_MODEL_NAME = "gemma-3-27b-it"
EMBEDDING_MODEL = "models/text-embedding-004"

# Baseline question IDs (adjust if needed)
BASELINE_QUESTION_IDS = [59, 61, 64, 65, 33, 4, 25]

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

def embed_text(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> Optional[List[float]]:
    """Generates embeddings for the given text using the Gemini API."""
    if not text:
        print("Warning: Attempted to embed empty text.")
        return None
    try:
        # Use RETRIEVAL_QUERY for generating embeddings for search queries
        # Use RETRIEVAL_DOCUMENT for embedding the documents being searched
        # Use SEMANTIC_SIMILARITY for general similarity tasks
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding for text snippet '{text[:50]}...': {e}")
        # Return None or a default zero vector if needed, but None is safer
        return None

def build_category_filter(survey_responses) -> dict:
    category_filter = {}  # Default: no category filter
    q16_answer = ""

    if isinstance(survey_responses, list):
        for resp in survey_responses:
            if resp.get("question_id") == "Q16":
                ans_val = resp.get("answer")
                if isinstance(ans_val, dict):
                    ans_val = ans_val.get("value") or ans_val.get("content")
                if ans_val == "1":
                    q16_answer = "green with software"
                    category_filter = {"category": "Green With Software"}
                elif ans_val == "2":
                    q16_answer = "green within software"
                    category_filter = {"category": "Green Within Software"}
                break
    elif isinstance(survey_responses, dict):
        if survey_responses.get("question_id") == "Q16":
            ans_val = survey_responses.get("answer")
            if isinstance(ans_val, dict):
                ans_val = ans_val.get("value") or ans_val.get("content")
            if ans_val == "1":
                q16_answer = "green with software"
                category_filter = {"category": "Green With Software"}
            elif ans_val == "2":
                q16_answer = "green within software"
                category_filter = {"category": "Green Within Software"}

    print(f"Category filter based on Q16 ('{q16_answer}'): {category_filter}")
    return category_filter

def get_survey_questions() -> List[Dict[str, Any]]:
    """Fetches all survey questions from Supabase."""
    try:
        response = supabase.table("survey_questions").select("*").execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching survey questions: {e}")
        return []


def get_survey_responses(session_id: str) -> Optional[List[Dict[str, Any]]]:
    """Fetches survey responses for a given session_id."""
    try:
        response = supabase.table("survey_responses") \
            .select("responses") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        if response.data:
            # Assuming 'responses' column stores the JSON array directly
            responses_json = response.data[0].get("responses")
            if isinstance(responses_json, list):
                 return responses_json
            elif isinstance(responses_json, str):
                 # Handle case where JSON might be stored as a string
                 try:
                     return json.loads(responses_json)
                 except json.JSONDecodeError:
                     print(f"Error decoding survey responses JSON for session {session_id}")
                     return None
            else:
                 print(f"Unexpected format for survey responses: {type(responses_json)}")
                 return None

        print(f"No survey responses found for session {session_id}")
        return None
    except Exception as e:
        print(f"Error fetching survey responses for session {session_id}: {e}")
        return None


def get_followup_responses(session_id: str, answered_only: bool = False, question_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Fetches follow-up responses, optionally filtering by answered status or specific question IDs."""
    try:
        query = supabase.table("followup_responses").select("*").eq("session_id", session_id)
        if answered_only:
            query = query.not_.is_("answer", None) # Filter for rows where answer is NOT null
        if question_ids:
             query = query.in_("question_id", question_ids)

        response = query.execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching follow-up responses for session {session_id}: {e}")
        return []

def get_question_details_by_ids(question_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Fetches full question details from the 'questions' table for a list of IDs."""
    if not question_ids:
        return {}
    try:
        response = supabase.table("questions").select("*").in_("id", question_ids).execute()
        return {q['id']: q for q in response.data} if response.data else {}
    except Exception as e:
        print(f"Error fetching question details for IDs {question_ids}: {e}")
        return {}


def get_all_fragments() -> List[str]:
    """Fetches all unique fragment identifiers from the fragment_info table."""
    try:
        # It's better to get fragments from the dedicated info table if it exists
        response = supabase.table("fragment_info").select("fragment").execute()
        if response.data:
             return sorted([item['fragment'] for item in response.data]) # Sort for consistent order
        else:
             # Fallback: Get distinct fragments from the questions table (less ideal)
             print("Warning: fragment_info table might be empty or missing. Falling back to questions table.")
             # This might require a custom SQL function or view for distinct values if performance is an issue
             # For simplicity, fetching all and using Python set (might be slow for large tables)
             q_response = supabase.table("questions").select("fragment").execute()
             if q_response.data:
                  return sorted(list(set(item['fragment'] for item in q_response.data if item.get('fragment'))))
             else:
                  print("Error: Could not retrieve fragments from questions table either.")
                  return []

    except Exception as e:
        print(f"Error fetching fragments: {e}")
        return []


def get_fragment_info(fragment_id: str) -> Optional[Dict[str, Any]]:
     """Fetches description and other info for a specific fragment."""
     try:
         response = supabase.table("fragment_info").select("*").eq("fragment", fragment_id).maybe_single().execute()
         return response.data
     except Exception as e:
         print(f"Error fetching info for fragment {fragment_id}: {e}")
         return None


def get_industry_context(survey_responses: Optional[List[Dict[str, Any]]]) -> str:
    """Extracts industry context from survey response Q17."""
    if not survey_responses:
        return "Not specified"
    for resp in survey_responses:
        if resp.get("question_id") == "Q17":
            # Assuming the answer is directly the industry string
            return resp.get("answer", "Not specified")
    return "Not specified"

def get_dependency_rules() -> List[Dict[str, Any]]:
    """Fetches all question dependency rules from Supabase."""
    try:
        response = supabase.table("question_dependency_rules").select("*").execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching dependency rules: {e}")
        return []
    
def extract_answer_value(answer_jsonb: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extracts the 'value' from the answer JSONB structure."""
    if isinstance(answer_jsonb, dict):
        return str(answer_jsonb.get("value")) if answer_jsonb.get("value") is not None else None
    return None # Return None if not a dict or 'value' is missing

def calculate_dependency_exclusions(rules: List[Dict[str, Any]], answered_responses: List[Dict[str, Any]]) -> List[int]:
    """
    Calculates which dependent question IDs should be excluded based on
    'must_not_ask' rules and the provided answered responses.
    """
    must_not_ask_rules = [r for r in rules if r.get("rule_type") == "must_not_ask"]
    if not must_not_ask_rules:
        return []

    # Create a map for quick answer lookup: {question_id: answer_value_str}
    answered_map = {resp['question_id']: extract_answer_value(resp.get('answer'))
                    for resp in answered_responses if resp.get('question_id') and resp.get('answer')}

    dependency_excluded_ids = set()

    for rule in must_not_ask_rules:
        dep_id = rule.get("dependent_question_id")
        indep_id_str = rule.get("independent_question_id")
        condition_values_str = rule.get("condition_answer_values")
        condition_type = rule.get("condition_type") # exact_match, match_any
        condition_logic = rule.get("condition_logic") # match_all, match_any, none

        if not dep_id or not indep_id_str or not condition_values_str or not condition_type or not condition_logic:
            # print(f"Warning: Skipping invalid rule: {rule}") # Optional
            continue

        try:
            indep_ids = [int(i.strip()) for i in indep_id_str.split(';') if i.strip()]
            condition_values = [v.strip() for v in condition_values_str.split(';') if v.strip()]
        except ValueError:
            # print(f"Warning: Could not parse IDs in rule: {rule}") # Optional
            continue

        if not indep_ids or not condition_values:
            continue # Skip rule if parsing failed

        # --- Check if prerequisite independent questions are answered ---
        prereqs_met = False
        num_answered_prereqs = sum(1 for i_id in indep_ids if i_id in answered_map)

        if condition_logic == 'match_all':
            if num_answered_prereqs == len(indep_ids):
                prereqs_met = True
        elif condition_logic == 'match_any' or condition_logic == 'none': # 'none' implies condition applies if any are answered
             if num_answered_prereqs > 0:
                prereqs_met = True
        # else: condition_logic is 'none' but no prereqs answered -> rule doesn't apply yet

        if not prereqs_met:
            continue # Conditions for checking the rule aren't met yet

        # --- Evaluate the answer condition ---
        condition_fulfilled = False
        match_results = [] # Store boolean result for each indep_id check

        # Iterate through the answered independent IDs relevant to this rule
        relevant_answered_indep_ids = [i_id for i_id in indep_ids if i_id in answered_map]

        for i_id in relevant_answered_indep_ids:
            actual_answer = answered_map.get(i_id)
            if actual_answer is None: # Should not happen due to answered_map construction, but safe check
                 match_results.append(False)
                 continue

            indep_match = False
            if condition_type == 'exact_match':
                # Check if the actual answer exactly matches any of the condition values
                if actual_answer in condition_values:
                    indep_match = True
            elif condition_type == 'match_any': # Check if condition value is a substring? Rule table says 'match_any' on 'condition_answer_values' - interpret as checking if any of the condition values match the answer
                 if actual_answer in condition_values:
                    indep_match = True # Treat match_any like exact_match based on examples (e.g., "Never" must match exactly)

            match_results.append(indep_match)


        # Apply the logic based on how many independent answers matched
        if condition_logic == 'match_all':
            # All answered relevant independent questions must match the condition
            if len(match_results) == len(relevant_answered_indep_ids) and all(match_results):
                condition_fulfilled = True
        elif condition_logic == 'match_any' or condition_logic == 'none':
             # Any one of the answered relevant independent questions matching is enough
             if any(match_results):
                 condition_fulfilled = True

        # If the condition is fulfilled, exclude the dependent question
        if condition_fulfilled:
            # print(f"DEBUG: Excluding Q {dep_id} based on rule for Q(s) {indep_ids} with answers. Rule: {rule}")
            dependency_excluded_ids.add(dep_id)

    return list(dependency_excluded_ids)

def format_survey_qa(survey_questions: List[Dict[str, Any]], survey_responses: List[Dict[str, Any]]) -> str:
    """Formats survey questions and answers for the LLM prompt."""
    qa_string = ""
    response_map = {resp.get("question_id"): resp.get("answer") for resp in survey_responses}

    for q in survey_questions:
        q_id = q.get("question_id")
        content = q.get("content", "N/A")
        options = q.get("options") # This might be JSONB
        user_answer_val = response_map.get(q_id)
        user_answer_text = str(user_answer_val) # Default to the raw answer

        # Try to map answer value back to option content if options exist
        if options and isinstance(options, list) and user_answer_val is not None:
            for opt in options:
                 # Assuming options have 'option_letter' or similar key that matches the answer value
                 # Adjust 'option_letter' key if your schema is different
                 if str(opt.get("option_letter")) == str(user_answer_val):
                      user_answer_text = opt.get("content", user_answer_text)
                      break
        elif q.get("subjective_answer"): # Check if it's a subjective question
             user_answer_text = str(user_answer_val) if user_answer_val is not None else "No answer"


        qa_string += f"Question ({q_id}): {content}\n"
        # Optional: Include options for context, but exclude scores
        if options and isinstance(options, list):
            options_text = ", ".join([f"{opt.get('option_letter', '?')}: {opt.get('content', '')}" for opt in options])
            qa_string += f"Options: [{options_text}]\n"

        qa_string += f"User Answer: {user_answer_text}\n\n"

    return qa_string.strip() if qa_string else "No survey questions or answers provided."


def format_followup_qa(followup_responses: List[Dict[str, Any]]) -> str:
    """Formats answered follow-up questions and answers for the LLM prompt."""
    qa_string = ""
    if not followup_responses:
         return "No previous follow-up questions answered in this fragment yet."

    for resp in followup_responses:
         # Ensure we only format answered questions (input should already be filtered, but good practice)
         if resp.get("answer") is not None:
            q_text = resp.get("question", "N/A") # Ensure question text is present
            q_id = resp.get("question_id", "N/A")
            answer_data = resp.get("answer", {}) # Answer is expected to be a dict like {"value": ...}
            answer_text = "N/A" # Default

            # --- FIX 3: Extract answer correctly from {"value": ...} ---
            if isinstance(answer_data, dict):
                # Prioritize the 'value' key based on observed data
                if 'value' in answer_data:
                    answer_text = str(answer_data['value'])
                # Add fallbacks if structure might vary
                elif 'content' in answer_data: # e.g., if survey answer format is used
                     answer_text = str(answer_data['content'])
                else: # Otherwise, just represent the dict as string
                    answer_text = str(answer_data)
            elif answer_data: # Handle case answer is not a dict (e.g., simple string/number)
                answer_text = str(answer_data)

            qa_string += f"Question (ID {q_id}): {q_text}\n"
            qa_string += f"User Answer: {answer_text}\n\n"

    # Return stripped string or the "No answered..." message if qa_string remains empty after loop
    return qa_string.strip() if qa_string.strip() else "No answered follow-up questions found for this fragment."


def format_retrieved_questions(retrieved_questions: List[Dict[str, Any]]) -> str:
     """Formats retrieved candidate questions for the ranking prompt."""
     q_string = ""
     for q in retrieved_questions:
          q_id = q.get("id", "N/A")
          q_text = q.get("question", "N/A")
          q_string += f"- [{q_id}] {q_text}\n"
          # Optionally include guidelines or options if helpful for ranking context
          # add_fields = q.get('additional_fields', {})
          # if 'guidelines' in add_fields:
          #     q_string += f"  Guidelines: {add_fields['guidelines']}\n"
          # elif 'answer_options' in add_fields:
          #      options_text = ", ".join([opt.get('content', '') for opt in add_fields['answer_options']])
          #      q_string += f"  Options: {options_text}\n"

     return q_string.strip() if q_string else "No questions retrieved."

def parse_llm_ranking(ranked_text: str, retrieved_ids: List[int]) -> List[int]:
    """Parses the LLM's ranked list and returns ordered IDs."""
    if not ranked_text:
        print("Warning: LLM ranking text is empty. Cannot parse.")
        return [] # Return empty list, caller should handle fallback

    ordered_ids = []
    # Regex to find lines like: 1. [123] Question text...
    # It captures the number within the brackets.
    pattern = re.compile(r"^\s*\d+\.\s*\[(\d+)\]", re.MULTILINE)
    matches = pattern.findall(ranked_text)

    seen_ids = set()
    for match_id_str in matches:
         try:
             q_id = int(match_id_str)
             if q_id in retrieved_ids and q_id not in seen_ids: # Ensure ID was in the original retrieved set and not duplicated
                  ordered_ids.append(q_id)
                  seen_ids.add(q_id)
             else:
                  print(f"Warning: Parsed ID {q_id} from ranking not in retrieved set or already seen. Skipping.")
         except ValueError:
              print(f"Warning: Could not parse ID '{match_id_str}' from ranking. Skipping.")

    # If parsing fails to extract any valid IDs, log a warning.
    # The caller might need a fallback mechanism (e.g., use the original retrieval order).
    if not ordered_ids:
         print("Warning: Failed to parse any valid IDs from the LLM ranking output.")

    return ordered_ids


def store_single_followup_question(session_id: str, question_data: Dict[str, Any]) -> bool:
    """Stores a single selected follow-up question into Supabase 'followup_responses' table."""
    try:
        # Extract necessary fields from the full question data fetched from 'questions' table
        q_id = question_data.get("id")
        if not q_id:
            print("Error: Question data missing ID for storage.")
            return False

        orig_fields = question_data.get("additional_fields", {})
        filtered_fields = {}
        # Carefully select only the fields needed for the frontend/user interaction
        if "guidelines" in orig_fields:
            filtered_fields["guidelines"] = orig_fields["guidelines"]
        if "answer_options" in orig_fields: # MCQ
            filtered_fields["answer_options"] = orig_fields["answer_options"]
        elif "subjective_answer" in orig_fields: # Subjective Input
            # Store the structure expected for subjective answer input (might just be a flag or type)
             filtered_fields["subjective_answer"] = orig_fields["subjective_answer"] # Or maybe just True/a type indicator
        elif "multiple_correct_answer_options" in orig_fields: # Multi-select
             filtered_fields["multiple_correct_answer_options"] = orig_fields["multiple_correct_answer_options"]

        record = {
            "session_id": session_id,
            "question_id": q_id,
            "question": question_data.get("question"),
            "category": question_data.get("category"),
            "subcategory": question_data.get("subcategory"),
            # Store only relevant interactive fields, not all metadata
            "additional_fields": filtered_fields,
            "answer": None # Initialize answer as null
        }

        res = supabase.table("followup_responses").insert(record).execute()
        if res.data:
            print(f"Stored follow-up question ID: {q_id} for session {session_id}")
            return True
        else:
             # Supabase Python client v1 might return empty list on success, v2 has different structure
             # Check for errors in the response if available or assume success if no exception
             print(f"Stored follow-up question ID: {q_id} (assuming success, check DB).")
             # Add more robust error checking based on your supabase-py version if needed
             # e.g., check response status code or specific error attributes
             # if hasattr(res, 'error') and res.error: print(f"Error storing: {res.error}") else: return True
             return True # Assume success for now

    except Exception as e:
        print(f"Error storing follow-up question ID {q_id}: {e}")
        # Check for unique constraint violations (e.g., asking the same question twice for a session)
        if "duplicate key value violates unique constraint" in str(e):
             print(f"Info: Question {q_id} likely already exists for session {session_id}.")
             # Decide if this should be treated as success or failure in the flow
             return False # Treat as failure to prevent infinite loops if logic depends on successful new storage
        return False


def wait_for_batch_answers(session_id: str, expected_question_ids: List[int]):
    """
    Waits until all questions in the provided list have non-null answers
    in the followup_responses table for the given session.
    """
    if not expected_question_ids:
        print("No questions provided to wait for.")
        return

    print(f"Waiting for user to answer batch of {len(expected_question_ids)} questions: {expected_question_ids}...")
    ids_to_check = set(expected_question_ids)
    answered_ids = set()

    while answered_ids != ids_to_check:
        remaining_ids = list(ids_to_check - answered_ids)
        try:
            # Fetch answer status for remaining questions
            response = supabase.table("followup_responses") \
                .select("question_id, answer") \
                .eq("session_id", session_id) \
                .in_("question_id", remaining_ids) \
                .execute()

            if response.data:
                newly_answered = {row['question_id'] for row in response.data if row.get('answer') is not None}
                answered_ids.update(newly_answered)

                if answered_ids == ids_to_check:
                     print("All questions in the batch have been answered. Proceeding...")
                     return # Exit function

            # Optional: More verbose waiting message
            print(f"  Waiting... {len(ids_to_check - answered_ids)} questions remaining.")
            time.sleep(15) # Wait before checking again (adjust interval as needed)

        except Exception as e:
            print(f"Error checking answers for batch {remaining_ids}: {e}")
            # Implement backoff or max retries if necessary
            time.sleep(20) # Wait longer on error

# --- Main RAG Pipeline ---

def run_iterative_rag_pipeline(session_id: str) -> None:
    print(f"\n{'='*20} Unified Pipeline Started for Session: {session_id} {'='*20}")

    # --- 1. Initial Data Fetching ---
    print("\n--- Fetching Initial Data & Setup ---")
    survey_questions = get_survey_questions()
    survey_responses = get_survey_responses(session_id)
    all_fragments_list = get_all_fragments()
    fragment_info_map = {info['fragment']: info for info in (get_fragment_info(f) for f in all_fragments_list) if info}
    all_dependency_rules = get_dependency_rules()
    all_questions_data = supabase.table("questions").select("id, fragments").execute().data or []
    all_questions_map = iteration_logic.get_fragment_question_map(all_questions_data)

    if not survey_questions or not survey_responses or not all_fragments_list or not fragment_info_map:
        print("Error: Failed to fetch critical initial data. Aborting.")
        return

    industry_context = get_industry_context(survey_responses)
    category_filter = build_category_filter(survey_responses) # Use the build_category_filter function

    fragment_states = {
        frag_id: {
            "quota_assigned": iteration_logic.FRAGMENT_QUOTAS.get(frag_id, 0),
            "quota_fulfilled": 0,
            "is_complete": False,
            "questions_asked_in_fragment": 0,
            "estimated_dependency_count": 0,
            "answered_independent_questions": 0,
            "readiness_score": 0.0,
            "maturity_score": 0.0,
            "status_summary": "No status yet." # For storing FRAGMENT_STATUS_SUMMARY output
        }
        for frag_id in all_fragments_list
    }

    print(f"Industry Context: {industry_context}")
    print(f"Fragments to process: {all_fragments_list}")
    print(f"Fragment Quotas: {iteration_logic.FRAGMENT_QUOTAS}")
    print(f"Category filter: {category_filter}")

    # --- 2. Determine and Store Baseline Questions ---
    print("\n--- Determining & Storing Baseline Questions ---")
    baseline_question_ids_to_ask = []
    if category_filter.get("category") == "Green With Software":
         baseline_question_ids_to_ask = [59, 61, 64, 65, 33, 4, 25] # Example GWS set
         print(f"Selected GWS baseline IDs: {baseline_question_ids_to_ask}")
    elif category_filter.get("category") == "Green Within Software":
         baseline_question_ids_to_ask = [] # Define baseline IDs for GWiS if applicable
         print(f"Selected GWiS baseline IDs: {baseline_question_ids_to_ask}")
    else:
         print("Warning: No category determined from Q16 or no baseline set defined for it.")
         # Decide baseline behavior if no category - maybe ask a default small set
     
    baseline_stored_count = 0
    if baseline_question_ids_to_ask:
        try:
            response = supabase.table("questions").select("*").in_("id", baseline_question_ids_to_ask).execute()
            baseline_questions_details = response.data or []

            # Use helper to format for storage (or adapt store_single_followup_question)
            for q_data in baseline_questions_details:
                 # Ensure q_data has needed fields for store_single_followup_question
                 if store_single_followup_question(session_id, q_data):
                      baseline_stored_count += 1

            print(f"Stored {baseline_stored_count} baseline questions.")
        except Exception as e:
            print(f"Error retrieving/storing baseline questions: {e}")

    # --- 3. Wait for Baseline Answers ---
    if baseline_question_ids_to_ask and baseline_stored_count > 0:
        print("\n--- Waiting for Baseline Question Answers ---")
        # Wait specifically for the baseline IDs that were successfully stored
        # Get the actual stored baseline IDs for this session to be precise
        stored_baselines = supabase.table("followup_responses")\
            .select("question_id")\
            .eq("session_id", session_id)\
            .in_("question_id", baseline_question_ids_to_ask)\
            .execute().data or []
        ids_to_wait_for = [item['question_id'] for item in stored_baselines]
        if ids_to_wait_for:
             wait_for_batch_answers(session_id, ids_to_wait_for)
        else:
             print("Warning: Could not confirm stored baseline IDs to wait for.")
    else:
        print("No baseline questions to wait for.")

    # --- 4. Generate Initial Summaries (AFTER Baseline Answers) ---
    print("\n--- Generating Initial Context Summaries (Post-Baseline) ---")
    # Fetch answers again now that baseline should be answered
    all_followups_post_baseline = get_followup_responses(session_id, answered_only=False)
    all_answered_followups_post_baseline = [r for r in all_followups_post_baseline if r.get("answer") is not None]

   # Format Survey QA (Survey context doesn't change based on baseline answers)
    formatted_survey_qa = format_survey_qa(survey_questions, survey_responses)
    # Call LLM for Survey Summary
    survey_context_summary = call_llm(
        prompt=prompt_templates.SUMMARIZE_SURVEY_CONTEXT_PROMPT.format(
            survey_questions_answers=formatted_survey_qa
        )
        # No specific max_tokens needed unless default is too small
    ) or "Survey summary unavailable."

    # print("\nSurvey Context Summary:\n", survey_context_summary)

    # Baseline summary uses the now-answered baseline questions
    answered_baseline_responses = [r for r in all_answered_followups_post_baseline if r.get("question_id") in baseline_question_ids_to_ask]
    formatted_baseline_qa = format_followup_qa(answered_baseline_responses)
    # Call LLM for Baseline Summary
    baseline_summary = call_llm(
        prompt=prompt_templates.BASELINE_SUMMARY_PROMPT.format(
            baseline_questions_answers=formatted_baseline_qa
        )
    ) or "Baseline summary unavailable."
    # print("\nBaseline Summary:\n", baseline_summary)
    # Generate initial common context summary using the two summaries just generated
    initial_common_context_summary = call_llm(
        prompt=prompt_templates.COMMON_INITIAL_CONTEXT_SUMMARY_PROMPT.format(
            baseline_summary=baseline_summary, # Use result from above
            survey_context_summary=survey_context_summary # Use result from above
        )
        # No specific max_tokens needed
    ) or "Initial common context unavailable."
    print("\nInitial Common Context Summary (Post-Baseline Wait):\n", initial_common_context_summary) # Keep print

    # This initial_common_context_summary will be used as 'common_context_summary'
    # for the first iteration (current_iteration == 1)
    common_context_summary = initial_common_context_summary
    # --- 5. Track Overall State (Post-Baseline) ---
    stored_question_ids = [r.get("question_id") for r in all_followups_post_baseline if r.get("question_id") is not None]
    total_questions_asked_so_far = sum(1 for qid in stored_question_ids if qid not in BASELINE_QUESTION_IDS) # Should be 0 here
    print(f"Stored IDs after baseline: {stored_question_ids}")
    print(f"Non-Baseline Questions Asked after baseline: {total_questions_asked_so_far}")

    unspent_from_previous = 0 # Initialize unspent quota tracker

    # --- 6. Main Iteration Loop ---
    print("\n" + "="*20 + " Starting Iterative Questioning " + "="*20)
    for current_iteration in range(1, iteration_logic.MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {current_iteration}/{iteration_logic.MAX_ITERATIONS} ---")

        # --- 3a. Update States ---
        all_followups = get_followup_responses(session_id, answered_only=False) # Fetch latest
        all_answered_followups = [r for r in all_followups if r.get("answer") is not None]
        # Call the update function from the imported module
        fragment_states = iteration_logic.update_fragment_states(
            fragment_states,
            all_followups, # Pass the fetched data
            all_dependency_rules,
            all_questions_map,
            BASELINE_QUESTION_IDS,
            current_iteration,
            # iteration_logic.MAX_ITERATIONS # Constant is used inside the function now
        )
        # --- 3b. Determine Targets for This Iteration ---
        # First, get current pool sizes for active fragments
        fragment_pool_counts = {}
        dependency_excluded_ids = calculate_dependency_exclusions(all_dependency_rules, all_answered_followups)
        combined_exclude_ids_for_count = list(set(stored_question_ids + dependency_excluded_ids))
        print(f"DEBUG Iteration {current_iteration}: Combined Exclusions for Count RPC: {combined_exclude_ids_for_count}") # Debug print
        rpc_filter_for_count = category_filter.copy()
        rpc_filter_for_count["exclude_ids"] = combined_exclude_ids_for_count

        for frag_id, state in fragment_states.items():
             if not state['is_complete']:
                  try:
                      count_response = supabase.rpc(
                          "count_matching_documents_by_fragment",
                          {"fragment_filter": frag_id, "filter": rpc_filter_for_count}
                      ).execute()
                      fragment_pool_counts[frag_id] = count_response.data if count_response.data is not None else 0
                  except Exception as e:
                      print(f"Error calling count RPC for {frag_id}: {e}")
                      fragment_pool_counts[frag_id] = 0 # Assume 0 if error
                  print(f"  Fragment {frag_id}: Available Pool Count = {fragment_pool_counts.get(frag_id, 0)}")

        # Call the completion check function from the imported module
        if iteration_logic.check_if_assessment_complete(fragment_states, total_questions_asked_so_far, fragment_pool_counts):
            print("Ending iterations early.")
            break
        
        # Determine targets based on maturity, pool, quota, and previous unspent
        iteration_targets, total_allocated = iteration_logic.determine_iteration_targets(
            fragment_states,
            #iteration_logic.QUESTIONS_PER_ITERATION,
            fragment_pool_counts,
            unspent_from_previous,
            current_iteration
        )

        if total_allocated == 0:
            print("No questions allocated for this iteration. Checking if assessment is complete or pools exhausted.")
            # Re-check completion based on zero allocation (might indicate pool exhaustion)
            pools_exhausted = all(fragment_pool_counts.get(f, 0) == 0 for f, s in fragment_states.items() if not s['is_complete'])
            if pools_exhausted:
                 print("All active fragment pools appear exhausted. Ending iterations.")
                 break
            else:
                 print("Pools still available but no target allocated (check logic). Ending iterations for safety.")
                 break # Avoid potential infinite loop if something is wrong

        # --- 3c. Generate Common Context (Switches after Iteration 1) ---
        if current_iteration > 1:
            print("Generating updated common context...")
            fragment_descriptors = "\n".join([f"- {f_id}: {fragment_info_map.get(f_id, {}).get('description', 'N/A')}" for f_id in all_fragments_list])
            # Use status summaries generated at the END of the previous iteration
            fragment_qa_summaries = "\n".join([f"Fragment {f_id} Status:\n{state.get('status_summary', 'No status summary generated yet.')}\n" for f_id, state in fragment_states.items()])

            common_context_summary = call_llm(
                prompt_templates.COMMON_CONTEXT_SUMMARY_PROMPT.format(
                    baseline_survey_summary=initial_common_context_summary, # Use the one from start
                    industry_context=industry_context,
                    fragment_descriptors=fragment_descriptors,
                    fragment_qa_summaries=fragment_qa_summaries
                )
            ) or "Common context update unavailable."
            #print("\nUpdated Common Context Summary:\n", common_context_summary)
        # Else: Use initial_common_context_summary generated before the loop

        print(f"@@@ ITERATION {current_iteration}: STARTING CANDIDATE GATHERING/PROCESSING @@@")

        # --- 3d. Fragment Processing: Retrieve, Rank, Select Candidates ---
        candidates_this_iteration = [] # List to hold {fragment: F, id: QID, rank: R, similarity: S}
        all_retrieved_this_iteration = {} # Dict {fragment: [retrieved_q_dicts]}
        shortfall_per_fragment = {} # Track how many questions couldn't be selected vs target

        # Get latest exclusions based on latest answers
        dependency_excluded_ids = calculate_dependency_exclusions(all_dependency_rules, all_answered_followups)
        combined_exclude_ids = list(set(stored_question_ids + dependency_excluded_ids))
        print(f"DEBUG Iteration {current_iteration}: Combined Exclusions for Retrieval: {combined_exclude_ids}")

        fragments_to_process = [f for f, target in iteration_targets.items() if target > 0]
        print(f"Processing fragments with targets: {fragments_to_process}")
        for current_fragment in fragments_to_process:
            target_count = iteration_targets[current_fragment]
            print(f"\n  Processing Fragment: {current_fragment} (Target: {target_count})")
            # (Get fragment descriptor - already have from fragment_info_map)
            ##########################second iteration error occurs here###############################

            fragment_descriptor = fragment_info_map.get(current_fragment, {}).get("description", "N/A")

            # (Generate fragment QA summary & pre-retrieval summary - Keep this logic)
            # Use latest answers all_answered_followups for QA summary
            frag_answered_q_details = get_question_details_by_ids([q['question_id'] for q in all_answered_followups])
            fragment_followup_responses = []
            for resp in all_answered_followups:
                q_detail = frag_answered_q_details.get(resp['question_id'])
                if q_detail and all_questions_map.get(resp['question_id']) == current_fragment:
                    if 'question' not in resp or not resp['question']: resp['question'] = q_detail.get('question', 'N/A')
                    fragment_followup_responses.append(resp)
            formatted_fragment_qa = format_followup_qa(fragment_followup_responses)
            fragment_qa_summary = call_llm(prompt_templates.FRAGMENT_QA_SUMMARY_PROMPT.format(fragment_descriptor=fragment_descriptor, fragment_questions_answers=formatted_fragment_qa)) or f"Frag {current_fragment} QA summary unavailable."
            #print(f"  Fragment {current_fragment} QA Summary:\n", fragment_qa_summary)
            ####################################################################################error from here
            final_pre_retrieval_summary = call_llm(
                 prompt_templates.FINAL_PRE_RETRIEVAL_PROMPT.format(
                     common_context_summary=common_context_summary, fragment_descriptor=fragment_descriptor,
                     fragment_qa_summary=fragment_qa_summary, industry_context=industry_context
                 ), max_tokens=1800
            ) or f"Pre-retrieval summary for {current_fragment} unavailable."
            #print(f"  Fragment {current_fragment} Pre-Retrieval Summary:\n", final_pre_retrieval_summary)

            if not final_pre_retrieval_summary or "unavailable" in final_pre_retrieval_summary:
                 print(f"Error: Cannot retrieve for {current_fragment} without pre-retrieval summary.")
                 shortfall_per_fragment[current_fragment] = target_count # Entire target becomes shortfall
                 continue

            # Embed and Retrieve
            query_embedding = embed_text(final_pre_retrieval_summary, task_type="SEMANTIC_SIMILARITY")
            if not query_embedding:
                 print(f"Error: Failed embedding for {current_fragment}. Skipping.")
                 shortfall_per_fragment[current_fragment] = target_count
                 continue

            current_rpc_filter = category_filter.copy()
            current_rpc_filter["exclude_ids"] = combined_exclude_ids
            retrieved_questions = []
            try:
                # Retrieve target + buffer (e.g., +3) to allow for ranking/fallback choices
                match_count_with_buffer = target_count + 3
                retrieval_response = supabase.rpc(
                     "match_documents_by_fragment",
                     {"query_embedding": query_embedding, "fragment_filter": current_fragment,
                      "filter": current_rpc_filter, "match_count": match_count_with_buffer}
                ).execute()
                retrieved_questions = retrieval_response.data or []
                print(f"  Retrieved {len(retrieved_questions)} candidates for {current_fragment} (target: {target_count}, retrieved up to: {match_count_with_buffer})")
                all_retrieved_this_iteration[current_fragment] = copy.deepcopy(retrieved_questions)
            except Exception as e:
                 print(f"Error retrieving for {current_fragment}: {e}")
                 all_retrieved_this_iteration[current_fragment] = []
                 shortfall_per_fragment[current_fragment] = target_count
                 continue

            if not retrieved_questions:
                 print(f"No questions retrieved for {current_fragment}.")
                 shortfall_per_fragment[current_fragment] = target_count
                 continue

            # Rank and Select TOP 'target_count' candidates
            formatted_retrieved = format_retrieved_questions(retrieved_questions)
            ranked_list_text = call_llm(
                 prompt_templates.POST_RETRIEVAL_FILTERING_PROMPT.format(
                     common_context_summary=common_context_summary, fragment_descriptor=fragment_descriptor,
                     fragment_qa_summary=fragment_qa_summary, industry_context=industry_context,
                     retrieved_questions=formatted_retrieved
                 )
             )
            #print(f"  LLM Ranking Output for {current_fragment}:\n", ranked_list_text or "Ranking failed.")

            retrieved_ids = [q['id'] for q in retrieved_questions]
            ranked_ids = parse_llm_ranking(ranked_list_text or "", retrieved_ids)

            selected_candidates_for_fragment = []
            if ranked_ids:
                 # Take top 'target_count' from LLM ranking
                 for rank, s_id in enumerate(ranked_ids):
                      if len(selected_candidates_for_fragment) < target_count:
                           # Find similarity for this ID
                           similarity = next((q['similarity'] for q in retrieved_questions if q['id'] == s_id), 0.0)
                           selected_candidates_for_fragment.append({
                                "fragment": current_fragment, "id": s_id, "rank": rank, "similarity": similarity
                           })
                      else: break # Reached target
                 print(f"  Selected {len(selected_candidates_for_fragment)} candidates via LLM rank for {current_fragment}.")
            else:
                 # Fallback: Take top 'target_count' based on similarity
                 print(f"  Warning: Using fallback (similarity) selection for {current_fragment}.")
                 for i, q in enumerate(retrieved_questions):
                      if len(selected_candidates_for_fragment) < target_count:
                           selected_candidates_for_fragment.append({
                                "fragment": current_fragment, "id": q['id'], "rank": -1, "similarity": q.get('similarity', 0.0)
                           })
                      else: break # Reached target
                 print(f"  Selected {len(selected_candidates_for_fragment)} candidates via fallback for {current_fragment}.")

            candidates_this_iteration.extend(selected_candidates_for_fragment)
            shortfall = target_count - len(selected_candidates_for_fragment)
            if shortfall > 0:
                 shortfall_per_fragment[current_fragment] = shortfall
                 print(f"  Shortfall for {current_fragment}: {shortfall}")

        # --- End of Fragment Processing Loop for this iteration ---
        print(f"\nDEBUG Iteration {current_iteration}: Raw candidates selected across fragments ({len(candidates_this_iteration)}):")
        print(f"  IDs only: {[c['id'] for c in candidates_this_iteration]}")

        print(f"@@@ ITERATION {current_iteration}: STARTING CANDIDATE GATHERING/PROCESSING @@@")

        # --- ADD DE-DUPLICATION STEP ---
        print(f"--- De-duplicating candidates for Iteration {current_iteration} ---")
        deduplicated_candidates_info = []
        seen_ids_this_iteration = set()
        for candidate in candidates_this_iteration:
            q_id = candidate['id']
            if q_id not in seen_ids_this_iteration:
                deduplicated_candidates_info.append(candidate)
                seen_ids_this_iteration.add(q_id)
            else:
                print(f"  DEBUG: Removing duplicate candidate ID {q_id} from fragment {candidate['fragment']}")
        
        # Use the deduplicated list going forward
        candidates_this_iteration = deduplicated_candidates_info
        print(f"DEBUG: Deduplicated candidates ({len(candidates_this_iteration)}): {[c['id'] for c in candidates_this_iteration]}")
        # --- END DE-DUPLICATION STEP ---
        # --- 3e. Calculate Unspent Quota for Next Iteration ---
        unspent_from_previous = sum(shortfall_per_fragment.values()) # Carry over total shortfall
        print(f"Total shortfall this iteration (unspent for next): {unspent_from_previous}")


        # --- 3f. Apply 'must_ask_first' Rule and Finalize Selection ---
        print(f"\n--- Applying 'must_ask_first' (No Co-occurrence) to {len(candidates_this_iteration)} candidates ---")
        # (Keep the revised 'must_ask_first' logic from the previous response here)
        # It operates on 'candidates_this_iteration' using 'all_retrieved_this_iteration' for fallbacks
        # and returns 'final_selection' (list of candidate dicts)
         # --- 4. Apply 'must_ask_first' Rule and Finalize Selection ---
        print("\n--- Applying 'must_ask_first' Rules (Revised: No Co-occurrence) ---") # Updated print

        must_ask_first_rules = [r for r in all_dependency_rules if r.get("rule_type") == "must_ask_first"]
        globally_excluded_ids = set(stored_question_ids + dependency_excluded_ids) # Combine all known exclusions

        final_selection = copy.deepcopy(candidates_this_iteration)
        if must_ask_first_rules and final_selection:
            made_change = True; loop_guard = 0; max_loops = len(final_selection) * len(must_ask_first_rules) + 5
            while made_change and loop_guard < max_loops:
                made_change = False; loop_guard += 1
                current_selected_ids = {c['id'] for c in final_selection}
                #print(f"DEBUG: Must-ask-first loop {loop_guard}. Current: {current_selected_ids}") # Verbose Debug
                rules_to_check = copy.deepcopy(must_ask_first_rules)
                for rule in rules_to_check:
                    dep_id = rule.get("dependent_question_id"); indep_id_str = rule.get("independent_question_id")
                    try: indep_id = int(indep_id_str.strip()) if indep_id_str else None
                    except ValueError: continue
                    if not dep_id or not indep_id: continue

                    if dep_id in current_selected_ids and indep_id in current_selected_ids:
                        print(f"INFO: 'must_ask_first' violation (Co-occurrence): Q {dep_id} and Q {indep_id}. Replacing Q {dep_id}.")
                        candidate_to_replace_index = -1
                        for i, c in enumerate(final_selection):
                            if c['id'] == dep_id: candidate_to_replace_index = i; break

                        if candidate_to_replace_index != -1:
                            fragment_of_replaced = final_selection[candidate_to_replace_index]['fragment']
                            fallback_options = all_retrieved_this_iteration.get(fragment_of_replaced, [])
                            replacement_found = False
                            for fallback_q in fallback_options:
                                fallback_id = fallback_q.get('id')
                                if fallback_id is None: continue
                                is_valid = (fallback_id != dep_id and fallback_id != indep_id and
                                            fallback_id not in current_selected_ids and fallback_id not in globally_excluded_ids)
                                if is_valid:
                                    print(f"INFO: Replacing Q {dep_id} with fallback Q {fallback_id}")
                                    final_selection[candidate_to_replace_index] = {
                                        "fragment": fragment_of_replaced, "id": fallback_id,
                                        "rank": -1, "similarity": fallback_q.get('similarity', 0.0) }
                                    replacement_found = True; made_change = True; break
                            if not replacement_found:
                                print(f"WARNING: No valid fallback for Q {dep_id}. Removing.")
                                final_selection.pop(candidate_to_replace_index); made_change = True
                        if made_change: break # Restart rule check if change was made
            if loop_guard >= max_loops: print("WARNING: Max loops for 'must_ask_first'.")
        print(f"Final selection for Iteration {current_iteration} ({len(final_selection)} questions): {[c['id'] for c in final_selection]}")


        # --- 3g. Store Final Questions & Update State ---
        final_stored_ids_this_iteration = [] # Holds IDs stored *in this specific iteration* for the wait function
        if final_selection:
            print(f"\n--- Storing {len(final_selection)} Final Questions for Iteration {current_iteration} ---")
            # Fetch details for all final questions at once for efficiency
            final_ids_to_fetch = [c['id'] for c in final_selection]
            final_q_details_map = get_question_details_by_ids(final_ids_to_fetch)

            for candidate in final_selection:
                selected_q_id = candidate['id']
                # frag_id = candidate['fragment'] # Not strictly needed for storage logic itself
                selected_q_data = final_q_details_map.get(selected_q_id)

                if selected_q_data:
                     # --- Check if already stored BEFORE attempting to store again ---
                     # This prevents errors if dependency logic selected an already stored question as fallback
                     if selected_q_id not in stored_question_ids:
                          # Attempt to store the question in the database
                          stored_successfully = store_single_followup_question(session_id, selected_q_data)

                          if stored_successfully:
                               # --- Updates happen ONLY on successful storage of a NEW question ---
                               print(f"  Successfully stored new question ID: {selected_q_id}")
                               # 1. Add to list for this iteration's batch wait
                               final_stored_ids_this_iteration.append(selected_q_id)
                               # 2. Add to the global list tracking all questions asked in the session
                               stored_question_ids.append(selected_q_id)
                               # 3. Increment total *non-baseline* count if applicable
                               if selected_q_id not in BASELINE_QUESTION_IDS:
                                    total_questions_asked_so_far += 1
                                    print(f"  Updated total_questions_asked_so_far to: {total_questions_asked_so_far}")
                               # --- End Updates ---
                          else:
                               # Storage failed (e.g., DB error other than duplicate)
                               print(f"Error: Failed to store final question {selected_q_id} (check store_single_followup_question logs).")
                               # Decide if you need specific error handling here
                     else:
                          # Question was selected but is already in the global stored list
                          print(f"INFO: Question {selected_q_id} was in final selection but already stored in this session. Skipping storage.")
                          # Optional: Add to final_stored_ids_this_iteration if you still need to wait for it?
                          # If it was stored previously but unanswered, it might need waiting.
                          # Let's assume for now we only wait for *newly* stored ones.
                else:
                    # Failed to fetch question details - shouldn't happen if selection logic is sound
                    print(f"Error: Could not fetch details for final selected question ID {selected_q_id}. Cannot store.")
        else:
             # The must_ask_first logic resulted in an empty final selection
             print("No questions selected in the final list for this iteration. Nothing to store.")

        # --- Make sure the erroneous print marker that was here is REMOVED ---
        # --- 3h. Generate Fragment Status Summaries for Next Iteration ---
        # Generate summaries only for fragments that had questions selected in this iteration's final list
        fragments_in_final_selection = {c['fragment'] for c in final_selection}
        if fragments_in_final_selection:
             print("\n--- Generating Fragment Status Summaries for Next Iteration ---")
             # Get latest answers again after potentially waiting
             latest_answered_followups = get_followup_responses(session_id, answered_only=True)
             latest_answered_q_details = get_question_details_by_ids([q['question_id'] for q in latest_answered_followups])

             for frag_id in fragments_in_final_selection:
                  frag_descriptor = fragment_info_map.get(frag_id, {}).get("description", "N/A")
                  # Get QA summary reflecting answers up to end of this iteration for this fragment
                  frag_resp_for_summary = []
                  for resp in latest_answered_followups:
                       q_detail = latest_answered_q_details.get(resp['question_id'])
                       if q_detail and all_questions_map.get(resp['question_id']) == frag_id:
                            if 'question' not in resp or not resp['question']: resp['question'] = q_detail.get('question', 'N/A')
                            frag_resp_for_summary.append(resp)
                  formatted_qa_for_status = format_followup_qa(frag_resp_for_summary)
                  frag_qa_summary_for_status = call_llm(prompt_templates.FRAGMENT_QA_SUMMARY_PROMPT.format(fragment_descriptor=frag_descriptor, fragment_questions_answers=formatted_qa_for_status)) or f"Frag {frag_id} QA summary unavailable."

                  frag_status_summary = call_llm(
                       prompt_templates.FRAGMENT_STATUS_SUMMARY_PROMPT.format(
                           fragment_descriptor=frag_descriptor,
                           fragment_qa_summary=frag_qa_summary_for_status
                       )
                  )
                  if frag_status_summary:
                       fragment_states[frag_id]["status_summary"] = frag_status_summary
                       print(f"  Generated Status Summary for Fragment {frag_id}.")
                  else:
                       print(f"  Failed to generate status summary for Fragment {frag_id}.")

        # --- 3i. Wait for User Answers for the Batch ---
        if final_stored_ids_this_iteration:
            wait_for_batch_answers(session_id, final_stored_ids_this_iteration)
        else:
             print("No new questions were stored in this iteration, skipping wait.")

        # --- Check max questions limit ---
        if total_questions_asked_so_far >= iteration_logic.MAX_TOTAL_QUESTIONS:
             print(f"Reached maximum total questions limit ({iteration_logic.MAX_TOTAL_QUESTIONS}). Ending iterations.")
             break

        # --- End of Iteration ---

    # --- 4. Pipeline End ---
    print(f"\n{'='*20} RAG Pipeline Completed for Session: {session_id} {'='*20}")
    print(f"Total Non-Baseline Questions Asked: {total_questions_asked_so_far}")
    print("Final Fragment States:")
    for frag_id, state in fragment_states.items():
         print(f"  Fragment {frag_id}: Quota {state['quota_fulfilled']}/{state['quota_assigned']}, Complete: {state['is_complete']}")
    return True


# --- Execution Guard ---
if __name__ == "_main_":
    # (Keep the _main_ block with the test session ID and baseline check)
    test_session_id = "6262805b-ed71-480f-bb4f-23a14d65d6f7" # Use a real session ID for testing
    print(f"--- Running Pre-check/Setup for Session {test_session_id} ---")
    baseline_responses = get_followup_responses(test_session_id, answered_only=True, question_ids=BASELINE_QUESTION_IDS)
    if len(baseline_responses) < len(BASELINE_QUESTION_IDS):
        print(f"Warning: Only {len(baseline_responses)} answered baseline questions found for session {test_session_id}. Baseline summary might be incomplete.")
    else:
        print("Baseline questions appear to be answered.")
    print("--- Starting Pipeline Execution ---")
    run_iterative_rag_pipeline(test_session_id)