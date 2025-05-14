import json
import time
import re
import copy
import sys, os
from collections import deque
from typing import Dict, Any, List, Optional, Tuple 
from supabase import create_client, Client
import google.generativeai as genai
import google.api_core.exceptions

# --- Project Path Setup ---
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.utils import config 
from . import prompt_templates 
from . import iteration_logic 

# --- Supabase and Gemini API Configuration ---
SUPABASE_URL = config.SUPABASE_URL
SUPABASE_KEY = config.SUPABASE_KEY
GOOGLE_API_KEY = config.GOOGLE_API_KEY 

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# --- Model and Embedding Configuration ---
LLM_MODEL_NAME = "gemma-3-27b-it"
EMBEDDING_MODEL = "models/text-embedding-004"

# --- Baseline Question ID Sets ---
BASELINE_IDS_GWS = [59, 61, 64, 65, 33, 4, 25]
BASELINE_IDS_GWIS = [287, 120, 126, 212, 178, 169, 171, 237, 264, 358]

# --- Rate Limiting Globals ---
LLM_CALL_TIMESTAMPS = deque()
MAX_RPM = 30 
TARGET_RPM = MAX_RPM - 3 
RPM_WINDOW_SECONDS = 60
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2

# --- Utility Functions ---

def call_llm(prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.4) -> Optional[str]:
    """
    Calls the configured Gemini model with rate limiting (RPM) and retry logic.
    """
    global LLM_CALL_TIMESTAMPS

    current_time = time.monotonic()

    # RPM Rate Limiting
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

    LLM_CALL_TIMESTAMPS.append(current_time)

    backoff_time = INITIAL_BACKOFF_SECONDS
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Calling LLM ({LLM_MODEL_NAME})...")
            llm = genai.GenerativeModel(LLM_MODEL_NAME)
            gen_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens if max_tokens is not None else 1024
            )
            response = llm.generate_content(prompt, generation_config=gen_config)

            if response.parts:
                estimated_input_tokens = len(prompt) / 4
                estimated_output_tokens = len(response.text) / 4
                print(f"LLM Call Success. Estimated Tokens: In={estimated_input_tokens:.0f}, Out={estimated_output_tokens:.0f}")
                return response.text.strip()
            elif response.prompt_feedback.block_reason:
                print(f"Warning: LLM call blocked. Reason: {response.prompt_feedback.block_reason}")
                return None 
            else:
                print("Warning: LLM returned empty response (no parts, no block reason).")
                return None 

        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"Error: ResourceExhausted (likely rate limit or quota issue) on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Waiting {backoff_time:.2f} seconds before retry...")
                time.sleep(backoff_time)
                backoff_time *= 2 
            else:
                print("Max retries reached after ResourceExhausted error.")
                return None
        except Exception as e:
            print(f"Error during LLM call on attempt {attempt + 1}: {e}")
            if "API key not valid" in str(e) or "permission" in str(e).lower():
                print("Critical: Invalid Google API Key or insufficient permissions. Aborting retry.")
                return None
            if attempt < MAX_RETRIES - 1:
                print(f"Waiting {backoff_time:.2f} seconds before retry for general error...")
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                print("Max retries reached after general error.")
                return None
    return None 

def embed_text(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> Optional[List[float]]:
    """Generates embeddings for the given text using the Gemini API."""
    if not text or not text.strip(): 
        print("Warning: Attempted to embed empty or whitespace-only text.")
        return None
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding for text snippet '{text[:50]}...': {e}")
        return None

def build_category_filter(survey_responses: Optional[List[Dict[str, Any]]]) -> dict:
    """
    Determines the assessment category based on Q16 survey response.
    """
    category_filter = {}
    q16_answer_text = "Not answered or not found"

    if isinstance(survey_responses, list):
        for resp in survey_responses:
            if str(resp.get("question_id")) == "Q16":
                ans_val_raw = resp.get("answer")
                ans_val_str = None
                if isinstance(ans_val_raw, dict): 
                    ans_val_str = str(ans_val_raw.get("value")) if ans_val_raw.get("value") is not None else None
                elif ans_val_raw is not None: 
                    ans_val_str = str(ans_val_raw)

                if ans_val_str == "1":
                    q16_answer_text = "Green With Software (Value: 1)"
                    category_filter = {"category": "Green With Software"}
                    break
                elif ans_val_str == "2":
                    q16_answer_text = "Green Within Software (Value: 2)"
                    category_filter = {"category": "Green Within Software"}
                    break
                else:
                    q16_answer_text = f"Q16 answered with unexpected value: '{ans_val_str}'"
    else:
        q16_answer_text = "Survey responses format not a list or is None."

    print(f"Category filter determination based on Q16 ('{q16_answer_text}'): {category_filter}")
    return category_filter

def get_survey_questions() -> List[Dict[str, Any]]:
    """Fetches all survey questions (Q1, Q2 etc.) from Supabase."""
    try:
        response = supabase.table("survey_questions").select("*").execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching survey_questions: {e}")
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
            responses_json_data = response.data[0].get("responses")
            if isinstance(responses_json_data, list):
                return responses_json_data
            elif isinstance(responses_json_data, str): 
                try:
                    return json.loads(responses_json_data)
                except json.JSONDecodeError:
                    print(f"Error decoding survey_responses JSON string for session {session_id}")
                    return None
            else:
                print(f"Unexpected format for survey_responses content: {type(responses_json_data)}")
                return None
        print(f"No survey_responses found for session {session_id}")
        return None
    except Exception as e:
        print(f"Error fetching survey_responses for session {session_id}: {e}")
        return None

def get_followup_responses(session_id: str, answered_only: bool = False, question_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Fetches follow-up responses, optionally filtering by answered status or specific question IDs."""
    try:
        query = supabase.table("followup_responses").select("*").eq("session_id", session_id)
        if answered_only:
            query = query.not_.is_("answer", None)
        if question_ids:
            cleaned_q_ids = [int(qid) for qid in question_ids if qid is not None]
            if cleaned_q_ids:
                query = query.in_("question_id", cleaned_q_ids)
            else: 
                if question_ids is not None: 
                    return []
        response = query.execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching followup_responses for session {session_id}: {e}")
        return []

def get_question_details_by_ids(question_ids: List[Any]) -> Dict[int, Dict[str, Any]]:
    """Fetches full question details from the 'questions' table for a list of IDs."""
    if not question_ids:
        return {}
    cleaned_ids = []
    for qid in question_ids:
        try:
            cleaned_ids.append(int(qid))
        except (ValueError, TypeError):
            print(f"Warning: Could not convert question ID '{qid}' to int. Skipping.")
    
    if not cleaned_ids:
        return {}
        
    try:
        response = supabase.table("questions").select("*").in_("id", cleaned_ids).execute()
        return {q['id']: q for q in response.data} if response.data else {}
    except Exception as e:
        print(f"Error fetching question details for IDs {cleaned_ids}: {e}")
        return {}

def get_all_fragments_info_from_db() -> List[Dict[str, Any]]:
    """
    Fetches all fragment identifiers and their descriptions from the fragment_info table.
    """
    try:
        response = supabase.table("fragment_info").select("fragment, description").execute()
        if response.data:
            return response.data 
        else:
            print("Warning: fragment_info table might be empty or not found.")
            return []
    except Exception as e:
        print(f"Error fetching all fragment info from DB: {e}")
        return []

def get_fragment_info_from_list(fragment_id: str, all_fragments_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Retrieves info for a specific fragment_id from a pre-fetched list of all fragment data.
    """
    for item in all_fragments_data:
        if item.get('fragment') == fragment_id:
            return item
    print(f"Warning: Info for fragment_id '{fragment_id}' not found in the provided list.")
    return None

def get_industry_context(survey_responses: Optional[List[Dict[str, Any]]]) -> str:
    """Extracts industry context from survey response Q17."""
    if not survey_responses:
        return "Not specified"
    for resp in survey_responses:
        if str(resp.get("question_id")) == "Q17": 
            answer_raw = resp.get("answer")
            if isinstance(answer_raw, dict):
                return str(answer_raw.get("value", "Not specified"))
            elif answer_raw is not None:
                return str(answer_raw)
            else:
                return "Not specified" 
    return "Not specified" 

def get_dependency_rules() -> List[Dict[str, Any]]:
    """Fetches all question dependency rules from Supabase."""
    try:
        response = supabase.table("question_dependency_rules").select("*").execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching question_dependency_rules: {e}")
        return []

def extract_answer_value(answer_jsonb: Optional[Any]) -> Optional[str]: 
    """Safely extracts the 'value' from various answer structures."""
    if isinstance(answer_jsonb, dict):
        if "value" in answer_jsonb and answer_jsonb["value"] is not None:
            return str(answer_jsonb["value"])
        if "content" in answer_jsonb and answer_jsonb["content"] is not None: 
            return str(answer_jsonb["content"])
    elif answer_jsonb is not None: 
        return str(answer_jsonb)
    return None 

def calculate_dependency_exclusions(rules: List[Dict[str, Any]], answered_responses: List[Dict[str, Any]]) -> List[int]:
    """
    Calculates which dependent question IDs should be excluded based on 'must_not_ask' rules.
    """
    must_not_ask_rules = [r for r in rules if r.get("rule_type") == "must_not_ask"]
    if not must_not_ask_rules:
        return []

    answered_map = { 
        int(resp['question_id']): (extract_answer_value(resp.get('answer')) or "").lower()
        for resp in answered_responses 
        if resp.get('question_id') is not None and resp.get('answer') is not None
    }
    dependency_excluded_ids = set()

    for rule in must_not_ask_rules:
        try:
            dep_id = int(rule.get("dependent_question_id"))
            indep_id_str = rule.get("independent_question_id") 
            condition_values_str = rule.get("condition_answer_values") 
            condition_type = rule.get("condition_type") 
            condition_logic = rule.get("condition_logic") 
        except (ValueError, TypeError):
            continue

        if not dep_id or not indep_id_str or not condition_values_str or not condition_type or not condition_logic:
            continue

        try:
            indep_ids_list = [int(i.strip()) for i in indep_id_str.split(';') if i.strip()]
            condition_values_list = [v.strip().lower() for v in condition_values_str.split(';') if v.strip()]
        except ValueError:
            continue

        if not indep_ids_list or not condition_values_list:
            continue
        
        num_answered_prereqs = sum(1 for i_id in indep_ids_list if i_id in answered_map)
        prereqs_met_for_rule_evaluation = False
        if condition_logic == 'match_all': 
            if num_answered_prereqs == len(indep_ids_list):
                prereqs_met_for_rule_evaluation = True
        elif condition_logic == 'match_any' or condition_logic == 'none': 
            if num_answered_prereqs > 0:
                prereqs_met_for_rule_evaluation = True
        
        if not prereqs_met_for_rule_evaluation:
            continue

        condition_fulfilled_to_exclude = False
        individual_match_results = [] 
        
        answered_relevant_indep_ids = [i_id for i_id in indep_ids_list if i_id in answered_map]
        if not answered_relevant_indep_ids: 
            continue

        for i_id in answered_relevant_indep_ids:
            actual_answer_lower = answered_map.get(i_id) 
            current_indep_match = False
            if condition_type == 'exact_match': 
                if actual_answer_lower in condition_values_list:
                    current_indep_match = True
            elif condition_type == 'match_any': 
                if actual_answer_lower in condition_values_list:
                    current_indep_match = True
            individual_match_results.append(current_indep_match)

        if not individual_match_results: 
            continue

        if condition_logic == 'match_all': 
            if all(individual_match_results):
                condition_fulfilled_to_exclude = True
        elif condition_logic == 'match_any' or condition_logic == 'none': 
            if any(individual_match_results):
                condition_fulfilled_to_exclude = True
        
        if condition_fulfilled_to_exclude:
            dependency_excluded_ids.add(dep_id)
            
    return list(dependency_excluded_ids)

def format_survey_qa(survey_questions: List[Dict[str, Any]], survey_responses: List[Dict[str, Any]]) -> str:
    """Formats survey questions and their answers for the LLM prompt."""
    qa_string = ""
    response_map = {}
    if survey_responses:
        for resp_item in survey_responses: 
            q_id_from_resp = str(resp_item.get("question_id")) 
            ans_raw = resp_item.get("answer")
            ans_str = extract_answer_value(ans_raw) if ans_raw is not None else "No answer"
            response_map[q_id_from_resp] = ans_str

    for q_data in survey_questions: 
        q_id_str = str(q_data.get("question_id")) 
        q_content = q_data.get("content", "N/A")
        q_options_list = q_data.get("options") 

        user_answer_final_text = response_map.get(q_id_str, "No answer")

        if q_options_list and isinstance(q_options_list, list) and not q_data.get("subjective_answer"):
            current_ans_values_from_map = []
            raw_ans_from_map = response_map.get(q_id_str)

            if raw_ans_from_map and raw_ans_from_map != "No answer":
                try: 
                    parsed_ans = json.loads(raw_ans_from_map)
                    if isinstance(parsed_ans, list):
                        current_ans_values_from_map = [str(v) for v in parsed_ans]
                    else: 
                        current_ans_values_from_map = [str(parsed_ans)]
                except json.JSONDecodeError: 
                    current_ans_values_from_map = [raw_ans_from_map]
            
            mapped_option_texts = []
            if current_ans_values_from_map:
                for opt_dict in q_options_list:
                    opt_val_letter = str(opt_dict.get("option_letter")) 
                    opt_content_text = opt_dict.get("content", "")
                    if opt_val_letter in current_ans_values_from_map:
                        mapped_option_texts.append(opt_content_text)
            
            if mapped_option_texts:
                user_answer_final_text = ", ".join(mapped_option_texts)
            elif raw_ans_from_map and raw_ans_from_map != "No answer":
                 user_answer_final_text = raw_ans_from_map

        qa_string += f"Question ({q_id_str}): {q_content}\n"
        if q_options_list and isinstance(q_options_list, list):
            options_display_text = ", ".join([f"{opt.get('option_letter', '?')}: {opt.get('content', '')}" for opt in q_options_list])
            qa_string += f"Options: [{options_display_text}]\n"
        qa_string += f"User Answer: {user_answer_final_text}\n\n"

    return qa_string.strip() if qa_string else "No survey questions or answers provided."

def format_followup_qa(followup_responses: List[Dict[str, Any]]) -> str:
    """Formats answered follow-up questions and their answers for the LLM prompt."""
    qa_string = ""
    if not followup_responses:
        return "No previous follow-up questions answered in this context yet."

    for resp_data in followup_responses:
        if resp_data.get("answer") is not None:
            q_text = resp_data.get("question", "N/A") 
            q_id = resp_data.get("question_id", "N/A")
            raw_answer_data = resp_data.get("answer") 
            
            answer_display_text = extract_answer_value(raw_answer_data) or "N/A"

            additional_fields = resp_data.get("additional_fields", {}) if isinstance(resp_data.get("additional_fields"), dict) else {}
            options_list = None
            if "answer_options" in additional_fields and isinstance(additional_fields["answer_options"], list):
                options_list = additional_fields["answer_options"]
            elif "multiple_correct_answer_options" in additional_fields and isinstance(additional_fields["multiple_correct_answer_options"], list):
                options_list = additional_fields["multiple_correct_answer_options"]

            if options_list and answer_display_text != "N/A":
                current_ans_values_as_strings = []
                try:
                    parsed_ans = json.loads(answer_display_text) 
                    if isinstance(parsed_ans, list):
                        current_ans_values_as_strings = [str(v) for v in parsed_ans]
                    else: 
                        current_ans_values_as_strings = [str(parsed_ans)]
                except json.JSONDecodeError: 
                    current_ans_values_as_strings = [answer_display_text]

                mapped_options_text_list = []
                for opt_item in options_list: 
                    opt_val_str = None
                    opt_content_str = None
                    if isinstance(opt_item, dict): 
                        opt_val_str = str(opt_item.get("value")) 
                        opt_content_str = opt_item.get("content", "")
                    elif isinstance(opt_item, str):
                        # If the option item is just a string, assume it's both the value and content
                        opt_val_str = opt_item
                        opt_content_str = opt_item
                        # No warning needed here as we are now handling it.
                    else:
                        # Log a warning if an option item is not a dictionary or string
                        print(f"Warning (format_followup_qa): Encountered unexpected option type for QID {q_id}. Option item: '{opt_item}'")
                        continue # Skip this malformed option

                    if opt_val_str is not None and opt_val_str in current_ans_values_as_strings:
                        if opt_content_str: # Ensure content is not None or empty before adding
                           mapped_options_text_list.append(opt_content_str)
                        elif opt_val_str: # Fallback to value if content is empty but value matched
                           mapped_options_text_list.append(opt_val_str)

                if mapped_options_text_list:
                    answer_display_text = ", ".join(mapped_options_text_list)
            
            qa_string += f"Question (ID {q_id}): {q_text}\n"
            qa_string += f"User Answer: {answer_display_text}\n\n"

    return qa_string.strip() if qa_string.strip() else "No relevant answered follow-up questions found."

def format_retrieved_questions(retrieved_questions: List[Dict[str, Any]]) -> str:
    """Formats retrieved candidate questions for the ranking prompt."""
    q_string = ""
    for q_data in retrieved_questions:
        q_id = q_data.get("id", "N/A")
        q_text = q_data.get("question", "N/A") 
        q_string += f"- [{q_id}] {q_text}\n"
    return q_string.strip() if q_string else "No questions retrieved."

def parse_llm_ranking(ranked_text: str, retrieved_ids: List[Any]) -> List[int]:
    """Parses the LLM's ranked list and returns ordered IDs."""
    if not ranked_text:
        print("Warning: LLM ranking text is empty. Cannot parse.")
        return []

    ordered_ids = []
    pattern = re.compile(r"^\s*\d+\.\s*\[(\d+)\]", re.MULTILINE)
    matches = pattern.findall(ranked_text)

    seen_ids = set()
    valid_retrieved_ids_set = set()
    for r_id in retrieved_ids:
        try:
            valid_retrieved_ids_set.add(int(r_id))
        except (ValueError, TypeError):
            pass 

    for match_id_str in matches:
        try:
            q_id = int(match_id_str)
            if q_id in valid_retrieved_ids_set and q_id not in seen_ids:
                ordered_ids.append(q_id)
                seen_ids.add(q_id)
        except ValueError:
            print(f"Warning: Could not parse ID '{match_id_str}' from LLM ranking. Skipping.")

    if not ordered_ids:
        print(f"Warning: Failed to parse any valid IDs from the LLM ranking output. LLM Output:\n'{ranked_text[:500]}'")
    return ordered_ids

def store_single_followup_question(session_id: str, question_data_full: Dict[str, Any]) -> bool:
    """
    Stores a single selected follow-up question into Supabase 'followup_responses' table.
    """
    try:
        q_id = question_data_full.get("id")
        if q_id is None: 
            print("Error: Question data missing ID for storage.")
            return False

        orig_additional_fields = question_data_full.get("additional_fields")
        interactive_fields_for_storage = {}
        if isinstance(orig_additional_fields, dict):
            if "guidelines" in orig_additional_fields:
                interactive_fields_for_storage["guidelines"] = orig_additional_fields["guidelines"]
            if "answer_options" in orig_additional_fields: 
                interactive_fields_for_storage["answer_options"] = orig_additional_fields["answer_options"]
            elif "multiple_correct_answer_options" in orig_additional_fields: 
                interactive_fields_for_storage["multiple_correct_answer_options"] = orig_additional_fields["multiple_correct_answer_options"]
            if "subjective_answer" in orig_additional_fields:
                 interactive_fields_for_storage["subjective_answer"] = orig_additional_fields["subjective_answer"]

        record_to_insert = {
            "session_id": session_id,
            "question_id": int(q_id), 
            "question": question_data_full.get("question"), 
            "category": question_data_full.get("category"), 
            "subcategory": question_data_full.get("subcategory"), 
            "additional_fields": interactive_fields_for_storage, 
            "answer": None, 
            "fragment_code": question_data_full.get("fragments") 
        }

        response = supabase.table("followup_responses").insert(record_to_insert).execute()
        
        if (hasattr(response, 'data') and response.data) or \
           (hasattr(response, 'status_code') and 200 <= response.status_code < 300):
            print(f"Stored follow-up question ID: {q_id} for session {session_id}")
            return True
        else:
            error_info = getattr(response, 'error', "Unknown error (no data/error in response)")
            print(f"Error storing follow-up QID {q_id}: {error_info}")
            return False

    except Exception as e:
        q_id_for_error = question_data_full.get('id', 'N/A')
        print(f"Exception storing follow-up question ID {q_id_for_error}: {e}")
        if "duplicate key value violates unique constraint" in str(e).lower(): 
            print(f"Info: Question {q_id_for_error} (session: {session_id}) likely already exists in followup_responses.")
            return False 
        return False

def wait_for_batch_answers(session_id: str, expected_question_ids: List[Any]):
    """
    Waits until all questions in the provided list have non-null answers.
    """
    if not expected_question_ids:
        print("No questions provided to wait for batch answers.")
        return

    expected_q_ids_int_set = set()
    for qid in expected_question_ids:
        try:
            expected_q_ids_int_set.add(int(qid))
        except (ValueError, TypeError):
            print(f"Warning: Invalid question ID '{qid}' in wait_for_batch_answers. Skipping it.")
    
    if not expected_q_ids_int_set:
        print("No valid integer question IDs to wait for after cleaning.")
        return

    print(f"Waiting for user to answer batch of {len(expected_q_ids_int_set)} questions: {list(expected_q_ids_int_set)}...")
    
    answered_ids_set = set()
    max_wait_loops = 120  
    current_loop = 0

    while answered_ids_set != expected_q_ids_int_set and current_loop < max_wait_loops:
        current_loop += 1
        remaining_ids_to_check = list(expected_q_ids_int_set - answered_ids_set)
        
        try:
            response = supabase.table("followup_responses") \
                .select("question_id, answer") \
                .eq("session_id", session_id) \
                .in_("question_id", remaining_ids_to_check) \
                .execute()

            if response.data:
                for row in response.data:
                    if row.get('answer') is not None: 
                        answered_ids_set.add(int(row['question_id']))
            
            if answered_ids_set == expected_q_ids_int_set:
                print("All questions in the batch have been answered. Proceeding...")
                return 

            if current_loop % 4 == 0: 
                 print(f"  Waiting... {len(expected_q_ids_int_set - answered_ids_set)} questions remaining (Attempt {current_loop}/{max_wait_loops}).")
            time.sleep(15)  

        except Exception as e:
            print(f"Error checking answers for batch {remaining_ids_to_check}: {e}. Retrying after delay.")
            time.sleep(20) 

    if answered_ids_set != expected_q_ids_int_set:
        unanswered_count = len(expected_q_ids_int_set - answered_ids_set)
        print(f"Timeout or max attempts reached waiting for answers. {unanswered_count} questions still unanswered.")

# --- Main RAG Pipeline ---

def run_iterative_rag_pipeline(session_id: str) -> bool:
    """
    Main orchestrator for the iterative RAG assessment pipeline.
    """
    print(f"\n{'='*20} Unified RAG Pipeline Started for Session: {session_id} {'='*20}")

    # --- 1. Initial Data Fetching & Configuration Setup ---
    print("\n--- Step 1: Fetching Initial Data & Setup ---")
    survey_questions_raw_list = get_survey_questions()
    survey_responses_list = get_survey_responses(session_id)
    all_db_fragments_info_list = get_all_fragments_info_from_db() 
    all_dependency_rules_list = get_dependency_rules()
    
    all_questions_raw_data_for_map = supabase.table("questions").select("id, fragments").execute().data or []
    all_questions_id_to_fragment_map = iteration_logic.get_fragment_question_map(all_questions_raw_data_for_map)

    if not survey_questions_raw_list or not survey_responses_list or not all_db_fragments_info_list:
        print("CRITICAL ERROR: Failed to fetch essential initial data. Aborting pipeline.")
        return False
    if not all_questions_id_to_fragment_map:
        print("CRITICAL ERROR: Failed to build question_id_to_fragment_map. Aborting.")
        return False

    industry_context_text = get_industry_context(survey_responses_list)
    category_filter_from_q16 = build_category_filter(survey_responses_list) 
    
    assessment_category_name = category_filter_from_q16.get("category")
    active_baseline_ids: List[int]
    active_fragment_quotas: Dict[str, int]
    active_offload_priority: Dict[str, int]
    active_target_fragments_list: List[str] 
    assessment_type_log_message: str

    if assessment_category_name == "Green Within Software":
        assessment_type_log_message = "Green Within Software (GWIS)"
        active_baseline_ids = BASELINE_IDS_GWIS
        active_fragment_quotas = iteration_logic.FRAGMENT_QUOTAS_GWIS
        active_offload_priority = iteration_logic.OFFLOAD_PRIORITY_GWIS
        active_target_fragments_list = iteration_logic.FRAGMENTS_GWIS
    elif assessment_category_name == "Green With Software":
        assessment_type_log_message = "Green With Software (GWS)"
        active_baseline_ids = BASELINE_IDS_GWS
        active_fragment_quotas = iteration_logic.FRAGMENT_QUOTAS_GWS
        active_offload_priority = iteration_logic.OFFLOAD_PRIORITY_GWS
        active_target_fragments_list = iteration_logic.FRAGMENTS_GWS
    else:
        print(f"Warning: Unknown or no assessment category ('{assessment_category_name}') from Q16. Defaulting to GWS settings.")
        assessment_type_log_message = "Defaulting to Green With Software (GWS) due to undefined category"
        active_baseline_ids = BASELINE_IDS_GWS
        active_fragment_quotas = iteration_logic.FRAGMENT_QUOTAS_GWS
        active_offload_priority = iteration_logic.OFFLOAD_PRIORITY_GWS
        active_target_fragments_list = iteration_logic.FRAGMENTS_GWS

    print(f"Assessment Type Determined: {assessment_type_log_message}")

    current_assessment_fragment_info_map = {
        item['fragment']: item 
        for item in all_db_fragments_info_list 
        if item.get('fragment') in active_target_fragments_list
    }
    for frag_code_check in active_target_fragments_list:
        if frag_code_check not in current_assessment_fragment_info_map:
            print(f"CRITICAL ERROR: Fragment code '{frag_code_check}' from active list "
                  f"is missing in fetched fragment_info_map. Aborting.")
            return False

    fragment_states_map = {
        frag_id_init: {
            "quota_assigned": active_fragment_quotas.get(frag_id_init, active_fragment_quotas.get('DEFAULT_QUOTA', 0)),
            "quota_fulfilled": 0, "is_complete": False, "questions_asked_in_fragment": 0,
            "estimated_dependency_count": 0, "answered_independent_questions": 0,
            "readiness_score": 0.0, "maturity_score": 0.0, 
            "status_summary": "No status summary generated yet." 
        }
        for frag_id_init in active_target_fragments_list
    }

    print(f"Industry Context: {industry_context_text}")
    print(f"Active Fragments for this assessment: {active_target_fragments_list}")
    print(f"Active Fragment Quotas: {{ {', '.join([f'{f}: {active_fragment_quotas.get(f)}' for f in active_target_fragments_list])} }}")
    print(f"Category filter for DB queries (from Q16): {category_filter_from_q16}")

    # --- 2. Determine and Store Baseline Questions ---
    print(f"\n--- Step 2: Determining & Storing Baseline Questions for {assessment_type_log_message} ---")
    baseline_question_ids_to_ask_list = active_baseline_ids 
    print(f"Selected baseline IDs: {baseline_question_ids_to_ask_list}")
        
    baseline_questions_actually_stored_count = 0
    if baseline_question_ids_to_ask_list:
        baseline_questions_details_map = get_question_details_by_ids(baseline_question_ids_to_ask_list)
        
        if len(baseline_questions_details_map) != len(set(baseline_question_ids_to_ask_list)):
            print(f"Warning: Not all baseline question IDs found in 'questions' table. "
                  f"Requested: {len(set(baseline_question_ids_to_ask_list))}, Found: {len(baseline_questions_details_map)}")

        for bq_id in baseline_question_ids_to_ask_list: 
            bq_data = baseline_questions_details_map.get(int(bq_id)) 
            if bq_data:
                if store_single_followup_question(session_id, bq_data):
                    baseline_questions_actually_stored_count += 1
            else:
                print(f"Warning: Details for baseline question ID {bq_id} not found. Skipping storage.")
        print(f"Attempted to store {len(baseline_question_ids_to_ask_list)} baseline questions. Successfully stored: {baseline_questions_actually_stored_count}.")
    else:
        print("No baseline questions configured for this assessment type.")

    # --- 3. Wait for Baseline Answers ---
    if baseline_question_ids_to_ask_list and baseline_questions_actually_stored_count > 0:
        print(f"\n--- Step 3: Waiting for {baseline_questions_actually_stored_count} Baseline Question Answers ---")
        wait_for_batch_answers(session_id, baseline_question_ids_to_ask_list)
    else:
        print("No baseline questions to wait for (either none selected, none configured, or none successfully stored).")

    # --- 4. Generate Initial Summaries (AFTER Baseline Answers) ---
    print("\n--- Step 4: Generating Initial Context Summaries (Post-Baseline) ---")
    all_followups_post_baseline_list = get_followup_responses(session_id, answered_only=False)
    all_answered_followups_post_baseline_list = [r for r in all_followups_post_baseline_list if r.get("answer") is not None]

    formatted_survey_qa_text = format_survey_qa(survey_questions_raw_list, survey_responses_list)
    survey_context_summary_text = call_llm(
        prompt=prompt_templates.SUMMARIZE_SURVEY_CONTEXT_PROMPT.format(
            survey_questions_answers=formatted_survey_qa_text
        )
    ) or "Survey context summary generation failed or returned empty."

    answered_baseline_responses_list = [
        r for r in all_answered_followups_post_baseline_list if r.get("question_id") in active_baseline_ids
    ]
    formatted_baseline_qa_text = format_followup_qa(answered_baseline_responses_list) 
    baseline_summary_text = call_llm(
        prompt=prompt_templates.BASELINE_SUMMARY_PROMPT.format(
            baseline_questions_answers=formatted_baseline_qa_text
        )
    ) or "Baseline summary generation failed or returned empty."
    
    initial_common_context_summary_text = call_llm(
        prompt=prompt_templates.COMMON_INITIAL_CONTEXT_SUMMARY_PROMPT.format(
            baseline_summary=baseline_summary_text, 
            survey_context_summary=survey_context_summary_text 
        )
    ) or "Initial common context summary generation failed or returned empty."
    print(f"\nInitial Common Context Summary (Post-Baseline Wait):\n{initial_common_context_summary_text[:500]}...")
    
    current_common_context_summary = initial_common_context_summary_text

    # --- 5. Track Overall State (Post-Baseline, Pre-Main Loop) ---
    all_followups_for_global_tracking = get_followup_responses(session_id, answered_only=False)
    globally_stored_qids_in_session = list(set(
        [int(r.get("question_id")) for r in all_followups_for_global_tracking if r.get("question_id") is not None]
    ))

    total_non_baseline_questions_asked_count = sum(
        1 for qid in globally_stored_qids_in_session if qid not in active_baseline_ids 
    )
    
    print(f"Total Questions Stored in 'followup_responses' (incl. baseline) after baseline phase: {len(globally_stored_qids_in_session)}")
    print(f"Non-Baseline Questions Asked so far: {total_non_baseline_questions_asked_count}")
    
    unspent_quota_from_previous_iteration = 0 

    # --- 6. Main Iteration Loop ---
    print("\n" + "="*25 + " Starting Main Iterative Questioning " + "="*25)
    for current_iteration_num in range(1, iteration_logic.MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {current_iteration_num}/{iteration_logic.MAX_ITERATIONS} ---")

        # --- 6a. Fetch latest answers & Update Fragment States ---
        all_followups_iter_start_list = get_followup_responses(session_id, answered_only=False)
        all_answered_followups_iter_start_list = [r for r in all_followups_iter_start_list if r.get("answer") is not None]
        
        globally_stored_qids_in_session = list(set(
            [int(r.get("question_id")) for r in all_followups_iter_start_list if r.get("question_id") is not None]
        ))

        print(f"Debug Iter {current_iteration_num}: Updating fragment states. Total globally stored QIDs: {len(globally_stored_qids_in_session)}")
        fragment_states_map = iteration_logic.update_fragment_states(
            fragment_states_map,
            all_followups_iter_start_list, 
            all_dependency_rules_list,
            all_questions_id_to_fragment_map, 
            active_baseline_ids, 
            current_iteration_num
        )
        
        # --- 6b. Determine Available Questions & Check for Assessment Completion ---
        current_fragment_pool_counts = {}
        dependency_excluded_ids_this_iter = calculate_dependency_exclusions(
            all_dependency_rules_list, all_answered_followups_iter_start_list
        )
        
        combined_exclude_ids_for_pool_rpc = list(set(globally_stored_qids_in_session + dependency_excluded_ids_this_iter))
        
        rpc_filter_for_pool_count = category_filter_from_q16.copy() 
        rpc_filter_for_pool_count["exclude_ids"] = combined_exclude_ids_for_pool_rpc

        print(f"Debug Iter {current_iteration_num}: Pool count exclude_ids count: {len(combined_exclude_ids_for_pool_rpc)}")
        for frag_code_pool_count in fragment_states_map: 
            if not fragment_states_map[frag_code_pool_count].get('is_complete', False):
                try:
                    count_response = supabase.rpc(
                        "count_matching_documents_by_fragment", 
                        {"fragment_filter": str(frag_code_pool_count), "filter": rpc_filter_for_pool_count}
                    ).execute()
                    current_fragment_pool_counts[frag_code_pool_count] = count_response.data if count_response.data is not None else 0
                except Exception as e_rpc_count:
                    print(f"Error calling count_matching_documents_by_fragment RPC for {frag_code_pool_count}: {e_rpc_count}")
                    current_fragment_pool_counts[frag_code_pool_count] = 0 
        
        if iteration_logic.check_if_assessment_complete(fragment_states_map, total_non_baseline_questions_asked_count, current_fragment_pool_counts):
            print(f"Assessment complete at start of Iteration {current_iteration_num}. Ending.")
            break
        
        # --- 6c. Determine Targets for This Iteration ---
        iteration_targets_map, total_questions_allocated_this_iter = iteration_logic.determine_iteration_targets(
            fragment_states_map,
            current_fragment_pool_counts,
            active_offload_priority, 
            unspent_quota_from_previous_iteration,
            current_iteration_num
        )

        if total_questions_allocated_this_iter == 0:
            print(f"No questions allocated by determine_iteration_targets for Iteration {current_iteration_num}.")
            if iteration_logic.check_if_assessment_complete(fragment_states_map, total_non_baseline_questions_asked_count, current_fragment_pool_counts):
                 print("Assessment confirmed complete or pools exhausted after 0 allocation. Ending iterations.")
            else:
                 print("Pools might still be available but no target allocated. Ending iterations for safety.")
            break 

        # --- 6d. Generate/Update Common Context Summary (for Iterations > 1) ---
        if current_iteration_num > 1:
            print(f"Generating updated common context for Iteration {current_iteration_num}...")
            current_iter_fragment_descriptors_text = "\n".join(
                [f"- {f_id_desc}: {current_assessment_fragment_info_map.get(f_id_desc, {}).get('description', 'N/A')}" 
                 for f_id_desc in active_target_fragments_list if f_id_desc in fragment_states_map]
            )
            current_iter_fragment_qa_status_summaries_text = "\n".join(
                [f"Fragment {f_id_stat_sum}: Status Update:\n{state_stat_sum.get('status_summary', 'No status summary available.')}\n" 
                 for f_id_stat_sum, state_stat_sum in fragment_states_map.items() if f_id_stat_sum in active_target_fragments_list]
            )
            current_common_context_summary = call_llm( 
                prompt_templates.COMMON_CONTEXT_SUMMARY_PROMPT.format(
                    baseline_survey_summary=initial_common_context_summary_text, 
                    industry_context=industry_context_text,
                    fragment_descriptors=current_iter_fragment_descriptors_text,
                    fragment_qa_summaries=current_iter_fragment_qa_status_summaries_text 
                )
            ) or "Common context update generation failed or returned empty."
            print(f"Updated Common Context Summary (Iter {current_iteration_num}):\n {current_common_context_summary[:500]}...")

        # --- 6e. Fragment Processing: Retrieve, Rank, Select Candidate Questions ---
        print(f"\n--- Iteration {current_iteration_num}: Processing Targeted Fragments ---")
        candidates_selected_this_iteration_list = [] 
        all_retrieved_qdata_this_iteration_map = {} 
        shortfall_per_fragment_map = {} 

        dependency_excluded_ids_for_retrieval_rpc = calculate_dependency_exclusions(
            all_dependency_rules_list, all_answered_followups_iter_start_list 
        )
        combined_exclude_ids_for_retrieval_rpc = list(set(globally_stored_qids_in_session + dependency_excluded_ids_for_retrieval_rpc))
        print(f"Debug Iter {current_iteration_num}: Combined Exclusions for Retrieval RPC: {len(combined_exclude_ids_for_retrieval_rpc)} IDs")

        fragments_to_process_in_iter = [f_code for f_code, target_val in iteration_targets_map.items() if target_val > 0]
        print(f"Targeting fragments in Iteration {current_iteration_num}: {fragments_to_process_in_iter}")

        for current_fragment_code_proc in fragments_to_process_in_iter:
            target_q_count_for_fragment = iteration_targets_map[current_fragment_code_proc]
            print(f"\n  Processing Fragment: {current_fragment_code_proc} (Target Questions: {target_q_count_for_fragment})")
            
            fragment_descriptor_obj = current_assessment_fragment_info_map.get(current_fragment_code_proc)
            if not fragment_descriptor_obj: 
                print(f"  Error: Description for active fragment {current_fragment_code_proc} not found. Skipping.")
                shortfall_per_fragment_map[current_fragment_code_proc] = target_q_count_for_fragment
                continue
            fragment_descriptor_text_proc = fragment_descriptor_obj.get("description", "N/A")

            frag_specific_answered_followups_list = []
            for resp in all_answered_followups_iter_start_list: 
                resp_qid = resp.get("question_id")
                if resp_qid is not None and all_questions_id_to_fragment_map.get(int(resp_qid)) == current_fragment_code_proc:
                    if not resp.get("question"): 
                        q_detail_for_frag_qa = get_question_details_by_ids([resp_qid]).get(int(resp_qid))
                        if q_detail_for_frag_qa: resp["question"] = q_detail_for_frag_qa.get("question", "N/A")
                    frag_specific_answered_followups_list.append(resp)
            
            formatted_fragment_qa_text_proc = format_followup_qa(frag_specific_answered_followups_list)
            fragment_qa_summary_text_proc = call_llm(
                prompt_templates.FRAGMENT_QA_SUMMARY_PROMPT.format(
                    fragment_descriptor=fragment_descriptor_text_proc, 
                    fragment_questions_answers=formatted_fragment_qa_text_proc
                )
            ) or f"Fragment {current_fragment_code_proc} QA summary generation failed."

            final_pre_retrieval_summary_text_proc = call_llm(
                prompt_templates.FINAL_PRE_RETRIEVAL_PROMPT.format(
                    common_context_summary=current_common_context_summary, 
                    fragment_descriptor=fragment_descriptor_text_proc,
                    fragment_qa_summary=fragment_qa_summary_text_proc, 
                    industry_context=industry_context_text
                ), max_tokens=1800 
            ) or f"Pre-retrieval summary for {current_fragment_code_proc} generation failed."

            if "failed" in final_pre_retrieval_summary_text_proc.lower() or not final_pre_retrieval_summary_text_proc.strip():
                print(f"  Error: Cannot retrieve for {current_fragment_code_proc} due to unavailable pre-retrieval summary. Summary: '{final_pre_retrieval_summary_text_proc[:100]}...'")
                shortfall_per_fragment_map[current_fragment_code_proc] = target_q_count_for_fragment
                continue
            
            query_embedding_vector = embed_text(final_pre_retrieval_summary_text_proc, task_type="RETRIEVAL_QUERY")
            if not query_embedding_vector:
                print(f"  Error: Failed to generate embedding for pre-retrieval summary of {current_fragment_code_proc}. Skipping.")
                shortfall_per_fragment_map[current_fragment_code_proc] = target_q_count_for_fragment
                continue

            rpc_filter_for_question_retrieval = category_filter_from_q16.copy()
            rpc_filter_for_question_retrieval["exclude_ids"] = combined_exclude_ids_for_retrieval_rpc
            
            retrieved_questions_from_rpc_list = []
            try:
                num_to_retrieve_with_buffer = target_q_count_for_fragment + 3 
                retrieval_response = supabase.rpc(
                    "match_documents_by_fragment", 
                    {
                        "query_embedding": query_embedding_vector, 
                        "fragment_filter": str(current_fragment_code_proc), 
                        "filter": rpc_filter_for_question_retrieval, 
                        "match_count": num_to_retrieve_with_buffer
                    }
                ).execute()
                retrieved_questions_from_rpc_list = retrieval_response.data or []
                print(f"  Retrieved {len(retrieved_questions_from_rpc_list)} candidate questions for {current_fragment_code_proc} "
                      f"(Target: {target_q_count_for_fragment}, Requested from RPC: {num_to_retrieve_with_buffer})")
                all_retrieved_qdata_this_iteration_map[current_fragment_code_proc] = copy.deepcopy(retrieved_questions_from_rpc_list)
            except Exception as e_rpc_match:
                print(f"  Error during RPC 'match_documents_by_fragment' for {current_fragment_code_proc}: {e_rpc_match}")
                all_retrieved_qdata_this_iteration_map[current_fragment_code_proc] = [] 
                shortfall_per_fragment_map[current_fragment_code_proc] = target_q_count_for_fragment
                continue 

            if not retrieved_questions_from_rpc_list:
                print(f"  No questions retrieved from RPC for {current_fragment_code_proc}.")
                shortfall_per_fragment_map[current_fragment_code_proc] = target_q_count_for_fragment
                continue

            formatted_retrieved_for_llm_ranking = format_retrieved_questions(retrieved_questions_from_rpc_list)
            ranked_list_text_from_llm = call_llm(
                prompt_templates.POST_RETRIEVAL_FILTERING_PROMPT.format(
                    common_context_summary=current_common_context_summary, 
                    fragment_descriptor=fragment_descriptor_text_proc,
                    fragment_qa_summary=fragment_qa_summary_text_proc, 
                    industry_context=industry_context_text,
                    retrieved_questions=formatted_retrieved_for_llm_ranking 
                )
            )
            
            original_retrieved_ids_list = [q_data.get('id') for q_data in retrieved_questions_from_rpc_list if q_data.get('id') is not None]
            ranked_qids_from_llm_list = parse_llm_ranking(ranked_list_text_from_llm or "", original_retrieved_ids_list)

            current_fragment_selected_candidates_list = []
            if ranked_qids_from_llm_list: 
                print(f"  LLM ranked {len(ranked_qids_from_llm_list)} IDs for {current_fragment_code_proc}. Selecting top {target_q_count_for_fragment}.")
                for rank_idx, selected_qid_from_llm in enumerate(ranked_qids_from_llm_list):
                    if len(current_fragment_selected_candidates_list) < target_q_count_for_fragment:
                        similarity_score = 0.0
                        for rpc_q_data in retrieved_questions_from_rpc_list:
                            if rpc_q_data.get('id') == selected_qid_from_llm:
                                similarity_score = rpc_q_data.get('similarity', 0.0)
                                break
                        current_fragment_selected_candidates_list.append({
                            "fragment": current_fragment_code_proc, "id": selected_qid_from_llm, 
                            "rank": rank_idx, "similarity": similarity_score 
                        })
                    else: break 
            else: 
                print(f"  Warning: Using fallback (similarity-based) selection for {current_fragment_code_proc} as LLM ranking failed or was empty.")
                for i_fb, q_data_fb in enumerate(retrieved_questions_from_rpc_list):
                    if len(current_fragment_selected_candidates_list) < target_q_count_for_fragment:
                         current_fragment_selected_candidates_list.append({
                            "fragment": current_fragment_code_proc, "id": q_data_fb['id'], 
                            "rank": -1, 
                            "similarity": q_data_fb.get('similarity', 0.0)
                        })
                    else: break 
            
            print(f"  Selected {len(current_fragment_selected_candidates_list)} candidate questions for {current_fragment_code_proc}.")
            candidates_selected_this_iteration_list.extend(current_fragment_selected_candidates_list)
            
            current_fragment_shortfall = target_q_count_for_fragment - len(current_fragment_selected_candidates_list)
            if current_fragment_shortfall > 0:
                shortfall_per_fragment_map[current_fragment_code_proc] = current_fragment_shortfall
                print(f"  Shortfall for {current_fragment_code_proc}: {current_fragment_shortfall} questions.")
        
        print(f"\nDebug Iter {current_iteration_num}: Raw candidates selected across all fragments ({len(candidates_selected_this_iteration_list)}): "
              f"{[c['id'] for c in candidates_selected_this_iteration_list]}")
        
        # --- 6f. De-duplicate Candidate Questions ---
        deduplicated_candidates_for_iter_list = []
        seen_qids_in_iter_selection = set()
        for cand_to_dedup in candidates_selected_this_iteration_list:
            q_id_cand_dedup = cand_to_dedup.get('id')
            if q_id_cand_dedup not in seen_qids_in_iter_selection:
                deduplicated_candidates_for_iter_list.append(cand_to_dedup)
                seen_qids_in_iter_selection.add(q_id_cand_dedup)
        
        candidates_after_deduplication_list = deduplicated_candidates_for_iter_list
        print(f"Debug Iter {current_iteration_num}: Deduplicated candidates for iteration ({len(candidates_after_deduplication_list)}): "
              f"{[c['id'] for c in candidates_after_deduplication_list]}")

        # --- 6g. Calculate Unspent Quota ---
        unspent_quota_from_previous_iteration = sum(shortfall_per_fragment_map.values()) 
        print(f"Total shortfall this iteration (becomes unspent for next iter): {unspent_quota_from_previous_iteration}")

        # --- 6h. Apply 'must_ask_first' Rule ---
        print(f"\n--- Iteration {current_iteration_num}: Applying 'must_ask_first' (No Co-occurrence) to {len(candidates_after_deduplication_list)} candidates ---")
        must_ask_first_rules_active_list = [r for r in all_dependency_rules_list if r.get("rule_type") == "must_ask_first"]
        
        final_selection_for_storage_list = copy.deepcopy(candidates_after_deduplication_list)

        if must_ask_first_rules_active_list and final_selection_for_storage_list:
            made_change_in_maf_loop = True 
            maf_loop_iteration_count = 0
            max_maf_loops = len(final_selection_for_storage_list) * len(must_ask_first_rules_active_list) + 5 
            
            while made_change_in_maf_loop and maf_loop_iteration_count < max_maf_loops:
                made_change_in_maf_loop = False
                maf_loop_iteration_count += 1
                current_selected_qids_in_maf_loop = {c.get('id') for c in final_selection_for_storage_list}
                
                for maf_rule in must_ask_first_rules_active_list:
                    try:
                        dep_id_maf = int(maf_rule.get("dependent_question_id"))
                        indep_id_str_maf = maf_rule.get("independent_question_id") 
                        indep_id_maf = int(indep_id_str_maf.strip()) if indep_id_str_maf and indep_id_str_maf.strip() else None
                    except (ValueError, TypeError): continue 

                    if not dep_id_maf or not indep_id_maf: continue

                    if dep_id_maf in current_selected_qids_in_maf_loop and indep_id_maf in current_selected_qids_in_maf_loop:
                        print(f"  INFO (must_ask_first): Co-occurrence violation. Dependent Q {dep_id_maf} and Independent Q {indep_id_maf}. Attempting to replace Q {dep_id_maf}.")
                        
                        index_of_candidate_to_replace = -1
                        for i_maf, c_maf_cand in enumerate(final_selection_for_storage_list):
                            if c_maf_cand.get('id') == dep_id_maf:
                                index_of_candidate_to_replace = i_maf
                                break
                        
                        if index_of_candidate_to_replace != -1:
                            fragment_code_of_replaced_q = final_selection_for_storage_list[index_of_candidate_to_replace].get('fragment')
                            fallback_options_for_maf = all_retrieved_qdata_this_iteration_map.get(fragment_code_of_replaced_q, [])
                            replacement_found_for_maf = False
                            for fb_q_maf_data in fallback_options_for_maf:
                                fb_qid_maf = fb_q_maf_data.get('id')
                                if fb_qid_maf is None: continue

                                is_valid_maf_fallback = (
                                    fb_qid_maf != dep_id_maf and \
                                    fb_qid_maf != indep_id_maf and \
                                    fb_qid_maf not in current_selected_qids_in_maf_loop and \
                                    fb_qid_maf not in combined_exclude_ids_for_retrieval_rpc 
                                )
                                if is_valid_maf_fallback:
                                    print(f"  INFO (must_ask_first): Replacing Q {dep_id_maf} with fallback Q {fb_qid_maf}.")
                                    final_selection_for_storage_list[index_of_candidate_to_replace] = {
                                        "fragment": fragment_code_of_replaced_q, "id": fb_qid_maf,
                                        "rank": -2, "similarity": fb_q_maf_data.get('similarity', 0.0) 
                                    }
                                    replacement_found_for_maf = True
                                    made_change_in_maf_loop = True 
                                    break 
                            
                            if not replacement_found_for_maf: 
                                print(f"  WARNING (must_ask_first): No valid fallback for Q {dep_id_maf}. Removing it.")
                                final_selection_for_storage_list.pop(index_of_candidate_to_replace)
                                made_change_in_maf_loop = True 
                            
                        if made_change_in_maf_loop:
                            break 
                
                if maf_loop_iteration_count >= max_maf_loops:
                    print("  WARNING (must_ask_first): Max loops reached. Proceeding with current selection.")
        
        print(f"Final selection after 'must_ask_first' for Iteration {current_iteration_num} ({len(final_selection_for_storage_list)} questions): "
              f"{[c.get('id') for c in final_selection_for_storage_list]}")

        # --- 6i. Store Final Questions & Update Global State ---
        qids_actually_stored_this_iteration_list = [] 
        if final_selection_for_storage_list:
            print(f"\n--- Iteration {current_iteration_num}: Storing {len(final_selection_for_storage_list)} Final Questions ---")
            
            final_qids_to_fetch_details = [c.get('id') for c in final_selection_for_storage_list if c.get('id') is not None]
            final_question_details_map_for_storage = get_question_details_by_ids(final_qids_to_fetch_details)

            for candidate_to_store in final_selection_for_storage_list:
                selected_qid_to_store = candidate_to_store.get('id')
                if selected_qid_to_store is None: continue

                question_data_for_storing = final_question_details_map_for_storage.get(int(selected_qid_to_store))

                if question_data_for_storing:
                    if int(selected_qid_to_store) not in globally_stored_qids_in_session:
                        if store_single_followup_question(session_id, question_data_for_storing):
                            print(f"  Successfully stored new question ID: {selected_qid_to_store} to followup_responses.")
                            qids_actually_stored_this_iteration_list.append(int(selected_qid_to_store))
                            globally_stored_qids_in_session.append(int(selected_qid_to_store)) 
                            
                            if int(selected_qid_to_store) not in active_baseline_ids:
                                total_non_baseline_questions_asked_count += 1
                    else:
                         print(f"  INFO: Question {selected_qid_to_store} was in final selection but already globally stored. Skipping re-storage.")
                else:
                    print(f"  Error: Could not fetch details for final selected question ID {selected_qid_to_store}. Cannot store.")
        else:
            print(f"No questions in the final selection list for Iteration {current_iteration_num}. Nothing to store.")
        
        print(f"Total non-baseline questions asked after Iteration {current_iteration_num} storage: {total_non_baseline_questions_asked_count}")

        # --- 6j. Generate Fragment Status Summaries ---
        print(f"\n--- Iteration {current_iteration_num}: Generating Fragment Status Summaries for Next Iteration ---")
        for frag_id_for_status_gen in active_target_fragments_list:
            if frag_id_for_status_gen in fragment_states_map: 
                frag_desc_for_status = current_assessment_fragment_info_map.get(frag_id_for_status_gen, {}).get("description", "N/A")
                
                frag_specific_answered_for_status = []
                for resp_stat_sum in all_answered_followups_iter_start_list: 
                    resp_stat_qid = resp_stat_sum.get("question_id")
                    if resp_stat_qid is not None and all_questions_id_to_fragment_map.get(int(resp_stat_qid)) == frag_id_for_status_gen:
                        if not resp_stat_sum.get("question"): 
                            q_detail_stat_sum = get_question_details_by_ids([resp_stat_qid]).get(int(resp_stat_qid))
                            if q_detail_stat_sum: resp_stat_sum["question"] = q_detail_stat_sum.get("question", "N/A")
                        frag_specific_answered_for_status.append(resp_stat_sum)

                formatted_qa_for_frag_status = format_followup_qa(frag_specific_answered_for_status)
                frag_qa_summary_for_status_prompt = call_llm(
                    prompt_templates.FRAGMENT_QA_SUMMARY_PROMPT.format(
                        fragment_descriptor=frag_desc_for_status, 
                        fragment_questions_answers=formatted_qa_for_frag_status
                    )
                ) or f"Frag {frag_id_for_status_gen} QA summary for status prompt failed."

                current_frag_status_summary_text = call_llm(
                    prompt_templates.FRAGMENT_STATUS_SUMMARY_PROMPT.format(
                        fragment_descriptor=frag_desc_for_status,
                        fragment_qa_summary=frag_qa_summary_for_status_prompt 
                    )
                )
                if current_frag_status_summary_text:
                    fragment_states_map[frag_id_for_status_gen]["status_summary"] = current_frag_status_summary_text
                else:
                    fragment_states_map[frag_id_for_status_gen]["status_summary"] = "Status summary generation failed for this iteration."

        # --- 6k. Wait for User Answers for THIS Iteration's Batch ---
        if qids_actually_stored_this_iteration_list: 
            print(f"\n--- Iteration {current_iteration_num}: Waiting for user to answer {len(qids_actually_stored_this_iteration_list)} questions ---")
            wait_for_batch_answers(session_id, qids_actually_stored_this_iteration_list)
        else:
            print(f"No new questions were successfully stored and posed in Iteration {current_iteration_num}. Skipping wait.")

        # --- 6l. Check Max Questions Limit ---
        if total_non_baseline_questions_asked_count >= iteration_logic.MAX_TOTAL_QUESTIONS:
            print(f"Reached maximum total non-baseline questions limit ({iteration_logic.MAX_TOTAL_QUESTIONS}) "
                  f"after Iteration {current_iteration_num}. Ending iterations.")
            break
        
        if current_iteration_num == iteration_logic.MAX_ITERATIONS:
            print(f"Reached maximum configured iterations ({iteration_logic.MAX_ITERATIONS}). Ending assessment.")
        
    # --- 7. Pipeline End & Final Summary ---
    print(f"\n{'='*25} RAG Pipeline Completed for Session: {session_id} {'='*25}")
    print(f"Total Non-Baseline Questions Asked During Pipeline: {total_non_baseline_questions_asked_count}")
    print(f"Assessment Type Run: {assessment_type_log_message}")
    print("Final Fragment States:")
    for frag_id_final_summary, state_final_summary in fragment_states_map.items():
        print(f"  Fragment {frag_id_final_summary}: "
              f"Quota Fulfilled {state_final_summary.get('quota_fulfilled', 'N/A')}/{state_final_summary.get('quota_assigned', 'N/A')}, "
              f"Complete: {state_final_summary.get('is_complete', 'N/A')}, "
              f"Maturity: {state_final_summary.get('maturity_score', 0.0):.2f}")
    return True 


# --- Execution Guard ---
if __name__ == "__main__":
    test_session_id = "your_test_session_id_here" 
    
    if test_session_id == "your_test_session_id_here":
        print("ERROR: Please replace 'your_test_session_id_here' with an actual session ID for testing.")
    else:
        print(f"--- Initiating Test Run for Iterative RAG Pipeline with Session ID: {test_session_id} ---")
        pipeline_status_success = run_iterative_rag_pipeline(test_session_id)
        
        if pipeline_status_success:
            print(f"\n--- Pipeline execution completed for session {test_session_id}. ---")
        else:
            print(f"\n--- Pipeline execution failed or was aborted for session {test_session_id}. ---")
