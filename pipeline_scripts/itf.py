import time
import json
import sys, os
from typing import Dict, Any, List, Tuple
import numpy as np
from math import ceil
import google.generativeai as genai
from backend.utils import config
from supabase import create_client, Client
from rag_pipeline import build_category_filter
from prompt_templates import (
    POST_RETRIEVAL_PROMPT
    )
from maturity_fragment_utils import*

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

SUPABASE_URL = config.SUPABASE_URL
SUPABASE_KEY = config.SUPABASE_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    genai.configure(api_key=config.GOOGLE_API_KEY)
except AttributeError as e:
    print(f"Error: Failed to get GOOGLE_API_KEY from config. Ensure it's set. Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during Gemini configuration: {e}")
    sys.exit(1)

llm_model_name = "gemma-3-27b-it"
llm_generation_config = genai.types.GenerationConfig(temperature=0.4, max_output_tokens=1800)
llm = genai.GenerativeModel(
    llm_model_name,
    generation_config=llm_generation_config
)

embedding_model = "models/text-embedding-004"

CATEGORY_CONFIGS = {
    "Green With Software": {
        "fragments": ["A", "B", "C", "D", "E", "F"],
        "initial_quotas": { "A": 4, "B": 15, "C": 10, "D": 14, "E": 4, "F": 3 },
        "target_total": 50,
        "max_iterations": 7 # Suitable for target 50
    },
    "Green Within Software": {
        # --- PLACEHOLDER: Define actual fragments and quotas ---
        "fragments": ["G", "H", "I"], # Example fragments
        "initial_quotas": { "G": 20, "H": 15, "I": 15 }, # Example quotas summing to 50
        "target_total": 50,
        "max_iterations": 7 # Suitable for target 50
    },
    "Both": { # Represents when no specific category filter is returned
        # --- PLACEHOLDER: Define actual fragments and quotas ---
        # Combine fragments from both? Or have a separate set? Combine for example:
        "fragments": ["A", "B", "C", "D", "E", "F", "G", "H", "I"], # Example combined
        "initial_quotas": {
             # Example quotas summing to 80
             "A": 6, "B": 18, "C": 12, "D": 16, "E": 6, "F": 4, # Adjusted Green With (sum=62)
             "G": 7, "H": 6, "I": 5                           # Adjusted Green Within (sum=18) -> Total 80
        },
        "target_total": 80,
        "max_iterations": 10 # Increase iterations for larger target
    }
}



def embed_text(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> List[float]:
    """Generates embeddings for the given text using the Gemini API."""
    try:
        response = genai.embed_content(
            model=embedding_model,
            content=text,
            task_type=task_type
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding for text snippet '{text[:50]}...': {e}")
        return [0.0] * 768

def wait_for_user_answers(session_id: str):
    """
    Check if there are any unanswered follow-up questions.
    If unanswered questions exist, wait until the user provides responses.
    """
    while True:
        try:
            unanswered = supabase.table("followup_responses") \
                .select("question_id") \
                .eq("session_id", session_id) \
                .is_("answer", None) \
                .execute().data or []

            if not unanswered:
                print("All follow-up questions have been answered. Proceeding to next iteration...")
                return  # Exit function once all questions are answered

            print(f"Waiting for user to answer {len(unanswered)} follow-up questions...")
            time.sleep(15)  # Wait before checking again
        except Exception as e:
            print(f"Error checking unanswered questions: {e}")
            time.sleep(15)

def save_intermediate_state(state: dict, iteration: int) -> None:
    """
    Save the provided state to a JSON file for debugging/auditing.
    Handles potential non-serializable items like model objects.
    """
    filename = os.path.join(project_root, f"intermediate_state_iter_{iteration}.json")
    serializable_state = {}
    for key, value in state.items():
        # Add simple types directly
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
             serializable_state[key] = value
        # Add specific handling for known non-serializable types if needed
        # elif isinstance(value, SomeNonSerializableType):
        #     serializable_state[key] = repr(value) # Or some other representation
        else:
             # For unknown types, convert to string representation
             serializable_state[key] = f"Non-serializable type: {type(value).__name__}"

    try:
        with open(filename, "w") as f:
            json.dump(serializable_state, f, indent=2)
        print(f"Intermediate state for iteration {iteration} saved to {filename}")
    except TypeError as e:
        print(f"Error serializing intermediate state for iteration {iteration}: {e}. State contents: {serializable_state}")
    except Exception as e:
        print(f"Error saving intermediate state for iteration {iteration}: {e}")


def build_category_filter_and_target(survey_context: dict) -> Tuple[dict, int, str]:
    """
    Builds a category filter based on the survey response to Q16
    using the imported `build_category_filter` function, and determines the target total
    number of questions and the target category name.

    Returns:
        A tuple containing:
        - filter (dict): The category filter for Supabase RPC.
        - target_total (int): The target number of follow-up questions.
        - target_category_name (str): The name of the category scenario ('Green With Software', 'Green Within Software', 'Both').
    """
    target_total = 50  # Default target
    category_filter = {}
    target_category_name = "Green With Software" # Default if no specific match

    # Retrieve the category filter using the imported function
    # Pass the actual list of responses if survey_context['responses'] exists
    survey_responses_list = survey_context.get("responses", [])
    category_filter = build_category_filter(survey_responses_list) # Pass the list

    # Determine the target number of questions and category name based on the filter
    category_value = category_filter.get("category")
    if category_value == "Green With Software":
        target_category_name = "Green With Software"
        target_total = CATEGORY_CONFIGS[target_category_name]["target_total"]
        print(f"Q16 Answer indicates '{target_category_name}'. Target set to {target_total}.")
    elif category_value == "Green Within Software":
        target_category_name = "Green Within Software"
        target_total = CATEGORY_CONFIGS[target_category_name]["target_total"]
        print(f"Q16 Answer indicates '{target_category_name}'. Target set to {target_total}.")
    elif not category_filter:  # No specific category means 'Both'
        target_category_name = "Both"
        target_total = CATEGORY_CONFIGS[target_category_name]["target_total"]
        print(f"Q16 Answer indicates '{target_category_name}'. Target set to {target_total}.")
    else:
        # Fallback if filter returns something unexpected, use 'Both' defaults
        target_category_name = "Both"
        target_total = CATEGORY_CONFIGS[target_category_name]["target_total"]
        print(f"Q16 answer resulted in unexpected filter '{category_value}'. Defaulting to '{target_category_name}' with target={target_total}.")

    return category_filter, target_total, target_category_name



def get_aggregated_context(session_id: str) -> Dict[str, Any]:
    """Retrieve and aggregate survey responses and follow-up responses for a session."""
    context = {
        "survey_responses": {},
        "followup_responses": []
    }
    try:
        # Fetch the entire survey response object which might contain the 'responses' list
        survey_resp_data = supabase.table("survey_responses").select("*").eq("session_id", session_id).limit(1).single().execute().data
        if survey_resp_data:
            # Store the whole survey response object
            context["survey_responses"] = survey_resp_data # Contains 'responses' key and potentially others
    except Exception as e:
        # Handle potential error if .single() finds no or multiple rows
        if "returned 0 rows" not in str(e):
             print(f"Error fetching survey responses for session {session_id}: {e}")
    try:
        followup_resp = supabase.table("followup_responses").select("*").eq("session_id", session_id).execute().data
        if followup_resp:
             context["followup_responses"] = followup_resp
    except Exception as e:
        print(f"Error fetching follow-up responses for session {session_id}: {e}")

    return context

def get_question_metadata(question_id: int) -> dict:
    """
    Retrieve full metadata for a question by its ID from the questions table.
    """
    try:
        response = supabase.table("questions").select("*").eq("id", question_id).limit(1).maybe_single().execute().data # Use maybe_single for safety
        return response if response else {}
    except Exception as e:
        print(f"Error retrieving metadata for question_id {question_id}:", e)
        return {}
    
def run_iterative_rag_pipeline(session_id: str) -> None:
    """
    Iteratively retrieve follow-up questions fragment-by-fragment for the
    "Green With Software" category using maturity scores and dynamic quotas.
    """
    print(f"Starting dynamic iterative RAG pipeline for session: {session_id}")

    # --- Initialization ---
    # 1. Fetch initial context and determine category, target, fragments, quotas
    initial_context = get_aggregated_context(session_id)
    survey_context = initial_context.get("survey_responses", {}) # Get the survey response object

    # Use the new function to get dynamic settings
    category_filter, target_total, target_category_name = build_category_filter_and_target(survey_context)

    # Select the configuration based on the determined category name
    if target_category_name not in CATEGORY_CONFIGS:
        print(f"Error: Configuration for category '{target_category_name}' not found in CATEGORY_CONFIGS. Exiting.")
        return
    
    config = CATEGORY_CONFIGS[target_category_name]
    fragments = config["fragments"]
    initial_quotas = config["initial_quotas"]
    total_iterations = config["max_iterations"] # Use max_iterations from config
    TARGET_CATEGORY_FOR_RPC = category_filter.get("category") # This might be None for 'Both' scenario

    print(f"Pipeline configured for: '{target_category_name}'")
    print(f"Target total questions: {target_total}, Max iterations: {total_iterations}")
    print(f"Fragments: {fragments}")
    # print(f"Initial Quotas: {initial_quotas}") # Optional: print quotas

    # Verify quota sum matches target for the selected config
    if sum(initial_quotas.values()) != target_total:
        print(f"Warning: Initial quotas for '{target_category_name}' sum to {sum(initial_quotas.values())}, but target is {target_total}. Check CATEGORY_CONFIGS.")
        # Decide how to handle: proceed with warning, adjust quotas, or exit?
        # Proceeding with warning for now.

    # 2. Initialize Fragment States using the dynamically selected config
    fragment_states = initialize_fragment_states(fragments, initial_quotas)

    # 3. Fetch Dependency Rules (if using dependency logic)
    # dependency_rules = get_dependency_rules(supabase) # Fetch rules once

    # 4. Calculate initial adaptive size (can be recalculated each iteration)
    adaptive_iteration_size = ceil(target_total / total_iterations) if total_iterations > 0 else target_total
    print(f"Initial adaptive iteration size (approx): {adaptive_iteration_size}")

    # 5. Get initial stored questions and update fulfilled quotas
    followup_context = initial_context.get("followup_responses", [])
    # Filter context to only include questions from the fragments relevant to the TARGET_CATEGORY_FOR_RPC if needed
    # Note: If 'Both', followup_context might contain questions from either category.
    # The quota update logic below handles this by checking fragment_id.
    stored_ids = [resp.get("question_id") for resp in followup_context if resp.get("question_id") is not None]

    # Update initial fulfilled quotas based on existing stored questions *matching the fragments*
    for resp in followup_context:
        frag_id = resp.get("fragment") # Assumes fragment info is stored/retrieved correctly
        q_id = resp.get("question_id")
        # Only update state if the fragment belongs to the CURRENT configuration
        if frag_id in fragment_states and q_id is not None:
             # Simple count based on fragment ID in the response
             fragment_states[frag_id]["quota_fulfilled"] = sum(1 for r in followup_context if r.get("fragment") == frag_id and r.get("question_id") is not None)

    total_fulfilled = sum(state["quota_fulfilled"] for state in fragment_states.values()) # Sum fulfilled across *relevant* fragments
    print(f"Initial stored questions count (relevant fragments): {total_fulfilled}")
    print("Initial Fragment States (relevant):")
    for frag_id, state in fragment_states.items():
         print(f"  Fragment {frag_id}: Assigned={state['quota_assigned']}, Fulfilled={state['quota_fulfilled']}")
   
    # --- Iteration Loop ---
    iteration = 0
    while total_fulfilled < target_total and iteration < total_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration}/{total_iterations} | Fulfilled: {total_fulfilled}/{target_total} ---")

        # Optional: Recalculate adaptive size based on remaining needs/iterations
        remaining_questions = target_total - total_fulfilled
        remaining_iterations = total_iterations - iteration + 1
        current_iteration_target_size = ceil(remaining_questions / remaining_iterations) if remaining_iterations > 0 else 0
        current_iteration_target_size = min(current_iteration_target_size, remaining_questions) # Don't target more than needed
        print(f"Recalculated target for this iteration: {current_iteration_target_size}")

        if current_iteration_target_size <= 0:
            print("No more questions needed or iterations left. Ending loop.")
            break

        wait_for_user_answers(session_id) # Ensure user has answered previous batch (if applicable)

        # Fetch latest context ONLY if wait_for_user_answers was called or state might change
        context = get_aggregated_context(session_id)
        followup_context = context.get("followup_responses", [])
        stored_ids = list(set(resp.get("question_id") for resp in followup_context if resp.get("question_id") is not None)) # Update stored IDs (use set for uniqueness)
        total_fulfilled = len(stored_ids) # Recalculate fulfilled count accurately

        # Update fragment fulfilled counts based on the latest context
        for frag_id in fragment_states:
            fragment_states[frag_id]["quota_fulfilled"] = sum(1 for r in followup_context if r.get("fragment") == frag_id and r.get("question_id") is not None)


        # --- Fragment Processing within Iteration ---
        # 1. Calculate Maturity Scores for active fragments
        # Pass dependency info etc. if available for more complex scoring
        update_all_maturity_scores(fragment_states) # Pass context if needed

        # 2. Determine Per-Fragment Targets for this iteration
        per_fragment_targets = determine_iteration_targets(fragment_states, current_iteration_target_size)

        # 3. Retrieve, Filter, and Select Questions per Fragment
        all_prioritized_questions_this_iter = []
        processed_fragment_ids_this_iter = sorted(per_fragment_targets.keys(), key=lambda f: fragment_states[f]['maturity_score'], reverse=True) # Process higher maturity first

        # Generate a common summary part (e.g., overall progress) once per iteration
        # This replaces the old multi-step summary
        common_summary = f"Overall progress: {total_fulfilled}/{target_total} questions answered. Focus for this iteration is based on fragment maturity."
        # You could add more details from followup_context summary if needed.

        for frag_id in processed_fragment_ids_this_iter:
            target_count = per_fragment_targets.get(frag_id, 0)
            state = fragment_states[frag_id]

            if target_count <= 0 or state["is_complete"]:
                print(f"Skipping Fragment {frag_id} (Target: {target_count}, Complete: {state['is_complete']})")
                continue

            print(f"\nProcessing Fragment {frag_id}: Target for this iteration = {target_count}")

            # --- Fragment-Specific Retrieval ---
            # Prepare fragment-specific query (e.g., combine common summary + fragment details)
            # Simple approach: use common summary for embedding query
            query_text = f"{common_summary} Seeking questions for fragment {frag_id}." # Simple query
            print(f"  Embedding query: '{query_text[:100]}...'")
            query_embedding = embed_text(query_text, task_type="RETRIEVAL_QUERY")

            # Build filters for this fragment
            # Essential filters: category, fragment_id, exclude already stored IDs
            current_filter = {
                "category": TARGET_CATEGORY_FOR_RPC,
                "exclude_ids": stored_ids
                # Add dependency filters here if applicable for this fragment
                # "fulfilled_dependencies": [dep_id for dep_id in ... ]
            }
            num_candidates_to_retrieve = target_count * 3 # Retrieve more than target (e.g., 3x) for filtering

            try:
                print(f"  Calling RPC match_documents_by_fragment for Fragment {frag_id}...")
                rpc_response = supabase.rpc(
                    "match_documents_by_fragment",
                    {
                        "query_embedding": query_embedding,
                        "fragment_filter": frag_id, # Pass the fragment ID
                        "filter": current_filter,
                         # Optional: Add match_count and threshold if needed in SQL function
                         "match_count": num_candidates_to_retrieve,
                         "match_threshold": 0.7 # Example threshold
                    }
                ).execute()

                if rpc_response.data:
                    candidate_docs_fragment = rpc_response.data
                    print(f"  Retrieved {len(candidate_docs_fragment)} candidates for Fragment {frag_id}.")
                    if len(candidate_docs_fragment) < target_count: # Check if pool might be exhausted
                         print(f"  Warning: Retrieved fewer candidates ({len(candidate_docs_fragment)}) than targeted ({target_count}). Candidate pool might be low.")
                         # Consider marking as complete if 0 retrieved, or based on threshold
                         if len(candidate_docs_fragment) == 0:
                              print(f"  Marking Fragment {frag_id} as complete (no candidates retrieved).")
                              state["is_complete"] = True
                              continue # Skip to next fragment
                else:
                    candidate_docs_fragment = []
                    print(f"  No candidates retrieved for Fragment {frag_id}. Marking as complete.")
                    state["is_complete"] = True
                    # Log potential errors
                    if hasattr(rpc_response, 'error') and rpc_response.error:
                         print(f"  Supabase RPC error: {rpc_response.error}")
                    continue # Skip to next fragment

            except Exception as e:
                print(f"  Error during RPC retrieval for Fragment {frag_id}: {e}")
                candidate_docs_fragment = []
                state["is_complete"] = True # Assume complete on error? Or retry?
                continue # Skip to next fragment


            # --- Fragment-Specific Filtering (LLM) ---
            # Prepare minimal candidates and fragment-specific context
            minimal_candidates_fragment = []
            for doc in candidate_docs_fragment:
                 doc_dict = doc if isinstance(doc, dict) else {}
                 add_fields = doc_dict.get("additional_fields", {})
                 minimal_add_fields = {}
                 if "guidelines" in add_fields: minimal_add_fields["guidelines"] = add_fields["guidelines"]
                 if "answer_options" in add_fields: minimal_add_fields["answer_options"] = add_fields["answer_options"]
                 elif "subjective_answer" in add_fields: minimal_add_fields["subjective_answer"] = add_fields["subjective_answer"]

                 minimal_candidates_fragment.append({
                     "id": doc_dict.get("id"),
                     "question": doc_dict.get("question"),
                     "similarity": doc_dict.get("similarity"),
                     "additional_fields": minimal_add_fields
                 })

            fragment_context_str = json.dumps({
                 "fragment_id": frag_id,
                 "target_questions_this_iteration": target_count,
                 "quota_status": f"{state['quota_fulfilled']}/{state['quota_assigned']}",
                 "maturity_score": f"{state['maturity_score']:.3f}",
                 # Include common context summary
                 "overall_summary": common_summary,
                 "answered_questions_summary": f"Total answered: {total_fulfilled}", # Add summary of *recent* answers if helpful
                 }, indent=2)

            candidate_questions_fragment_str = json.dumps(minimal_candidates_fragment, indent=2)

            # Adjust POST_RETRIEVAL_PROMPT template to use fragment_context_str
            # Example adjustment (modify your actual prompt template):
            # """Context for Fragment {fragment_id}: {fragment_context}
            # Candidate Questions for Fragment {fragment_id}: {candidate_questions}
            # Select the best {num_required} questions..."""

            post_prompt = POST_RETRIEVAL_PROMPT.format(
                 # Pass fragment-specific context and common elements as needed by the adjusted prompt
                 fragment_id=frag_id, # Example: If needed directly in prompt
                 fragment_context=fragment_context_str, # Contains detailed context
                 candidate_questions=candidate_questions_fragment_str,
                 num_required=target_count # Guide LLM for this fragment's target
            )

            try:
                 print(f"  Invoking LLM for post-retrieval filtering for Fragment {frag_id}...")
                 post_filtered_response = llm.generate_content(post_prompt)
                 post_filtered_output = post_filtered_response.text
                 print(f"  LLM Filter Output for Fragment {frag_id}:\n{post_filtered_output[:200]}...") # Log snippet

                 # Parse output (assuming JSON list of {"id": ...}) - reuse existing parsing logic
                 if post_filtered_output.strip().startswith("```json"):
                     post_filtered_output = post_filtered_output.strip()[7:-3].strip()
                 elif post_filtered_output.strip().startswith("```"):
                     post_filtered_output = post_filtered_output.strip()[3:-3].strip()

                 filtered_ids_info = json.loads(post_filtered_output)
                 if not isinstance(filtered_ids_info, list) or not all(isinstance(item, dict) and 'id' in item for item in filtered_ids_info):
                      print(f"  Warning: LLM filter output for Fragment {frag_id} bad format. Using top N candidates by similarity.")
                      # Fallback: Sort by similarity and take top N
                      sorted_candidates = sorted(candidate_docs_fragment, key=lambda d: d.get('similarity', 0), reverse=True)
                      filtered_ids_info = [{"id": c.get("id")} for c in sorted_candidates[:target_count]]
                 else:
                      # Limit LLM selection to the target count for this fragment
                      filtered_ids_info = filtered_ids_info[:target_count]


            except Exception as e:
                 print(f"  Error during LLM filtering/parsing for Fragment {frag_id}: {e}. Using top N by similarity.")
                 # Fallback: Sort by similarity and take top N
                 sorted_candidates = sorted(candidate_docs_fragment, key=lambda d: d.get('similarity', 0), reverse=True)
                 filtered_ids_info = [{"id": c.get("id")} for c in sorted_candidates[:target_count]]


            # --- Metadata Lookup & Final Selection for Fragment ---
            fragment_questions_to_store = []
            print(f"  Looking up metadata for {len(filtered_ids_info)} selected candidates for Fragment {frag_id}...")
            for item in filtered_ids_info:
                q_id = item.get("id")
                if q_id is None: continue

                full_metadata = get_question_metadata(q_id)
                if full_metadata:
                    # Find original similarity score if needed
                    original_candidate = next((doc for doc in candidate_docs_fragment if doc.get("id") == q_id), None)
                    similarity_score = original_candidate.get('similarity') if original_candidate else None

                    # Add fragment_id to metadata if not already present
                    if 'fragment' not in full_metadata:
                         full_metadata['fragment'] = frag_id # Ensure fragment is associated

                    add_fields = full_metadata.get("additional_fields", {})
                    if not isinstance(add_fields, dict): add_fields = {}

                    chunk = {
                        "page_content": full_metadata.get("question", "N/A"),
                        "metadata": {
                            "id": full_metadata.get("id"),
                            "category": full_metadata.get("category"),
                            "subcategory": full_metadata.get("subcategory"),
                            "fragment": full_metadata.get("fragment"), # Store fragment
                            "sr_no": full_metadata.get("sr_no"),
                            "similarity": similarity_score,
                            "additional_fields": add_fields
                        }
                    }
                    fragment_questions_to_store.append(chunk)
                else:
                    print(f"    Warning: Could not retrieve full metadata for question_id {q_id}. Skipping.")

            print(f"  Selected {len(fragment_questions_to_store)} questions for Fragment {frag_id} this iteration.")
            all_prioritized_questions_this_iter.extend(fragment_questions_to_store)

        # --- End Fragment Processing Loop for Iteration ---


        # 4. Quota Redistribution (Check and apply after processing all fragments)
        redistribute_quotas(fragment_states)

        # 5. Store selected questions from this iteration
        if all_prioritized_questions_this_iter:
            print(f"\nStoring {len(all_prioritized_questions_this_iter)} new questions selected this iteration...")
            store_followup_questions(session_id, all_prioritized_questions_this_iter)
            # Update total count for the end-of-iteration message
            total_fulfilled += len(all_prioritized_questions_this_iter)
        else:
            print("\nNo new questions selected across all fragments this iteration.")

        # 6. Save Intermediate State
        intermediate_state = {
             "iteration": iteration,
             "target_total": target_total,
             "current_total_fulfilled": total_fulfilled,
             "adaptive_iteration_target": current_iteration_target_size,
             "fragment_states": fragment_states, # Store the whole state
             # Add specific outputs if needed for debugging
             "questions_stored_this_iter_count": len(all_prioritized_questions_this_iter),
        }
        save_intermediate_state(intermediate_state, iteration) # Ensure this handles dicts well

        print(f"--- Iteration {iteration} complete. Total stored: {total_fulfilled}/{target_total} ---")
        time.sleep(5) # Delay

    # --- Loop End ---
    final_state = get_aggregated_context(session_id)
    final_count = len(final_state.get("followup_responses", []))
    print(f"\nFragment-based RAG pipeline finished for session: {session_id}.")
    if final_count >= target_total:
        print(f"Target of {target_total} questions reached or exceeded (Final Count: {final_count}).")
    elif iteration >= total_iterations:
        print(f"Maximum iterations ({total_iterations}) reached. Final Count: {final_count}/{target_total}")
    else:
         print(f"Loop finished for other reasons (e.g., all fragments completed). Final Count: {final_count}/{target_total}")

    print("\nFinal Fragment Quota Status:")
    for frag_id, state in fragment_states.items():
         status = "Complete" if state['is_complete'] or state['quota_fulfilled'] >= state['quota_assigned'] else "Active"
         print(f"  Fragment {frag_id}: Fulfilled={state['quota_fulfilled']}, Assigned={state['quota_assigned']}, Status={status}")


def store_followup_questions(session_id: str, questions: List[dict]) -> None:
    """
    Store the retrieved candidate follow-up questions into a new Supabase table 'followup_responses'.
    The followup_responses table is assumed to have:
      - session_id (foreign key referencing user_sessions)
      - question_id (foreign key referencing questions.id)
      - question (text)
      - category (varchar)
      - subcategory (varchar)
      - additional_fields (JSONB) containing only the 'guidelines' and either 'answer_options' or 'subjective_answer'
      - answer (JSONB) which is initially null.
    """
    for q in questions:
        # Retrieve the candidate question's additional_fields from its metadata.
        orig_fields = q.get("metadata", {}).get("additional_fields", {})
        # Create a new additional_fields dict with only the desired keys.
        filtered_fields = {}
        if "guidelines" in orig_fields:
            filtered_fields["guidelines"] = orig_fields["guidelines"]
        # Check for answer options first; if not present, then check for subjective_answer.
        if "answer_options" in orig_fields:
            filtered_fields["answer_options"] = orig_fields["answer_options"]
        elif "subjective_answer" in orig_fields:
            filtered_fields["subjective_answer"] = orig_fields["subjective_answer"]
        
        record = {
            "session_id": session_id,
            # Convert the metadata 'id' to integer if needed. Here we assume it's stored as such.
            "question_id": q.get("metadata", {}).get("id"),
            "question": q.get("page_content"),
            "category": q.get("metadata", {}).get("category"),
            "subcategory": q.get("metadata", {}).get("subcategory"),
            "additional_fields": filtered_fields,
            "answer": None  # Initially, no answer is provided.
        }
        try:
            res = supabase.table("followup_responses").insert(record).execute()
            print("Stored follow-up question:", res.data)
        except Exception as e:
            print("Error storing follow-up question:", e)



# --- Main Execution Guard ---
if __name__ == "__main__":
    test_session_id = "your-actual-test-session-uuid" # <<< IMPORTANT: SET A REAL UUID

    if test_session_id == "your-actual-test-session-uuid":
         print("Error: Please replace 'your-actual-test-session-uuid' with a real session ID.")
    else:
        print(f"Starting FRAGMENT-BASED iterative RAG pipeline for session: {test_session_id}")
        # Ensure the session corresponds to the "Green With Software" category path
        # Maybe add a check here based on survey context if needed
        run_iterative_rag_pipeline(test_session_id)
        print(f"\nPipeline execution finished for session: {test_session_id}")
