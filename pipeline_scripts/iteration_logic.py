import math
import copy
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import json

MAX_ITERATIONS = 7
MAX_TOTAL_QUESTIONS = 50 
QUESTIONS_PER_ITERATION = 7
FRAGMENT_QUOTAS = {'A': 4, 'B': 15, 'C': 10, 'D': 14, 'E': 4, 'F': 3}
# priorities for offloading (lower number = higher priority)
OFFLOAD_PRIORITY = {'B': 1, 'C': 2, 'D': 3, 'A': 4, 'E': 5, 'F': 6, 'DEFAULT': 99}
# weight for time vs. readiness in maturity score (0=readiness only, 1=time only)
MATURITY_TIME_WEIGHT = 0.3


def extract_answer_value(answer_jsonb: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extracts the 'value' from the answer JSONB structure."""
    if isinstance(answer_jsonb, dict):
        return str(answer_jsonb.get("value")) if answer_jsonb.get("value") is not None else None
    return None

def count_relevant_independent_questions(
    fragment_id: str,
    rules: List[Dict[str, Any]],
    all_questions_map: Dict[int, str], # Map {q_id: fragment}
    baseline_ids: List[int]
) -> int:
    """Counts unique independent questions affecting this fragment from other fragments or baseline."""
    relevant_indep_ids = set()
    baseline_set = set(baseline_ids)
    for rule in rules:
        dep_id = rule.get("dependent_question_id")
        dep_fragment = all_questions_map.get(dep_id)

        if dep_fragment == fragment_id: # Rule affects this fragment
             indep_id_str = rule.get("independent_question_id")
             if indep_id_str:
                 try:
                      indep_ids = [int(i.strip()) for i in indep_id_str.split(';') if i.strip()]
                      for i_id in indep_ids:
                           indep_fragment = all_questions_map.get(i_id)
                           # Count if independent is baseline OR from another fragment
                           if i_id in baseline_set or (indep_fragment and indep_fragment != fragment_id):
                                relevant_indep_ids.add(i_id)
                 except ValueError:
                      pass # Ignore malformed rule IDs
    return len(relevant_indep_ids)

# --- Core Iteration Logic Functions ---

def update_fragment_states(
    fragment_states: Dict[str, dict],
    all_followups: List[Dict[str, Any]], # Pass fetched followups
    all_dependency_rules: List[Dict[str, Any]],
    all_questions_map: Dict[int, str], # Map {q_id: fragment}
    baseline_question_ids: List[int],
    current_iteration: int,
    # MAX_ITERATIONS # Use constant defined above
) -> Dict[str, dict]:
    """
    Updates fragment states with fulfilled quotas, readiness, and maturity scores.
    """
    print("--- Updating Fragment States ---")
    updated_states = copy.deepcopy(fragment_states) # Work on a copy
    all_answered_followups = [r for r in all_followups if r.get("answer") is not None]
    baseline_set = set(baseline_question_ids)

    for frag_id, state in updated_states.items():
        # Reset counts for recalculation
        fulfilled_count = 0
        questions_asked_in_frag_non_baseline = 0
        questions_answered_in_frag_non_baseline = 0
        answered_baseline_in_fragment = 0

        # 1. Calculate Fulfilled Quota & Answered Counts
        for resp in all_followups: # Iterate through all stored followups
            q_id = resp.get("question_id")
            if not q_id: continue
            q_fragment = all_questions_map.get(q_id)

            if q_fragment == frag_id:
                 is_answered = resp.get("answer") is not None
                 is_baseline = q_id in baseline_set

                 if not is_baseline:
                      questions_asked_in_frag_non_baseline += 1
                      if is_answered:
                           fulfilled_count += 1 # Increment fulfilled quota ONLY if answered and NOT baseline
                           questions_answered_in_frag_non_baseline += 1
                 elif is_answered: # Is baseline and is answered
                      answered_baseline_in_fragment += 1

        state["quota_fulfilled"] = fulfilled_count
        state["questions_asked_in_fragment"] = questions_asked_in_frag_non_baseline

        # 2. Estimate Dependencies
        state["estimated_dependency_count"] = count_relevant_independent_questions(
            frag_id, all_dependency_rules, all_questions_map, baseline_question_ids
        )

        # 3. Calculate Readiness Score Numerator Components
        answered_cross_fragment_baselines = 0
        answered_map = {resp['question_id']: extract_answer_value(resp.get('answer'))
                        for resp in all_answered_followups if resp.get('question_id') and resp.get('answer')}
        for rule in all_dependency_rules:
            dep_id = rule.get("dependent_question_id"); dep_fragment = all_questions_map.get(dep_id)
            if dep_fragment == frag_id:
                indep_id_str = rule.get("independent_question_id")
                if indep_id_str:
                    try:
                        indep_ids = [int(i.strip()) for i in indep_id_str.split(';') if i.strip()]
                        for i_id in indep_ids:
                            if i_id in baseline_set and i_id in answered_map:
                                answered_cross_fragment_baselines += 1
                    except ValueError: pass
        state["answered_independent_questions"] = answered_cross_fragment_baselines

        # 4. Calculate Readiness Score
        readiness_numerator = (
            questions_answered_in_frag_non_baseline
            + answered_baseline_in_fragment
            + answered_cross_fragment_baselines
        )
        estimated_total_needed_context = (state["quota_assigned"] + state["estimated_dependency_count"])
        state["estimated_total_needed_context"] = estimated_total_needed_context
        readiness_score = (readiness_numerator / estimated_total_needed_context) if estimated_total_needed_context > 0 else 0
        state["readiness_score"] = readiness_score

        # 5. Calculate Maturity Score
        time_progress = (current_iteration - 1) / max(1, MAX_ITERATIONS - 1)
        quota_progress = (state["questions_asked_in_fragment"] / state["quota_assigned"]) if state["quota_assigned"] > 0 else 0
        maturity_score = ((1 - MATURITY_TIME_WEIGHT) * readiness_score + MATURITY_TIME_WEIGHT * quota_progress)
        state["maturity_score"] = max(0.0, min(1.0, maturity_score))

        # 6. Check Completion
        state["is_complete"] = state["quota_fulfilled"] >= state["quota_assigned"]

        print(f"  Fragment {frag_id}: Quota {state['quota_fulfilled']}/{state['quota_assigned']}, "
              f"Readiness {readiness_score:.2f}, Maturity {state['maturity_score']:.2f}, Complete: {state['is_complete']}")

    return updated_states


def determine_iteration_targets(
    fragment_states: Dict[str, dict],
    # QUESTIONS_PER_ITERATION # Use constant
    fragment_pool_counts: Dict[str, int],
    unspent_from_previous: int = 0,
    iteration: int = 1
) -> Tuple[Dict[str, int], int]:
    """
    Determines how many questions to target for each fragment in this iteration.
    Returns: (targets_per_fragment, total_allocated_this_iteration)
    """
    print(f"--- Determining Targets for Iteration {iteration} (Base Target: {QUESTIONS_PER_ITERATION}, Unspent from Prev: {unspent_from_previous}) ---")
    targets = {frag_id: 0 for frag_id in fragment_states}
    effective_target_total = QUESTIONS_PER_ITERATION + unspent_from_previous
    print(f"Effective total target for allocation: {effective_target_total}")

    active_fragments = {}
    for frag_id, state in fragment_states.items():
        remaining_quota = state["quota_assigned"] - state["quota_fulfilled"]
        available_pool = fragment_pool_counts.get(frag_id, 0)
        if not state.get("is_complete", False) and remaining_quota > 0 and available_pool > 0:
            active_fragments[frag_id] = state
            print(f"  Fragment {frag_id}: Active (Rem Quota: {remaining_quota}, Pool: {available_pool}, Maturity: {state.get('maturity_score', 0):.2f})")

    if not active_fragments or effective_target_total <= 0:
        print("No active fragments or zero target total. No targets assigned.")
        return targets, 0

    fragment_ids_active = list(active_fragments.keys())
    # Weight based on INVERSE maturity (lower maturity = higher priority) & Priority
    maturity_scores = np.array([1.0 - state.get("maturity_score", 0.0) for state in active_fragments.values()])
    priority_values = np.array([OFFLOAD_PRIORITY.get(fid, OFFLOAD_PRIORITY['DEFAULT']) for fid in fragment_ids_active])
    priority_weights = 1.0 / np.maximum(1.0, priority_values)
    combined_scores = maturity_scores * priority_weights

    if np.sum(combined_scores) <= 1e-6: # Use tolerance for floating point comparison
        weights = np.ones(len(combined_scores)) / max(1, len(combined_scores))
        print("Combined scores sum near zero, using equal weights.")
    else:
        exp_scores = np.exp(combined_scores - np.max(combined_scores))
        weights = exp_scores / np.sum(exp_scores)

    allocated_sum = 0
    for i, frag_id in enumerate(fragment_ids_active):
        proportional_target = math.ceil(weights[i] * effective_target_total)
        remaining_quota = fragment_states[frag_id]["quota_assigned"] - fragment_states[frag_id]["quota_fulfilled"]
        available_pool = fragment_pool_counts.get(frag_id, 0)
        capped_target = min(proportional_target, remaining_quota, available_pool)
        targets[frag_id] = capped_target
        allocated_sum += capped_target

    discrepancy = allocated_sum - effective_target_total
    sorted_for_adjust = sorted(
         fragment_ids_active,
         key=lambda fid: (OFFLOAD_PRIORITY.get(fid, OFFLOAD_PRIORITY['DEFAULT']), active_fragments[fid].get("maturity_score", 0.0))
     )

    if discrepancy > 0: # Overallocated
        print(f"Overallocated by {discrepancy}. Reducing targets...")
        reduced = 0
        for frag_id in reversed(sorted_for_adjust):
            reduction = min(targets[frag_id], discrepancy - reduced)
            targets[frag_id] -= reduction
            reduced += reduction
            if reduced >= discrepancy: break
    elif discrepancy < 0: # Underallocated
        print(f"Underallocated by {-discrepancy}. Increasing targets...")
        added = 0; needed = -discrepancy
        for frag_id in sorted_for_adjust:
            remaining_quota = fragment_states[frag_id]["quota_assigned"] - fragment_states[frag_id]["quota_fulfilled"]
            available_pool = fragment_pool_counts.get(frag_id, 0)
            potential_add = max(0, min(remaining_quota, available_pool) - targets[frag_id])
            increase = min(potential_add, needed - added)
            targets[frag_id] += increase
            added += increase
            if added >= needed: break

    final_allocated_total = sum(targets.values())
    print(f"--- Final Targets Assigned: {targets} (Total: {final_allocated_total}) ---")
    return targets, final_allocated_total


def check_if_assessment_complete(
    fragment_states: Dict[str, dict],
    total_questions_asked: int,
    fragment_pool_counts: Dict[str, int] # Pass pool counts for exhaustion check
) -> bool:
    """Checks if all fragment quotas are met, max questions reached, or pools exhausted."""
    all_quotas_met = all(state.get("is_complete", False) for state in fragment_states.values())
    max_questions_reached = total_questions_asked >= MAX_TOTAL_QUESTIONS

    if all_quotas_met:
        print("Assessment Complete Check: All fragment quotas fulfilled.")
        return True
    if max_questions_reached:
        print(f"Assessment Complete Check: Maximum total questions ({MAX_TOTAL_QUESTIONS}) reached.")
        return True

    # Check if all *active* fragments have exhausted their pools
    pools_exhausted = all(fragment_pool_counts.get(f, 0) == 0 for f, s in fragment_states.items() if not s.get('is_complete', False))
    if pools_exhausted:
         # Check if there actually ARE active fragments. If all are complete, pools_exhausted might be true but doesn't matter.
         has_active_fragments = any(not s.get('is_complete', False) for s in fragment_states.values())
         if has_active_fragments:
              print("Assessment Complete Check: All active fragment candidate pools appear exhausted.")
              return True
         # else: all fragments are complete, caught by all_quotas_met check already

    return False

# You might also want to move get_fragment_question_map here if it's only used by update_fragment_states
def get_fragment_question_map(questions_table_data: List[Dict[str, Any]]) -> Dict[int, str]:
    """Creates a map of {question_id: fragment}."""
    q_map = {}
    for q in questions_table_data:
        q_id = q.get("id")
        fragments_str = q.get("fragments")
        if q_id and fragments_str and isinstance(fragments_str, str):
             try:
                 fragments_list = json.loads(fragments_str)
                 if isinstance(fragments_list, list) and fragments_list:
                      q_map[q_id] = fragments_list[0] # Assume first fragment listed is primary
             except json.JSONDecodeError:
                 if len(fragments_str) == 1: # Handle case like "A"
                      q_map[q_id] = fragments_str
    return q_map
