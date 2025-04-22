import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Define a structure (using dict for simplicity, could be a class)
# to hold the state for each fragment.
FragmentState = Dict[str, Any]

def initialize_fragment_states(fragments: List[str], initial_quotas: Dict[str, int]) -> Dict[str, FragmentState]:
    """Initializes the state dictionary for all fragments."""
    states = {}
    for frag_id in fragments:
        states[frag_id] = {
            "id": frag_id,
            "quota_assigned": initial_quotas.get(frag_id, 0),
            "quota_fulfilled": 0, # How many questions *stored* for this fragment
            "maturity_score": 0.0, # To be calculated each iteration
            "is_complete": False, # Flag if candidate pool is exhausted
            "candidate_pool_estimated_size": None, # Optional: could estimate based on retrieval
            # Add other state variables as needed (e.g., dependencies fulfilled)
        }
    return states

def calculate_maturity_score(
    fragment_state: FragmentState,
    # --- Parameters needed for calculation ---
    # Example: contextual_readiness_score: float = 0.0, # Calculated based on dependencies
    # Example: w1: float = 0.5, # Weight for readiness
    # Example: w2: float = 0.5, # Weight for completion
) -> float:
    """
    Calculates the maturity score for a single fragment.
    Placeholder implementation: Focuses on completion for now.
    Needs enhancement with contextual_readiness based on dependencies.
    """
    quota_assigned = fragment_state["quota_assigned"]
    quota_fulfilled = fragment_state["quota_fulfilled"]

    if quota_assigned <= 0:
        # Avoid division by zero, fragment might have 0 quota or be complete
        normalized_completion = 1.0
    else:
        normalized_completion = min(quota_fulfilled / quota_assigned, 1.0) # Cap at 1.0

    # Placeholder: Using only completion factor (lower completion = higher score)
    # Replace with the full formula: w1 * contextual_readiness + w2 * (1 - normalized_completion)
    # For now, fragments that are less complete get higher priority.
    maturity_score = 1.0 - normalized_completion

    # Add contextual readiness logic here when available
    # maturity_score = w1 * contextual_readiness_score + w2 * (1.0 - normalized_completion)

    # Ensure score is non-negative
    return max(maturity_score, 0.0)

def update_all_maturity_scores(
    fragment_states: Dict[str, FragmentState],
    # Add any other parameters needed for contextual readiness calculation
) -> None:
    """Updates the maturity score for all active fragments."""
    print("Updating maturity scores...")
    for frag_id, state in fragment_states.items():
        if not state["is_complete"] and state["quota_assigned"] > state["quota_fulfilled"]:
            # Calculate score only for active fragments that still need questions
            state["maturity_score"] = calculate_maturity_score(state)
            print(f"  Fragment {frag_id}: Maturity = {state['maturity_score']:.3f} (Fulfilled: {state['quota_fulfilled']}/{state['quota_assigned']})")
        else:
            # Mark score as irrelevant (e.g., -1 or 0) if complete or quota met
            state["maturity_score"] = -1.0
            status = "Complete (Pool Exhausted)" if state["is_complete"] else "Complete (Quota Met)"
            print(f"  Fragment {frag_id}: {status}")


def determine_iteration_targets(
    fragment_states: Dict[str, FragmentState],
    target_total_this_iteration: int
) -> Dict[str, int]:
    """
    Determines how many questions to target for each fragment in this iteration,
    based on maturity scores and remaining quotas.
    """
    targets = {frag_id: 0 for frag_id in fragment_states}
    
    # Filter for active fragments that still need questions and have positive maturity
    active_fragments = {
        frag_id: state for frag_id, state in fragment_states.items()
        if not state["is_complete"] and \
           state["quota_fulfilled"] < state["quota_assigned"] and \
           state["maturity_score"] >= 0 # Use >=0 to include just-started fragments
    }

    if not active_fragments or target_total_this_iteration <= 0:
        return targets # No targets if no active fragments or iteration target is 0

    # Calculate weights based on maturity scores (handle potential zero scores)
    maturity_scores = np.array([state["maturity_score"] for state in active_fragments.values()])
    
    # If all active fragments have score 0 (e.g., initial state), distribute evenly
    if np.sum(maturity_scores) == 0:
         weights = np.ones(len(maturity_scores)) / len(maturity_scores)
         print("Maturity scores are all zero, distributing targets evenly among active fragments.")
    else:
        # Softmax is good for converting scores to probabilities/weights
        # Adding a small epsilon to avoid issues with exactly zero scores if needed,
        # but maturity calculation should yield > 0 for incomplete fragments.
        exp_scores = np.exp(maturity_scores - np.max(maturity_scores)) # Stability trick
        weights = exp_scores / np.sum(exp_scores)
        print(f"Calculated weights based on maturity: {[f'{w:.3f}' for w in weights]}")


    # Distribute the target number of questions proportionally
    fragment_ids_active = list(active_fragments.keys())
    for i, frag_id in enumerate(fragment_ids_active):
        # Calculate proportional target, round for whole numbers
        proportional_target = math.ceil(weights[i] * target_total_this_iteration) # Ceiling ensures we aim high initially

        # Limit target by remaining quota for the fragment
        remaining_quota = fragment_states[frag_id]["quota_assigned"] - fragment_states[frag_id]["quota_fulfilled"]
        target = min(proportional_target, remaining_quota)
        targets[frag_id] = max(target, 0) # Ensure non-negative

    # Adjust if total targets exceed iteration target due to rounding up
    current_total_target = sum(targets.values())
    over_allocated = current_total_target - target_total_this_iteration

    # If overallocated, reduce targets starting from fragments with highest targets/lowest priority (lowest maturity)
    if over_allocated > 0:
        print(f"Target allocation ({current_total_target}) exceeds iteration limit ({target_total_this_iteration}). Adjusting...")
        # Sort fragments by target (desc), then maturity (asc) to remove excess fairly
        sorted_frags_for_reduction = sorted(
            fragment_ids_active,
            key=lambda fid: (targets[fid], -fragment_states[fid]["maturity_score"]), # Reduce large targets first, then low maturity
            reverse=True
        )
        
        removed_count = 0
        for frag_id in sorted_frags_for_reduction:
            reduction = min(targets[frag_id], over_allocated - removed_count)
            targets[frag_id] -= reduction
            removed_count += reduction
            if removed_count >= over_allocated:
                break
        print(f"Adjusted targets: {targets}")


    print(f"Determined targets for this iteration: {targets}")
    return targets

def redistribute_quotas(fragment_states: Dict[str, FragmentState]) -> None:
    """
    Identifies surplus quota from completed fragments and redistributes it
    proportionally to active, non-complete fragments based on maturity.
    """
    surplus_quota = 0
    fragments_needing_redistribution = []

    print("\nChecking for quota redistribution...")
    # Calculate total surplus from fragments that finished *without* fulfilling quota
    for frag_id, state in fragment_states.items():
        if state["is_complete"] and state["quota_fulfilled"] < state["quota_assigned"]:
            unfulfilled = state["quota_assigned"] - state["quota_fulfilled"]
            print(f"  Fragment {frag_id} completed with {unfulfilled} unfulfilled quota.")
            surplus_quota += unfulfilled
            # Set assigned quota to fulfilled to prevent further targeting
            state["quota_assigned"] = state["quota_fulfilled"]

    if surplus_quota == 0:
        print("  No surplus quota to redistribute.")
        return

    print(f"  Total surplus quota to redistribute: {surplus_quota}")

    # Identify eligible fragments to receive quota
    eligible_fragments = {
        frag_id: state for frag_id, state in fragment_states.items()
        if not state["is_complete"] and \
           state["quota_fulfilled"] < state["quota_assigned"] and \
           state["maturity_score"] >= 0 # Must be active and have a score
    }

    if not eligible_fragments:
        print("  No eligible fragments found to receive redistributed quota.")
        return

    # Calculate redistribution weights (using maturity scores of eligible fragments)
    maturity_scores = np.array([state["maturity_score"] for state in eligible_fragments.values()])
    if np.sum(maturity_scores) == 0: # Handle case where all eligible have 0 maturity
        weights = np.ones(len(maturity_scores)) / len(maturity_scores)
    else:
        exp_scores = np.exp(maturity_scores - np.max(maturity_scores)) # Stability
        weights = exp_scores / np.sum(exp_scores)

    print(f"  Redistributing surplus {surplus_quota} based on weights: {[f'{w:.3f}' for w in weights]}")

    # Add redistributed quota (integer amounts)
    distributed_total = 0
    eligible_ids = list(eligible_fragments.keys())
    for i, frag_id in enumerate(eligible_ids):
        add_quota = math.floor(weights[i] * surplus_quota) # Floor to avoid exceeding surplus initially
        fragment_states[frag_id]["quota_assigned"] += add_quota
        distributed_total += add_quota
        print(f"    Fragment {frag_id}: +{add_quota} quota (New total: {fragment_states[frag_id]['quota_assigned']})")

    # Distribute any remainder due to flooring (give to highest weight fragments)
    remainder = surplus_quota - distributed_total
    if remainder > 0:
        print(f"  Distributing remainder {remainder}...")
        sorted_indices = np.argsort(weights)[::-1] # Indices sorted by weight desc
        for i in range(remainder):
            frag_id_to_add = eligible_ids[sorted_indices[i % len(eligible_ids)]] # Cycle through top weighted if remainder > num_eligible
            fragment_states[frag_id_to_add]["quota_assigned"] += 1
            print(f"    Fragment {frag_id_to_add}: +1 quota (remainder) (New total: {fragment_states[frag_id_to_add]['quota_assigned']})")

    print("--- Quota redistribution complete ---")