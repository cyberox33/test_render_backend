import pandas as pd
from collections import Counter, defaultdict

def parse_ids_from_string(id_string: str) -> list[int]:
    """
    Helper to parse a semicolon-delimited string of IDs into a list of unique integers.
    Handles potential float representations (e.g., '99.0') and empty/NaN values.
    """
    if pd.isna(id_string) or str(id_string).strip() == "":
        return []
    ids = set()
    try:
        for i_str in str(id_string).split(';'):
            if i_str.strip():
                ids.add(int(float(i_str.strip()))) # Convert to float first, then int
    except ValueError as e:
        print(f"Warning: Could not parse ID string '{id_string}': {e}")
    return list(ids)

def analyze_rules_for_baseline_candidates(csv_filepath: str) -> pd.DataFrame:
    """
    Analyzes cleaned survey rules to identify potential baseline questions.

    Args:
        csv_filepath (str): Path to the cleaned survey rules CSV file.

    Returns:
        pd.DataFrame: DataFrame with question_id, controls_how_many_dependents, 
                      and is_dependent_how_many_times, ranked for baseline candidacy.
    """
    try:
        rules_df = pd.read_csv(csv_filepath, dtype=str) # Read all as string initially
        print(f"Successfully loaded {csv_filepath} with {len(rules_df)} rules.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading CSV file '{csv_filepath}': {e}")
        return pd.DataFrame()

    # Initialize dictionaries to store our metrics
    # For controls_how_many_dependents: maps independent_id to a set of dependent_ids it controls
    controls_dependents_map = defaultdict(set)
    # For is_dependent_how_many_times: counts occurrences as a dependent_id
    is_dependent_counts = Counter()
    
    all_question_ids = set()

    print("Processing rules to calculate influence and dependency...")
    for _, row in rules_df.iterrows():
        dep_id_str = row.get('dependent_question_id')
        indep_ids_str = row.get('independent_question_id')

        current_dep_ids = parse_ids_from_string(dep_id_str) # Should be one, but parse robustly
        
        if not current_dep_ids: # Skip if dependent_question_id is missing or invalid
            # This can happen if a rule (e.g. must_ask_first) has an empty dependent_id
            # which might indicate a structural rule not tied to a specific dependent q.
            # For this analysis, we are interested in direct q-to-q dependencies.
            pass
        
        current_dep_id = None
        if current_dep_ids: # Assuming dependent_question_id is always singular if present
            current_dep_id = current_dep_ids[0]
            is_dependent_counts[current_dep_id] += 1
            all_question_ids.add(current_dep_id)

        parsed_indep_ids = parse_ids_from_string(indep_ids_str)
        for indep_id in parsed_indep_ids:
            all_question_ids.add(indep_id)
            if current_dep_id is not None: # Only add to map if there's a dependent question
                controls_dependents_map[indep_id].add(current_dep_id)
    
    # Prepare data for DataFrame
    q_stats = []
    for qid in sorted(list(all_question_ids)):
        num_controlled = len(controls_dependents_map.get(qid, set()))
        num_dependent = is_dependent_counts.get(qid, 0)
        q_stats.append({
            'question_id': qid,
            'controls_how_many_dependents': num_controlled,
            'is_dependent_how_many_times': num_dependent
        })
    
    if not q_stats:
        print("No question statistics generated. Check CSV content and parsing.")
        return pd.DataFrame()

    stats_df = pd.DataFrame(q_stats)

    # Rank: Higher 'controls_how_many_dependents' is better.
    #       Lower 'is_dependent_how_many_times' is better for tie-breaking.
    ranked_df = stats_df.sort_values(
        by=['controls_how_many_dependents', 'is_dependent_how_many_times'],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    print("Analysis complete.")
    return ranked_df

# --- Main Execution ---
if __name__ == "__main__":
    # Assuming the cleaned rules CSV from the previous step is named 'cleaned_survey_rules.csv'
    # and is in the same directory as this script.
    input_csv_file = "cleaned_survey_rules.csv" 
    
    ranked_baseline_candidates_df = analyze_rules_for_baseline_candidates(input_csv_file)
    
    if not ranked_baseline_candidates_df.empty:
        print("\n--- Top 20 Potential Baseline Questions ---")
        print("(Ranked by: 1. How many unique dependents they control (more is better),")
        print("             2. How many times they are dependent (fewer is better))")
        
        # Select relevant columns for display
        display_df = ranked_baseline_candidates_df[[
            'question_id', 
            'controls_how_many_dependents', 
            'is_dependent_how_many_times'
        ]]
        print(display_df.head(20).to_string())
    else:
        print("\nCould not generate baseline candidate rankings.")