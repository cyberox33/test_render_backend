import pandas as pd
import io
from collections import Counter

def parse_ids_from_string(id_string):
    """Helper to parse semicolon-delimited string of IDs into a list of integers."""
    if pd.isna(id_string) or str(id_string).strip() == "":
        return []
    try:
        return [int(i.strip()) for i in str(id_string).split(';') if i.strip()]
    except ValueError as e:
        print(f"Warning: Could not parse ID string '{id_string}': {e}")
        return []

def clean_and_consolidate_rules(rules_plaintext_input: str) -> pd.DataFrame:
    """
    Cleans and consolidates survey dependency rules.
    Reduces symmetric 'must_ask_first' co-occurrence rules to a single asymmetric rule
    based on heuristic (frequency in other conditional rules, then numerical tie-break).
    """
    # 1. Parse the input plaintext into a DataFrame
    try:
        rules_df = pd.read_csv(io.StringIO(rules_plaintext_input), sep='\t')
    except Exception as e:
        print(f"Error reading input plaintext into DataFrame: {e}")
        return pd.DataFrame()

    # Ensure correct data types, especially for IDs
    id_cols = ['dependent_question_id', 'independent_question_id']
    for col in id_cols:
        if col in rules_df.columns:
            # Convert to string first to handle mixed types or errors, then to numeric
            rules_df[col] = rules_df[col].astype(str)


    # 2. Separate rule types
    # Conditional 'must_not_ask' rules (used for frequency count and kept as is)
    # A conditional MNA rule has a non-empty condition_answer_values
    conditional_mna_rules_df = rules_df[
        (rules_df['rule_type'] == 'must_not_ask') &
        (rules_df['condition_answer_values'].notna()) &
        (rules_df['condition_answer_values'].astype(str).str.strip() != '')
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    # 'Must_ask_first' rules intended for co-occurrence
    # These typically have empty condition_answer_values and condition_type 'none'
    co_occurrence_maf_rules_df = rules_df[
        (rules_df['rule_type'] == 'must_ask_first') &
        ((rules_df['condition_answer_values'].isna()) | (rules_df['condition_answer_values'].astype(str).str.strip() == '')) &
        (rules_df['condition_type'].astype(str).str.lower() == 'none')
    ].copy()

    # 3. Calculate independent_question_id frequencies from conditional_mna_rules
    print("\nCalculating independent_question_id frequencies from conditional 'must_not_ask' rules...")
    independent_counts = Counter()
    for _, row in conditional_mna_rules_df.iterrows():
        # independent_question_id in conditional_mna_rules can be single or multiple (';' separated)
        ids_in_rule = parse_ids_from_string(row['independent_question_id'])
        independent_counts.update(ids_in_rule)
    
    print(f"Frequencies calculated for {len(independent_counts)} unique independent question IDs.")
    # print("Sample Frequencies:", dict(independent_counts.most_common(5)))


    # 4. Identify unique co-occurrence pairs from co_occurrence_maf_rules_df
    print("\nIdentifying unique co-occurrence pairs from 'must_ask_first' rules...")
    co_occurrence_pairs = set()
    for _, row in co_occurrence_maf_rules_df.iterrows():
        try:
            # For these rules, independent_question_id should be a single ID string
            dep_id = int(float(row['dependent_question_id']))
            indep_id_str = str(row['independent_question_id']).split(';')[0].strip() # Take first if multiple, though not expected here
            indep_id = int(float(indep_id_str))
            # Add as a frozenset to handle order and ensure uniqueness
            co_occurrence_pairs.add(frozenset({dep_id, indep_id}))
        except ValueError as e:
            print(f"Warning: Could not parse pair from MAF rule: {row}. Error: {e}")
            continue
    print(f"Found {len(co_occurrence_pairs)} unique co-occurrence pairs.")

    # 5. Generate cleaned (single, asymmetric) 'must_ask_first' rules
    print("\nGenerating cleaned 'must_ask_first' rules for co-occurrence pairs...")
    cleaned_maf_rules_list = []
    for pair_set in co_occurrence_pairs:
        if len(pair_set) != 2: continue # Should always be 2
        q1, q2 = list(pair_set)

        count1 = independent_counts.get(q1, 0)
        count2 = independent_counts.get(q2, 0)

        final_dep_id, final_indep_id = -1, -1

        # Heuristic: Question with lower frequency as an independent_id in conditional rules
        # becomes the dependent_id (the one to be removed).
        # If frequencies are equal, the larger numerical ID becomes dependent_id (smaller ID is 'kept').
        if count1 < count2:
            final_dep_id, final_indep_id = q1, q2
        elif count2 < count1:
            final_dep_id, final_indep_id = q2, q1
        else: # Counts are equal, tie-break using numerical ID (larger ID becomes dependent)
            if q1 > q2:
                final_dep_id, final_indep_id = q1, q2
            else:
                final_dep_id, final_indep_id = q2, q1
        
        note = (f"{final_dep_id} and {final_indep_id} should not co-occur. "
                f"Based on heuristic, {final_dep_id} is removed if both present "
                f"(Influence Score: {final_indep_id}={independent_counts.get(final_indep_id,0)}, "
                f"{final_dep_id}={independent_counts.get(final_dep_id,0)}).")
        if count1 == count2:
            note += " (Tie-broken by ID)."

        cleaned_maf_rules_list.append({
            'dependent_question_id': final_dep_id,
            'independent_question_id': final_indep_id,
            'rule_type': 'must_ask_first',
            'condition_answer_values': '', # Explicitly empty
            'condition_type': 'none',
            'condition_logic': 'none',
            'note': note
        })
    
    cleaned_maf_df = pd.DataFrame(cleaned_maf_rules_list)
    print(f"Generated {len(cleaned_maf_df)} cleaned 'must_ask_first' rules.")

    # 6. Combine and Finalize
    print("\nCombining conditional 'must_not_ask' rules with cleaned 'must_ask_first' rules...")
    # Ensure conditional_mna_rules_df has string IDs for concat if types differ
    conditional_mna_rules_df['dependent_question_id'] = conditional_mna_rules_df['dependent_question_id'].astype(str)
    # independent_question_id can remain as string since it might contain ';'
    
    if not cleaned_maf_df.empty:
        cleaned_maf_df['dependent_question_id'] = cleaned_maf_df['dependent_question_id'].astype(str)
        cleaned_maf_df['independent_question_id'] = cleaned_maf_df['independent_question_id'].astype(str)
        final_rules_df = pd.concat([conditional_mna_rules_df, cleaned_maf_df], ignore_index=True)
    else: # If there were no co-occurrence rules to clean
        final_rules_df = conditional_mna_rules_df

    # Ensure all original columns are present
    expected_cols = ['dependent_question_id', 'independent_question_id', 'rule_type', 
                     'condition_answer_values', 'condition_type', 'condition_logic', 'note']
    for col in expected_cols:
        if col not in final_rules_df.columns:
            final_rules_df[col] = None # Add missing columns with None or appropriate default

    # Fill NaN for string-like columns that should be empty for MAF rules
    for col in ['condition_answer_values']:
        final_rules_df[col] = final_rules_df[col].fillna('')
    for col in ['condition_type', 'condition_logic']:
         final_rules_df[col] = final_rules_df[col].fillna('none')


    print(f"Total final rules: {len(final_rules_df)}.")
    return final_rules_df[expected_cols] # Reorder columns to match schema


# --- Main Execution ---
if __name__ == "__main__":
    # This is the plaintext output from your previous request.
    # For a real application, you might read this from a file or another source.
    plaintext_rules = """
dependent_question_id	independent_question_id	rule_type	condition_answer_values	condition_type	condition_logic	note
99	96	must_not_ask	Not Implemented	exact_match	none	99 should not be asked if 96 is Not Implemented
100	96	must_not_ask	Not Implemented	exact_match	none	100 should not be asked if 96 is Not Implemented
99	97	must_not_ask	Not Implemented	exact_match	none	99 should not be asked if 97 is Not Implemented
100	97	must_not_ask	Not Implemented	exact_match	none	100 should not be asked if 97 is Not Implemented
101	97	must_not_ask	Not Implemented	exact_match	none	101 should not be asked if 97 is Not Implemented
99	98	must_not_ask	Not Implemented	exact_match	none	99 should not be asked if 98 is Not Implemented
100	98	must_not_ask	Not Implemented	exact_match	none	100 should not be asked if 98 is Not Implemented
103	102	must_not_ask	Not Implemented	exact_match	none	103 should not be asked if 102 is Not Implemented
107	103	must_not_ask	Not Implemented	exact_match	none	107 should not be asked if 103 is Not Implemented
100	103	must_not_ask	Not Implemented	exact_match	none	100 should not be asked if 103 is Not Implemented
116	115	must_ask_first		none	none	116 and 115 should not co-occur (if both selected, 116 is targeted for removal by this rule instance)
115	116	must_ask_first		none	none	115 and 116 should not co-occur (if both selected, 115 is targeted for removal by this rule instance)
117	115	must_ask_first		none	none	117 and 115 should not co-occur (if both selected, 117 is targeted for removal by this rule instance)
115	117	must_ask_first		none	none	115 and 117 should not co-occur (if both selected, 115 is targeted for removal by this rule instance)
117	116	must_ask_first		none	none	117 and 116 should not co-occur (if both selected, 117 is targeted for removal by this rule instance)
116	117	must_ask_first		none	none	116 and 117 should not co-occur (if both selected, 116 is targeted for removal by this rule instance)
139	119	must_ask_first		none	none	139 and 119 should not co-occur (if both selected, 139 is targeted for removal by this rule instance)
119	139	must_ask_first		none	none	119 and 139 should not co-occur (if both selected, 119 is targeted for removal by this rule instance)
121	120	must_not_ask	Not Implemented	exact_match	none	121 should not be asked if 120 is Not Implemented
122	120	must_not_ask	Not Implemented	exact_match	none	122 should not be asked if 120 is Not Implemented
123	120	must_not_ask	Not Implemented	exact_match	none	123 should not be asked if 120 is Not Implemented
124	120	must_not_ask	Not Implemented	exact_match	none	124 should not be asked if 120 is Not Implemented
125	120	must_not_ask	Not Implemented	exact_match	none	125 should not be asked if 120 is Not Implemented
130	126	must_not_ask	Not Implemented	exact_match	none	130 should not be asked if 126 is Not Implemented
132	126	must_not_ask	Not Implemented	exact_match	none	132 should not be asked if 126 is Not Implemented
135	126	must_not_ask	Not Implemented	exact_match	none	135 should not be asked if 126 is Not Implemented
127	126	must_ask_first		none	none	127 and 126 should not co-occur (if both selected, 127 is targeted for removal by this rule instance)
126	127	must_ask_first		none	none	126 and 127 should not co-occur (if both selected, 126 is targeted for removal by this rule instance)
131	126	must_ask_first		none	none	131 and 126 should not co-occur (if both selected, 131 is targeted for removal by this rule instance)
126	131	must_ask_first		none	none	126 and 131 should not co-occur (if both selected, 126 is targeted for removal by this rule instance)
133	126	must_ask_first		none	none	133 and 126 should not co-occur (if both selected, 133 is targeted for removal by this rule instance)
126	133	must_ask_first		none	none	126 and 133 should not co-occur (if both selected, 126 is targeted for removal by this rule instance)
134	126	must_ask_first		none	none	134 and 126 should not co-occur (if both selected, 134 is targeted for removal by this rule instance)
126	134	must_ask_first		none	none	126 and 134 should not co-occur (if both selected, 126 is targeted for removal by this rule instance)
133	126;120;127;131	must_not_ask	Not Implemented	exact_match	match_all	133 should not be asked if 126, 120, 127, and 131 are all Not Implemented
134	126;120;133	must_not_ask	Not Implemented	exact_match	match_all	134 should not be asked if 126, 120, and 133 are all Not Implemented
146	140	must_not_ask	Not Implemented	exact_match	none	146 should not be asked if 140 is Not Implemented
151	140	must_not_ask	Not Implemented	exact_match	none	151 should not be asked if 140 is Not Implemented
161	140	must_not_ask	Not Implemented	exact_match	none	161 should not be asked if 140 is Not Implemented
162	140	must_not_ask	Not Implemented	exact_match	none	162 should not be asked if 140 is Not Implemented
140	146	must_not_ask	Not Implemented	exact_match	none	140 should not be asked if 146 is Not Implemented (vice versa)
140	151	must_not_ask	Not Implemented	exact_match	none	140 should not be asked if 151 is Not Implemented (vice versa)
140	161	must_not_ask	Not Implemented	exact_match	none	140 should not be asked if 161 is Not Implemented (vice versa)
140	162	must_not_ask	Not Implemented	exact_match	none	140 should not be asked if 162 is Not Implemented (vice versa)
145	141	must_not_ask	Not Implemented	exact_match	none	145 should not be asked if 141 is Not Implemented
157	141	must_not_ask	Not Implemented	exact_match	none	157 should not be asked if 141 is Not Implemented
141	145	must_not_ask	Not Implemented	exact_match	none	141 should not be asked if 145 is Not Implemented
157	145	must_not_ask	Not Implemented	exact_match	none	157 should not be asked if 145 is Not Implemented
145	141	must_ask_first		none	none	145 and 141 should not co-occur (if both selected, 145 is targeted for removal by this rule instance)
141	145	must_ask_first		none	none	141 and 145 should not co-occur (if both selected, 141 is targeted for removal by this rule instance)
155	142	must_not_ask	Not Implemented	exact_match	none	155 should not be asked if 142 is Not Implemented
152	142	must_ask_first		none	none	152 and 142 should not co-occur (if both selected, 152 is targeted for removal by this rule instance)
142	152	must_ask_first		none	none	142 and 152 should not co-occur (if both selected, 142 is targeted for removal by this rule instance)
158	143	must_ask_first		none	none	158 and 143 should not co-occur (if both selected, 158 is targeted for removal by this rule instance)
143	158	must_ask_first		none	none	143 and 158 should not co-occur (if both selected, 143 is targeted for removal by this rule instance)
158	143	must_not_ask	Not Implemented	exact_match	none	158 should not be asked if 143 is Not Implemented
160	143	must_not_ask	Not Implemented	exact_match	none	160 should not be asked if 143 is Not Implemented
143	158	must_not_ask	Not Implemented	exact_match	none	143 should not be asked if 158 is Not Implemented
160	158	must_not_ask	Not Implemented	exact_match	none	160 should not be asked if 158 is Not Implemented
143	160	must_not_ask	Not Implemented;None	exact_match	match_any	143 should not be asked if 160 is Not Implemented or None
158	160	must_not_ask	Not Implemented;None	exact_match	match_any	158 should not be asked if 160 is Not Implemented or None
161	160	must_ask_first		none	none	161 and 160 should not co-occur (if both selected, 161 is targeted for removal by this rule instance)
160	161	must_ask_first		none	none	160 and 161 should not co-occur (if both selected, 160 is targeted for removal by this rule instance)
156	147	must_ask_first		none	none	156 and 147 should not co-occur (if both selected, 156 is targeted for removal by this rule instance)
147	156	must_ask_first		none	none	147 and 156 should not co-occur (if both selected, 147 is targeted for removal by this rule instance)
156	147	must_not_ask	Not Implemented	exact_match	none	156 should not be asked if 147 is Not Implemented
147	156	must_not_ask	Not Implemented	exact_match	none	147 should not be asked if 156 is Not Implemented
159	147	must_ask_first		none	none	159 and 147 should not co-occur (if both selected, 159 is targeted for removal by this rule instance)
147	159	must_ask_first		none	none	147 and 159 should not co-occur (if both selected, 147 is targeted for removal by this rule instance)
159	156	must_ask_first		none	none	159 and 156 should not co-occur (if both selected, 159 is targeted for removal by this rule instance)
156	159	must_ask_first		none	none	156 and 159 should not co-occur (if both selected, 156 is targeted for removal by this rule instance)
153	148	must_ask_first		none	none	153 and 148 should not co-occur (if both selected, 153 is targeted for removal by this rule instance)
148	153	must_ask_first		none	none	148 and 153 should not co-occur (if both selected, 148 is targeted for removal by this rule instance)
152	150	must_not_ask	Not Implemented	exact_match	none	152 should not be asked if 150 is Not Implemented
150	152	must_not_ask	Not Implemented	exact_match	none	150 should not be asked if 152 is Not Implemented
162	146	must_ask_first		none	none	162 and 146 should not co-occur (if both selected, 162 is targeted for removal by this rule instance)
146	162	must_ask_first		none	none	146 and 162 should not co-occur (if both selected, 146 is targeted for removal by this rule instance)
178	166	must_not_ask	Not Implemented	exact_match	none	178 should not be asked if 166 is Not Implemented
180	166	must_not_ask	Not Implemented	exact_match	none	180 should not be asked if 166 is Not Implemented
177	167	must_not_ask	Not Implemented	exact_match	none	177 should not be asked if 167 is Not Implemented
174	168	must_not_ask	Not Implemented	exact_match	none	174 should not be asked if 168 is Not Implemented
168	167	must_ask_first		none	none	168 and 167 should not co-occur (if both selected, 168 is targeted for removal by this rule instance)
167	168	must_ask_first		none	none	167 and 168 should not co-occur (if both selected, 167 is targeted for removal by this rule instance)
169	167	must_ask_first		none	none	169 and 167 should not co-occur (if both selected, 169 is targeted for removal by this rule instance)
167	169	must_ask_first		none	none	167 and 169 should not co-occur (if both selected, 167 is targeted for removal by this rule instance)
169	168	must_ask_first		none	none	169 and 168 should not co-occur (if both selected, 169 is targeted for removal by this rule instance)
168	169	must_ask_first		none	none	168 and 169 should not co-occur (if both selected, 168 is targeted for removal by this rule instance)
179	169	must_not_ask	Not Implemented	exact_match	none	179 should not be asked if 169 is Not Implemented
181	169	must_not_ask	Not Implemented	exact_match	none	181 should not be asked if 169 is Not Implemented
185	171	must_not_ask	Not Implemented	exact_match	none	185 should not be asked if 171 is Not Implemented
172	171	must_ask_first		none	none	172 and 171 should not co-occur (if both selected, 172 is targeted for removal by this rule instance)
171	172	must_ask_first		none	none	171 and 172 should not co-occur (if both selected, 171 is targeted for removal by this rule instance)
173	171	must_ask_first		none	none	173 and 171 should not co-occur (if both selected, 173 is targeted for removal by this rule instance)
171	173	must_ask_first		none	none	171 and 173 should not co-occur (if both selected, 171 is targeted for removal by this rule instance)
174	171	must_ask_first		none	none	174 and 171 should not co-occur (if both selected, 174 is targeted for removal by this rule instance)
171	174	must_ask_first		none	none	171 and 174 should not co-occur (if both selected, 171 is targeted for removal by this rule instance)
173	172	must_ask_first		none	none	173 and 172 should not co-occur (if both selected, 173 is targeted for removal by this rule instance)
172	173	must_ask_first		none	none	172 and 173 should not co-occur (if both selected, 172 is targeted for removal by this rule instance)
174	172	must_ask_first		none	none	174 and 172 should not co-occur (if both selected, 174 is targeted for removal by this rule instance)
172	174	must_ask_first		none	none	172 and 174 should not co-occur (if both selected, 172 is targeted for removal by this rule instance)
174	173	must_ask_first		none	none	174 and 173 should not co-occur (if both selected, 174 is targeted for removal by this rule instance)
173	174	must_ask_first		none	none	173 and 174 should not co-occur (if both selected, 173 is targeted for removal by this rule instance)
173	172	must_not_ask	Not Implemented	exact_match	none	173 should not be asked if 172 is Not Implemented
172	173	must_not_ask	Not Implemented	exact_match	none	172 should not be asked if 173 is Not Implemented
179	172	must_ask_first		none	none	179 and 172 should not co-occur (if both selected, 179 is targeted for removal by this rule instance)
172	179	must_ask_first		none	none	172 and 179 should not co-occur (if both selected, 172 is targeted for removal by this rule instance)
181	172	must_ask_first		none	none	181 and 172 should not co-occur (if both selected, 181 is targeted for removal by this rule instance)
172	181	must_ask_first		none	none	172 and 181 should not co-occur (if both selected, 172 is targeted for removal by this rule instance)
179	173	must_ask_first		none	none	179 and 173 should not co-occur (if both selected, 179 is targeted for removal by this rule instance)
173	179	must_ask_first		none	none	173 and 179 should not co-occur (if both selected, 173 is targeted for removal by this rule instance)
181	173	must_ask_first		none	none	181 and 173 should not co-occur (if both selected, 181 is targeted for removal by this rule instance)
173	181	must_ask_first		none	none	173 and 181 should not co-occur (if both selected, 173 is targeted for removal by this rule instance)
172	178	must_not_ask	Not Implemented	exact_match	none	172 should not be asked if 178 is Not Implemented
173	178	must_not_ask	Not Implemented	exact_match	none	173 should not be asked if 178 is Not Implemented
179	178	must_not_ask	Not Implemented	exact_match	none	179 should not be asked if 178 is Not Implemented
180	178	must_not_ask	Not Implemented	exact_match	none	180 should not be asked if 178 is Not Implemented
181	178	must_not_ask	Not Implemented	exact_match	none	181 should not be asked if 178 is Not Implemented
181	179	must_not_ask	Not Implemented	exact_match	none	181 should not be asked if 179 is Not Implemented
179	181	must_not_ask	Not Implemented	exact_match	none	179 should not be asked if 181 is Not Implemented
215	212	must_not_ask	Not Implemented	exact_match	none	215 should not be asked if 212 is Not Implemented
216	212	must_not_ask	Not Implemented	exact_match	none	216 should not be asked if 212 is Not Implemented
230	212	must_not_ask	Not Implemented	exact_match	none	230 should not be asked if 212 is Not Implemented
231	212	must_not_ask	Not Implemented	exact_match	none	231 should not be asked if 212 is Not Implemented
232	212	must_not_ask	Not Implemented	exact_match	none	232 should not be asked if 212 is Not Implemented
231	213	must_not_ask	Not Implemented	exact_match	none	231 should not be asked if 213 is Not Implemented
213	231	must_not_ask	Not Implemented	exact_match	none	213 should not be asked if 231 is Not Implemented
231	213	must_ask_first		none	none	231 and 213 should not co-occur (if both selected, 231 is targeted for removal by this rule instance)
213	231	must_ask_first		none	none	213 and 231 should not co-occur (if both selected, 213 is targeted for removal by this rule instance)
216	214	must_not_ask	Not Implemented	exact_match	none	216 should not be asked if 214 is Not Implemented (assuming 'Q214 &' was a typo for the dependent Q)
223	222	must_not_ask	Not Implemented	exact_match	none	223 should not be asked if 222 is Not Implemented
223	222	must_ask_first		none	none	223 and 222 should not co-occur (if both selected, 223 is targeted for removal by this rule instance)
222	223	must_ask_first		none	none	222 and 223 should not co-occur (if both selected, 222 is targeted for removal by this rule instance)
238	237	must_not_ask	Not Implemented	exact_match	none	238 should not be asked if 237 is Not Implemented
241	237	must_not_ask	Not Implemented	exact_match	none	241 should not be asked if 237 is Not Implemented
260	237	must_not_ask	Not Implemented	exact_match	none	260 should not be asked if 237 is Not Implemented
261	237	must_not_ask	Not Implemented	exact_match	none	261 should not be asked if 237 is Not Implemented
254	240	must_ask_first		none	none	254 and 240 should not co-occur (if both selected, 254 is targeted for removal by this rule instance)
240	254	must_ask_first		none	none	240 and 254 should not co-occur (if both selected, 240 is targeted for removal by this rule instance)
254	240	must_not_ask	Not Implemented	exact_match	none	254 should not be asked if 240 is Not Implemented
256	240	must_not_ask	Not Implemented	exact_match	none	256 should not be asked if 240 is Not Implemented
240	254	must_not_ask	Not Implemented	exact_match	none	240 should not be asked if 254 is Not Implemented
256	254	must_not_ask	Not Implemented	exact_match	none	256 should not be asked if 254 is Not Implemented
256	255	must_not_ask	Not Implemented	exact_match	none	256 should not be asked if 255 is Not Implemented
256	255	must_ask_first		none	none	256 and 255 should not co-occur (if both selected, 256 is targeted for removal by this rule instance)
255	256	must_ask_first		none	none	255 and 256 should not co-occur (if both selected, 255 is targeted for removal by this rule instance)
255	256	must_not_ask	Not Implemented	exact_match	none	255 should not be asked if 256 is Not Implemented
243	242	must_not_ask	Not Implemented	exact_match	none	243 should not be asked if 242 is Not Implemented
244	242	must_not_ask	Not Implemented	exact_match	none	244 should not be asked if 242 is Not Implemented
243	242	must_ask_first		none	none	243 and 242 should not co-occur (if both selected, 243 is targeted for removal by this rule instance)
242	243	must_ask_first		none	none	242 and 243 should not co-occur (if both selected, 242 is targeted for removal by this rule instance)
244	242	must_ask_first		none	none	244 and 242 should not co-occur (if both selected, 244 is targeted for removal by this rule instance)
242	244	must_ask_first		none	none	242 and 244 should not co-occur (if both selected, 242 is targeted for removal by this rule instance)
253	245	must_ask_first		none	none	253 and 245 should not co-occur (if both selected, 253 is targeted for removal by this rule instance)
245	253	must_ask_first		none	none	245 and 253 should not co-occur (if both selected, 245 is targeted for removal by this rule instance)
253	245	must_not_ask	Not Implemented	exact_match	none	253 should not be asked if 245 is Not Implemented
245	253	must_not_ask	Not Implemented	exact_match	none	245 should not be asked if 253 is Not Implemented
252	251	must_ask_first		none	none	252 and 251 should not co-occur (if both selected, 252 is targeted for removal by this rule instance)
251	252	must_ask_first		none	none	251 and 252 should not co-occur (if both selected, 251 is targeted for removal by this rule instance)
252	251	must_not_ask	Not Implemented	exact_match	none	252 should not be asked if 251 is Not Implemented
251	252	must_not_ask	Not Implemented	exact_match	none	251 should not be asked if 252 is Not Implemented
261	260	must_ask_first		none	none	261 and 260 should not co-occur (if both selected, 261 is targeted for removal by this rule instance)
260	261	must_ask_first		none	none	260 and 261 should not co-occur (if both selected, 260 is targeted for removal by this rule instance)
261	260	must_not_ask	Not Implemented	exact_match	none	261 should not be asked if 260 is Not Implemented
260	261	must_not_ask	Not Implemented	exact_match	none	260 should not be asked if 261 is Not Implemented
343	338	must_ask_first		none	none	343 and 338 should not co-occur (if both selected, 343 is targeted for removal by this rule instance)
338	343	must_ask_first		none	none	338 and 343 should not co-occur (if both selected, 338 is targeted for removal by this rule instance)
358	338	must_ask_first		none	none	358 and 338 should not co-occur (if both selected, 358 is targeted for removal by this rule instance)
338	358	must_ask_first		none	none	338 and 358 should not co-occur (if both selected, 338 is targeted for removal by this rule instance)
360	338	must_ask_first		none	none	360 and 338 should not co-occur (if both selected, 360 is targeted for removal by this rule instance)
338	360	must_ask_first		none	none	338 and 360 should not co-occur (if both selected, 338 is targeted for removal by this rule instance)
345	340	must_ask_first		none	none	345 and 340 should not co-occur (if both selected, 345 is targeted for removal by this rule instance)
340	345	must_ask_first		none	none	340 and 345 should not co-occur (if both selected, 340 is targeted for removal by this rule instance)
345	340	must_not_ask	Not Implemented	exact_match	none	345 should not be asked if 340 is Not Implemented
340	345	must_not_ask	Not Implemented	exact_match	none	340 should not be asked if 345 is Not Implemented
354	340	must_ask_first		none	none	354 and 340 should not co-occur (if both selected, 354 is targeted for removal by this rule instance)
340	354	must_ask_first		none	none	340 and 354 should not co-occur (if both selected, 340 is targeted for removal by this rule instance)
347	341	must_ask_first		none	none	347 and 341 should not co-occur (if both selected, 347 is targeted for removal by this rule instance)
341	347	must_ask_first		none	none	341 and 347 should not co-occur (if both selected, 341 is targeted for removal by this rule instance)
352	347	must_ask_first		none	none	352 and 347 should not co-occur (if both selected, 352 is targeted for removal by this rule instance)
347	352	must_ask_first		none	none	347 and 352 should not co-occur (if both selected, 347 is targeted for removal by this rule instance)
352	341	must_ask_first		none	none	352 and 341 should not co-occur (if both selected, 352 is targeted for removal by this rule instance)
341	352	must_ask_first		none	none	341 and 352 should not co-occur (if both selected, 341 is targeted for removal by this rule instance)
347	341	must_not_ask	Not Implemented	exact_match	none	347 should not be asked if 341 is Not Implemented
341	347	must_not_ask	Not Implemented	exact_match	none	341 should not be asked if 347 is Not Implemented
352	345	must_ask_first		none	none	352 and 345 should not co-occur (if both selected, 352 is targeted for removal by this rule instance)
345	352	must_ask_first		none	none	345 and 352 should not co-occur (if both selected, 345 is targeted for removal by this rule instance)
343	358	must_not_ask	Not Implemented	exact_match	none	343 should not be asked if 358 is Not Implemented
354	358	must_not_ask	Not Implemented	exact_match	none	354 should not be asked if 358 is Not Implemented
360	358	must_not_ask	Not Implemented	exact_match	none	360 should not be asked if 358 is Not Implemented
343	358	must_ask_first		none	none	343 and 358 should not co-occur (if both selected, 343 is targeted for removal by this rule instance)
358	343	must_ask_first		none	none	358 and 343 should not co-occur (if both selected, 358 is targeted for removal by this rule instance)
354	358	must_ask_first		none	none	354 and 358 should not co-occur (if both selected, 354 is targeted for removal by this rule instance)
358	354	must_ask_first		none	none	358 and 354 should not co-occur (if both selected, 358 is targeted for removal by this rule instance)
360	358	must_ask_first		none	none	360 and 358 should not co-occur (if both selected, 360 is targeted for removal by this rule instance)
358	360	must_ask_first		none	none	358 and 360 should not co-occur (if both selected, 358 is targeted for removal by this rule instance)
317	312	must_ask_first		none	none	317 and 312 should not co-occur (if both selected, 317 is targeted for removal by this rule instance)
312	317	must_ask_first		none	none	312 and 317 should not co-occur (if both selected, 312 is targeted for removal by this rule instance)
330	314	must_ask_first		none	none	330 and 314 should not co-occur (if both selected, 330 is targeted for removal by this rule instance)
314	330	must_ask_first		none	none	314 and 330 should not co-occur (if both selected, 314 is targeted for removal by this rule instance)
322	315	must_ask_first		none	none	322 and 315 should not co-occur (if both selected, 322 is targeted for removal by this rule instance)
315	322	must_ask_first		none	none	315 and 322 should not co-occur (if both selected, 315 is targeted for removal by this rule instance)
329	315	must_ask_first		none	none	329 and 315 should not co-occur (if both selected, 329 is targeted for removal by this rule instance)
315	329	must_ask_first		none	none	315 and 329 should not co-occur (if both selected, 315 is targeted for removal by this rule instance)
329	322	must_ask_first		none	none	329 and 322 should not co-occur (if both selected, 329 is targeted for removal by this rule instance)
322	329	must_ask_first		none	none	322 and 329 should not co-occur (if both selected, 322 is targeted for removal by this rule instance)
322	315	must_not_ask	Not Implemented	exact_match	none	322 should not be asked if 315 is Not Implemented
329	315	must_not_ask	Not Implemented	exact_match	none	329 should not be asked if 315 is Not Implemented
315	322	must_not_ask	Not Implemented	exact_match	none	315 should not be asked if 322 is Not Implemented
329	322	must_not_ask	Not Implemented	exact_match	none	329 should not be asked if 322 is Not Implemented
332	316	must_ask_first		none	none	332 and 316 should not co-occur (if both selected, 332 is targeted for removal by this rule instance)
316	332	must_ask_first		none	none	316 and 332 should not co-occur (if both selected, 316 is targeted for removal by this rule instance)
332	316	must_not_ask	Not Implemented	exact_match	none	332 should not be asked if 316 is Not Implemented
316	332	must_not_ask	Not Implemented	exact_match	none	316 should not be asked if 332 is Not Implemented
331	319	must_ask_first		none	none	331 and 319 should not co-occur (if both selected, 331 is targeted for removal by this rule instance)
319	331	must_ask_first		none	none	319 and 331 should not co-occur (if both selected, 319 is targeted for removal by this rule instance)
331	319	must_not_ask	Not Implemented	exact_match	none	331 should not be asked if 319 is Not Implemented
319	331	must_not_ask	Not Implemented	exact_match	none	319 should not be asked if 331 is Not Implemented
330	323	must_ask_first		none	none	330 and 323 should not co-occur (if both selected, 330 is targeted for removal by this rule instance)
323	330	must_ask_first		none	none	323 and 330 should not co-occur (if both selected, 323 is targeted for removal by this rule instance)
333	325	must_ask_first		none	none	333 and 325 should not co-occur (if both selected, 333 is targeted for removal by this rule instance)
325	333	must_ask_first		none	none	325 and 333 should not co-occur (if both selected, 325 is targeted for removal by this rule instance)
333	325	must_not_ask	Not Implemented	exact_match	none	333 should not be asked if 325 is Not Implemented
325	333	must_not_ask	Not Implemented	exact_match	none	325 should not be asked if 333 is Not Implemented
335	328	must_ask_first		none	none	335 and 328 should not co-occur (if both selected, 335 is targeted for removal by this rule instance)
328	335	must_ask_first		none	none	328 and 335 should not co-occur (if both selected, 328 is targeted for removal by this rule instance)
335	328	must_not_ask	Not Implemented	exact_match	none	335 should not be asked if 328 is Not Implemented
328	335	must_not_ask	Not Implemented	exact_match	none	328 should not be asked if 335 is Not Implemented
275	262	must_ask_first		none	none	275 and 262 should not co-occur (if both selected, 275 is targeted for removal by this rule instance)
262	275	must_ask_first		none	none	262 and 275 should not co-occur (if both selected, 262 is targeted for removal by this rule instance)
263	262	must_ask_first		none	none	263 and 262 should not co-occur (if both selected, 263 is targeted for removal by this rule instance)
262	263	must_ask_first		none	none	262 and 263 should not co-occur (if both selected, 262 is targeted for removal by this rule instance)
265	262	must_ask_first		none	none	265 and 262 should not co-occur (if both selected, 265 is targeted for removal by this rule instance)
262	265	must_ask_first		none	none	262 and 265 should not co-occur (if both selected, 262 is targeted for removal by this rule instance)
263	262	must_not_ask	Not Implemented	exact_match	none	263 should not be asked if 262 is Not Implemented
265	262	must_not_ask	Not Implemented	exact_match	none	265 should not be asked if 262 is Not Implemented
275	262	must_not_ask	Not Implemented	exact_match	none	275 should not be asked if 262 is Not Implemented
262	263	must_not_ask	Not Implemented	exact_match	none	262 should not be asked if 263 is Not Implemented
265	263	must_not_ask	Not Implemented	exact_match	none	265 should not be asked if 263 is Not Implemented
275	263	must_not_ask	Not Implemented	exact_match	none	275 should not be asked if 263 is Not Implemented
265	263	must_ask_first		none	none	265 and 263 should not co-occur (if both selected, 265 is targeted for removal by this rule instance)
263	265	must_ask_first		none	none	263 and 265 should not co-occur (if both selected, 263 is targeted for removal by this rule instance)
275	263	must_ask_first		none	none	275 and 263 should not co-occur (if both selected, 275 is targeted for removal by this rule instance)
263	275	must_ask_first		none	none	263 and 275 should not co-occur (if both selected, 263 is targeted for removal by this rule instance)
273	264	must_not_ask	Not Implemented	exact_match	none	273 should not be asked if 264 (baseline q) is Not Implemented
274	264	must_not_ask	Not Implemented	exact_match	none	274 should not be asked if 264 (baseline q) is Not Implemented
283	264	must_not_ask	Not Implemented	exact_match	none	283 should not be asked if 264 (baseline q) is Not Implemented
284	264	must_not_ask	Not Implemented	exact_match	none	284 should not be asked if 264 (baseline q) is Not Implemented
273	274	must_not_ask	Not Implemented	exact_match	none	273 should not be asked if 274 is Not Implemented
283	274	must_not_ask	Not Implemented	exact_match	none	283 should not be asked if 274 is Not Implemented
284	274	must_not_ask	Not Implemented	exact_match	none	284 should not be asked if 274 is Not Implemented
274	283	must_not_ask	Not Implemented	exact_match	none	274 should not be asked if 283 is Not Implemented
273	283	must_not_ask	Not Implemented	exact_match	none	273 should not be asked if 283 is Not Implemented
284	283	must_not_ask	Not Implemented	exact_match	none	284 should not be asked if 283 is Not Implemented
283	274	must_ask_first		none	none	283 and 274 should not co-occur (if both selected, 283 is targeted for removal by this rule instance)
274	283	must_ask_first		none	none	274 and 283 should not co-occur (if both selected, 274 is targeted for removal by this rule instance)
284	273	must_ask_first		none	none	284 and 273 should not co-occur (if both selected, 284 is targeted for removal by this rule instance)
273	284	must_ask_first		none	none	273 and 284 should not co-occur (if both selected, 273 is targeted for removal by this rule instance)
280	273	must_not_ask	Not integrated with DevOps	exact_match	none	280 should not be asked if 273 is 'Not integrated with DevOps'
284	273	must_not_ask	Not integrated with DevOps	exact_match	none	284 should not be asked if 273 is 'Not integrated with DevOps'
284	280	must_not_ask	Manual unit testing procedures	exact_match	none	284 should not be asked if 280 is 'Manual unit testing procedures'
280	284	must_not_ask	Manual validation processes	exact_match	none	280 should not be asked if 284 is 'Manual validation processes'
280	273	must_ask_first		none	none	280 and 273 should not co-occur (if both selected, 280 is targeted for removal by this rule instance)
273	280	must_ask_first		none	none	273 and 280 should not co-occur (if both selected, 273 is targeted for removal by this rule instance)
284	280	must_ask_first		none	none	284 and 280 should not co-occur (if both selected, 284 is targeted for removal by this rule instance)
280	284	must_ask_first		none	none	280 and 284 should not co-occur (if both selected, 280 is targeted for removal by this rule instance)
274	273	must_ask_first		none	none	274 and 273 should not co-occur (if both selected, 274 is targeted for removal by this rule instance)
273	274	must_ask_first		none	none	273 and 274 should not co-occur (if both selected, 273 is targeted for removal by this rule instance)
283	273	must_ask_first		none	none	283 and 273 should not co-occur (if both selected, 283 is targeted for removal by this rule instance)
273	283	must_ask_first		none	none	273 and 283 should not co-occur (if both selected, 273 is targeted for removal by this rule instance)
284	283	must_ask_first		none	none	284 and 283 should not co-occur (if both selected, 284 is targeted for removal by this rule instance)
283	284	must_ask_first		none	none	283 and 284 should not co-occur (if both selected, 283 is targeted for removal by this rule instance)
269	265	must_not_ask	Not Implemented	exact_match	none	269 should not be asked if 265 is Not Implemented
272	265	must_not_ask	Not Implemented	exact_match	none	272 should not be asked if 265 is Not Implemented
282	265	must_not_ask	Not Implemented	exact_match	none	282 should not be asked if 265 is Not Implemented
269	265	must_ask_first		none	none	269 and 265 should not co-occur (if both selected, 269 is targeted for removal by this rule instance)
265	269	must_ask_first		none	none	265 and 269 should not co-occur (if both selected, 265 is targeted for removal by this rule instance)
272	265	must_ask_first		none	none	272 and 265 should not co-occur (if both selected, 272 is targeted for removal by this rule instance)
265	272	must_ask_first		none	none	265 and 272 should not co-occur (if both selected, 265 is targeted for removal by this rule instance)
282	265	must_ask_first		none	none	282 and 265 should not co-occur (if both selected, 282 is targeted for removal by this rule instance)
265	282	must_ask_first		none	none	265 and 282 should not co-occur (if both selected, 265 is targeted for removal by this rule instance)
272	269	must_ask_first		none	none	272 and 269 should not co-occur (if both selected, 272 is targeted for removal by this rule instance)
269	272	must_ask_first		none	none	269 and 272 should not co-occur (if both selected, 269 is targeted for removal by this rule instance)
282	269	must_ask_first		none	none	282 and 269 should not co-occur (if both selected, 282 is targeted for removal by this rule instance)
269	282	must_ask_first		none	none	269 and 282 should not co-occur (if both selected, 269 is targeted for removal by this rule instance)
282	272	must_ask_first		none	none	282 and 272 should not co-occur (if both selected, 282 is targeted for removal by this rule instance)
272	282	must_ask_first		none	none	272 and 282 should not co-occur (if both selected, 272 is targeted for removal by this rule instance)
282	272	must_not_ask	Not Implemented	exact_match	none	282 should not be asked if 272 is Not Implemented
276	267	must_ask_first		none	none	276 and 267 should not co-occur (if both selected, 276 is targeted for removal by this rule instance)
267	276	must_ask_first		none	none	267 and 276 should not co-occur (if both selected, 267 is targeted for removal by this rule instance)
281	267	must_ask_first		none	none	281 and 267 should not co-occur (if both selected, 281 is targeted for removal by this rule instance)
267	281	must_ask_first		none	none	267 and 281 should not co-occur (if both selected, 267 is targeted for removal by this rule instance)
281	276	must_ask_first		none	none	281 and 276 should not co-occur (if both selected, 281 is targeted for removal by this rule instance)
276	281	must_ask_first		none	none	276 and 281 should not co-occur (if both selected, 276 is targeted for removal by this rule instance)
276	267	must_not_ask	Not Implemented	exact_match	none	276 should not be asked if 267 is Not Implemented
281	267	must_not_ask	Not Implemented	exact_match	none	281 should not be asked if 267 is Not Implemented
267	276	must_not_ask	Not Implemented	exact_match	none	267 should not be asked if 276 is Not Implemented
281	276	must_not_ask	Not Implemented	exact_match	none	281 should not be asked if 276 is Not Implemented
267	281	must_not_ask	Not Implemented	exact_match	none	267 should not be asked if 281 is Not Implemented
276	281	must_not_ask	Not Implemented	exact_match	none	276 should not be asked if 281 is Not Implemented
286	268	must_ask_first		none	none	286 and 268 should not co-occur (if both selected, 286 is targeted for removal by this rule instance)
268	286	must_ask_first		none	none	268 and 286 should not co-occur (if both selected, 268 is targeted for removal by this rule instance)
286	268	must_not_ask	Not Implemented	exact_match	none	286 should not be asked if 268 is Not Implemented
268	286	must_not_ask	Not Implemented	exact_match	none	268 should not be asked if 286 is Not Implemented
279	269	must_ask_first		none	none	279 and 269 should not co-occur (if both selected, 279 is targeted for removal by this rule instance)
269	279	must_ask_first		none	none	269 and 279 should not co-occur (if both selected, 269 is targeted for removal by this rule instance)
279	269	must_not_ask	Not Implemented	exact_match	none	279 should not be asked if 269 is Not Implemented
269	279	must_not_ask	Not Implemented	exact_match	none	269 should not be asked if 279 is Not Implemented
285	277	must_ask_first		none	none	285 and 277 should not co-occur (if both selected, 285 is targeted for removal by this rule instance)
277	285	must_ask_first		none	none	277 and 285 should not co-occur (if both selected, 277 is targeted for removal by this rule instance)
269	277	must_ask_first		none	none	269 and 277 should not co-occur (if both selected, 269 is targeted for removal by this rule instance)
277	269	must_ask_first		none	none	277 and 269 should not co-occur (if both selected, 277 is targeted for removal by this rule instance)
279	277	must_ask_first		none	none	279 and 277 should not co-occur (if both selected, 279 is targeted for removal by this rule instance)
277	279	must_ask_first		none	none	277 and 279 should not co-occur (if both selected, 277 is targeted for removal by this rule instance)
279	277	must_not_ask	Not Implemented	exact_match	none	279 should not be asked if 277 is Not Implemented
285	277	must_not_ask	Not Implemented	exact_match	none	285 should not be asked if 277 is Not Implemented
277	285	must_not_ask	Not Implemented	exact_match	none	277 should not be asked if 285 is Not Implemented
279	285	must_not_ask	Not Implemented	exact_match	none	279 should not be asked if 285 is Not Implemented
279	285	must_ask_first		none	none	279 and 285 should not co-occur (if both selected, 279 is targeted for removal by this rule instance)
285	279	must_ask_first		none	none	285 and 279 should not co-occur (if both selected, 285 is targeted for removal by this rule instance)
269	285	must_ask_first		none	none	269 and 285 should not co-occur (if both selected, 269 is targeted for removal by this rule instance)
285	269	must_ask_first		none	none	285 and 269 should not co-occur (if both selected, 285 is targeted for removal by this rule instance)
288	287	must_not_ask	Not Implemented	exact_match	none	288 should not be asked if 287 is Not Implemented
291	287	must_not_ask	Not Implemented	exact_match	none	291 should not be asked if 287 is Not Implemented
297	287	must_not_ask	Not Implemented	exact_match	none	297 should not be asked if 287 is Not Implemented
298	287	must_not_ask	Not Implemented	exact_match	none	298 should not be asked if 287 is Not Implemented
300	287	must_not_ask	Not Implemented	exact_match	none	300 should not be asked if 287 is Not Implemented
301	287	must_not_ask	Not Implemented	exact_match	none	301 should not be asked if 287 is Not Implemented
303	287	must_not_ask	Not Implemented	exact_match	none	303 should not be asked if 287 is Not Implemented
304	287	must_not_ask	Not Implemented	exact_match	none	304 should not be asked if 287 is Not Implemented
310	287	must_not_ask	Not Implemented	exact_match	none	310 should not be asked if 287 is Not Implemented
311	287	must_not_ask	Not Implemented	exact_match	none	311 should not be asked if 287 is Not Implemented
298	288	must_not_ask	Not Implemented	exact_match	none	298 should not be asked if 288 is Not Implemented
309	288	must_not_ask	Not Implemented	exact_match	none	309 should not be asked if 288 is Not Implemented
310	288	must_not_ask	Not Implemented	exact_match	none	310 should not be asked if 288 is Not Implemented
293	289	must_not_ask	Not Implemented	exact_match	none	293 should not be asked if 289 is Not Implemented
295	290	must_not_ask	no	exact_match	none	295 should not be asked if 290 is 'no'
302	290	must_ask_first		none	none	302 and 290 should not co-occur (if both selected, 302 is targeted for removal by this rule instance)
290	302	must_ask_first		none	none	290 and 302 should not co-occur (if both selected, 290 is targeted for removal by this rule instance)
307	290	must_ask_first		none	none	307 and 290 should not co-occur (if both selected, 307 is targeted for removal by this rule instance)
290	307	must_ask_first		none	none	290 and 307 should not co-occur (if both selected, 290 is targeted for removal by this rule instance)
297	291	must_not_ask	Not Implemented	exact_match	none	297 should not be asked if 291 is Not Implemented
303	291	must_not_ask	Not Implemented	exact_match	none	303 should not be asked if 291 is Not Implemented
310	291	must_not_ask	Not Implemented	exact_match	none	310 should not be asked if 291 is Not Implemented
288	298	must_not_ask	Not Implemented	exact_match	none	288 should not be asked if 298 is Not Implemented
299	292	must_not_ask	Not Implemented	exact_match	none	299 should not be asked if 292 is Not Implemented
301	292	must_not_ask	Not Implemented	exact_match	none	301 should not be asked if 292 is Not Implemented
311	292	must_not_ask	Not Implemented	exact_match	none	311 should not be asked if 292 is Not Implemented
305	293	must_not_ask	We perform dependency scanning, but we may not always address identified vulnerabilities promptly.;We do not have a dedicated process for dependency scanning in our deployment.	exact_match	match_any	305 should not be asked if 293 is 'We perform dependency scanning...' or 'We do not have a dedicated process...'
290	295	must_not_ask	Our deployment process does not currently prioritize resource efficiency.	exact_match	none	290 should not be asked if 295 is 'Our deployment process does not currently prioritize resource efficiency.'
302	295	must_not_ask	Our deployment process does not currently prioritize resource efficiency.	exact_match	none	302 should not be asked if 295 is 'Our deployment process does not currently prioritize resource efficiency.'
307	295	must_not_ask	Our deployment process does not currently prioritize resource efficiency.	exact_match	none	307 should not be asked if 295 is 'Our deployment process does not currently prioritize resource efficiency.'
291	297	must_not_ask	Not Implemented	exact_match	none	291 should not be asked if 297 is Not Implemented
303	297	must_not_ask	Not Implemented	exact_match	none	303 should not be asked if 297 is Not Implemented
310	297	must_not_ask	Not Implemented	exact_match	none	310 should not be asked if 297 is Not Implemented
288	298	must_not_ask	Our deployment process does not currently prioritize minimizing downtime.	exact_match	none	288 should not be asked if 298 is 'Our deployment process does not currently prioritize minimizing downtime.'
309	298	must_not_ask	Our deployment process does not currently prioritize minimizing downtime.	exact_match	none	309 should not be asked if 298 is 'Our deployment process does not currently prioritize minimizing downtime.'
291	298	must_not_ask	Our deployment process does not currently prioritize minimizing downtime.	exact_match	none	291 should not be asked if 298 is 'Our deployment process does not currently prioritize minimizing downtime.'
303	300	must_not_ask	Not Implemented	exact_match	none	303 should not be asked if 300 is Not Implemented
304	300	must_not_ask	Not Implemented	exact_match	none	304 should not be asked if 300 is Not Implemented
290	302	must_not_ask	Our deployment process does not currently leverage containerization or serverless architecture for resource efficiency.	exact_match	none	290 should not be asked if 302 is 'Our deployment process does not currently leverage containerization...'
295	302	must_not_ask	Our deployment process does not currently leverage containerization or serverless architecture for resource efficiency.	exact_match	none	295 should not be asked if 302 is 'Our deployment process does not currently leverage containerization...'
307	302	must_not_ask	Our deployment process does not currently leverage containerization or serverless architecture for resource efficiency.	exact_match	none	307 should not be asked if 302 is 'Our deployment process does not currently leverage containerization...'
291	303	must_not_ask	Not Implemented	exact_match	none	291 should not be asked if 303 is Not Implemented
297	303	must_not_ask	Not Implemented	exact_match	none	297 should not be asked if 303 is Not Implemented
300	304	must_not_ask	Not Implemented	exact_match	none	300 should not be asked if 304 is Not Implemented
288	309	must_not_ask	Our deployment process does not currently prioritize minimizing downtime for user experience during updates.	exact_match	none	288 should not be asked if 309 is 'Our deployment process does not currently prioritize minimizing downtime for user experience...'
298	309	must_not_ask	Our deployment process does not currently prioritize minimizing downtime for user experience during updates.	exact_match	none	298 should not be asked if 309 is 'Our deployment process does not currently prioritize minimizing downtime for user experience...'
291	310	must_not_ask	Not Implemented	exact_match	none	291 should not be asked if 310 is Not Implemented
297	310	must_not_ask	Not Implemented	exact_match	none	297 should not be asked if 310 is Not Implemented
303	310	must_not_ask	Not Implemented	exact_match	none	303 should not be asked if 310 is Not Implemented
292	311	must_not_ask	Not Implemented	exact_match	none	292 should not be asked if 311 is Not Implemented
299	311	must_not_ask	Not Implemented	exact_match	none	299 should not be asked if 311 is Not Implemented
301	311	must_not_ask	Not Implemented	exact_match	none	301 should not be asked if 311 is Not Implemented
298	288	must_ask_first		none	none	298 and 288 should not co-occur (if both selected, 298 is targeted for removal by this rule instance)
288	298	must_ask_first		none	none	288 and 298 should not co-occur (if both selected, 288 is targeted for removal by this rule instance)
309	288	must_ask_first		none	none	309 and 288 should not co-occur (if both selected, 309 is targeted for removal by this rule instance)
288	309	must_ask_first		none	none	288 and 309 should not co-occur (if both selected, 288 is targeted for removal by this rule instance)
310	288	must_ask_first		none	none	310 and 288 should not co-occur (if both selected, 310 is targeted for removal by this rule instance)
288	310	must_ask_first		none	none	288 and 310 should not co-occur (if both selected, 288 is targeted for removal by this rule instance)
293	289	must_ask_first		none	none	293 and 289 should not co-occur (if both selected, 293 is targeted for removal by this rule instance)
289	293	must_ask_first		none	none	289 and 293 should not co-occur (if both selected, 289 is targeted for removal by this rule instance)
295	290	must_ask_first		none	none	295 and 290 should not co-occur (if both selected, 295 is targeted for removal by this rule instance)
290	295	must_ask_first		none	none	290 and 295 should not co-occur (if both selected, 290 is targeted for removal by this rule instance)
307	290	must_ask_first		none	none	307 and 290 should not co-occur (if both selected, 307 is targeted for removal by this rule instance)
290	307	must_ask_first		none	none	290 and 307 should not co-occur (if both selected, 290 is targeted for removal by this rule instance)
297	291	must_ask_first		none	none	297 and 291 should not co-occur (if both selected, 297 is targeted for removal by this rule instance)
291	297	must_ask_first		none	none	291 and 297 should not co-occur (if both selected, 291 is targeted for removal by this rule instance)
303	291	must_ask_first		none	none	303 and 291 should not co-occur (if both selected, 303 is targeted for removal by this rule instance)
291	303	must_ask_first		none	none	291 and 303 should not co-occur (if both selected, 291 is targeted for removal by this rule instance)
310	291	must_ask_first		none	none	310 and 291 should not co-occur (if both selected, 310 is targeted for removal by this rule instance)
291	310	must_ask_first		none	none	291 and 310 should not co-occur (if both selected, 291 is targeted for removal by this rule instance)
299	292	must_ask_first		none	none	299 and 292 should not co-occur (if both selected, 299 is targeted for removal by this rule instance)
292	299	must_ask_first		none	none	292 and 299 should not co-occur (if both selected, 292 is targeted for removal by this rule instance)
301	292	must_ask_first		none	none	301 and 292 should not co-occur (if both selected, 301 is targeted for removal by this rule instance)
292	301	must_ask_first		none	none	292 and 301 should not co-occur (if both selected, 292 is targeted for removal by this rule instance)
311	292	must_ask_first		none	none	311 and 292 should not co-occur (if both selected, 311 is targeted for removal by this rule instance)
292	311	must_ask_first		none	none	292 and 311 should not co-occur (if both selected, 292 is targeted for removal by this rule instance)
305	293	must_ask_first		none	none	305 and 293 should not co-occur (if both selected, 305 is targeted for removal by this rule instance)
293	305	must_ask_first		none	none	293 and 305 should not co-occur (if both selected, 293 is targeted for removal by this rule instance)
302	295	must_ask_first		none	none	302 and 295 should not co-occur (if both selected, 302 is targeted for removal by this rule instance)
295	302	must_ask_first		none	none	295 and 302 should not co-occur (if both selected, 295 is targeted for removal by this rule instance)
307	295	must_ask_first		none	none	307 and 295 should not co-occur (if both selected, 307 is targeted for removal by this rule instance)
295	307	must_ask_first		none	none	295 and 307 should not co-occur (if both selected, 295 is targeted for removal by this rule instance)
303	297	must_ask_first		none	none	303 and 297 should not co-occur (if both selected, 303 is targeted for removal by this rule instance)
297	303	must_ask_first		none	none	297 and 303 should not co-occur (if both selected, 297 is targeted for removal by this rule instance)
310	297	must_ask_first		none	none	310 and 297 should not co-occur (if both selected, 310 is targeted for removal by this rule instance)
297	310	must_ask_first		none	none	297 and 310 should not co-occur (if both selected, 297 is targeted for removal by this rule instance)
309	298	must_ask_first		none	none	309 and 298 should not co-occur (if both selected, 309 is targeted for removal by this rule instance)
298	309	must_ask_first		none	none	298 and 309 should not co-occur (if both selected, 298 is targeted for removal by this rule instance)
291	298	must_ask_first		none	none	291 and 298 should not co-occur (if both selected, 291 is targeted for removal by this rule instance)
298	291	must_ask_first		none	none	298 and 291 should not co-occur (if both selected, 298 is targeted for removal by this rule instance)
303	300	must_ask_first		none	none	303 and 300 should not co-occur (if both selected, 303 is targeted for removal by this rule instance)
300	303	must_ask_first		none	none	300 and 303 should not co-occur (if both selected, 300 is targeted for removal by this rule instance)
304	300	must_ask_first		none	none	304 and 300 should not co-occur (if both selected, 304 is targeted for removal by this rule instance)
300	304	must_ask_first		none	none	300 and 304 should not co-occur (if both selected, 300 is targeted for removal by this rule instance)
307	302	must_ask_first		none	none	307 and 302 should not co-occur (if both selected, 307 is targeted for removal by this rule instance)
302	307	must_ask_first		none	none	302 and 307 should not co-occur (if both selected, 302 is targeted for removal by this rule instance)
310	303	must_ask_first		none	none	310 and 303 should not co-occur (if both selected, 310 is targeted for removal by this rule instance)
303	310	must_ask_first		none	none	303 and 310 should not co-occur (if both selected, 303 is targeted for removal by this rule instance)
300	304	must_ask_first		none	none	300 and 304 should not co-occur (if both selected, 300 is targeted for removal by this rule instance)
304	300	must_ask_first		none	none	304 and 300 should not co-occur (if both selected, 304 is targeted for removal by this rule instance)
298	309	must_ask_first		none	none	298 and 309 should not co-occur (if both selected, 298 is targeted for removal by this rule instance)
309	298	must_ask_first		none	none	309 and 298 should not co-occur (if both selected, 309 is targeted for removal by this rule instance)
291	310	must_ask_first		none	none	291 and 310 should not co-occur (if both selected, 291 is targeted for removal by this rule instance)
310	291	must_ask_first		none	none	310 and 291 should not co-occur (if both selected, 310 is targeted for removal by this rule instance)
297	310	must_ask_first		none	none	297 and 310 should not co-occur (if both selected, 297 is targeted for removal by this rule instance)
310	297	must_ask_first		none	none	310 and 297 should not co-occur (if both selected, 310 is targeted for removal by this rule instance)
303	310	must_ask_first		none	none	303 and 310 should not co-occur (if both selected, 303 is targeted for removal by this rule instance)
310	303	must_ask_first		none	none	310 and 303 should not co-occur (if both selected, 310 is targeted for removal by this rule instance)
292	311	must_ask_first		none	none	292 and 311 should not co-occur (if both selected, 292 is targeted for removal by this rule instance)
311	292	must_ask_first		none	none	311 and 292 should not co-occur (if both selected, 311 is targeted for removal by this rule instance)
299	311	must_ask_first		none	none	299 and 311 should not co-occur (if both selected, 299 is targeted for removal by this rule instance)
311	299	must_ask_first		none	none	311 and 299 should not co-occur (if both selected, 311 is targeted for removal by this rule instance)
301	311	must_ask_first		none	none	301 and 311 should not co-occur (if both selected, 301 is targeted for removal by this rule instance)
311	301	must_ask_first		none	none	311 and 301 should not co-occur (if both selected, 311 is targeted for removal by this rule instance)
"""
    # Strip leading/trailing whitespace from the whole block, especially the first line.
    cleaned_plaintext_rules = plaintext_rules.strip()

    final_rules_df = clean_and_consolidate_rules(cleaned_plaintext_rules)

    if not final_rules_df.empty:
        output_filename = "cleaned_survey_rules.csv"
        # Use tab separator for consistency with Supabase if that's what it expects from CSV
        # Otherwise, common CSV uses comma. Let's use comma for standard CSV.
        final_rules_df.to_csv(output_filename, sep=',', index=False, encoding='utf-8')
        print(f"\nSuccessfully generated cleaned rules CSV: {output_filename}")
        print(f"\nFirst 5 rows of the output:\n{final_rules_df.head().to_string()}")
    else:
        print("\nNo rules were processed or an error occurred.")