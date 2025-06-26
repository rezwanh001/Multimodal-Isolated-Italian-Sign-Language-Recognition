import pandas as pd
import os
from collections import Counter

# Define the input file paths and their corresponding accuracies
input_files = {
    './outputs/submissions/submission_valtest_(0.94240).csv': 0.94240,
    './outputs/submissions/submission_valtest_(0.98662).csv': 0.98662,
    './outputs/submissions/submission_valtest_(0.98752).csv': 0.98752,
    './outputs/submissions/submission_valtest_(0.98956).csv': 0.98956,
    './outputs/submissions/submission_valtest_(0.99274).csv': 0.99274,
    #....
}

output_file = './outputs/submissions/final_submission.csv'

# Create a list to store DataFrames from each submission
dfs = []

# Load each submission file and store it with its accuracy
print("Loading submission files...")
for i, (file_path, accuracy) in enumerate(input_files.items()):
    try:
        # Read 'id' as string to avoid potential float conversion if 'id' values are numeric
        # Read 'Pred' as object/string first to ensure no implicit float conversion,
        # then convert to integer where possible later if necessary for tie-breaking.
        df = pd.read_csv(file_path, dtype={'id': str, 'Pred': object})
        # Rename the 'Pred' column to distinguish it from other submissions
        df.rename(columns={'Pred': f'Pred_{i}'}, inplace=True)
        df[f'Accuracy_{i}'] = accuracy # Store accuracy for tie-breaking
        dfs.append(df)
        print(f"Loaded {file_path} with accuracy {accuracy}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure the paths are correct.")
        exit()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        exit()

if not dfs:
    print("No valid submission files loaded. Exiting.")
    exit()

# Merge all Dataframes on the 'id' column
# Initialize merged_df with the first DataFrame
merged_df = dfs[0].drop(columns=[f'Accuracy_{0}']) # Drop accuracy column for the initial merge

for i, df in enumerate(dfs[1:]):
    # Perform an outer merge to ensure all IDs are included, even if some files miss an ID
    # This assumes 'id' is consistent across files.
    merged_df = pd.merge(merged_df, df, on='id', how='outer')

print("All submission files merged.")

# Define the function to apply majority voting with accuracy-based tie-breaking
def get_final_prediction(row, accuracies):
    # Extract all prediction columns for the current row
    predictions = {col: row[col] for col in row.index if col.startswith('Pred_')}

    # Filter out NaN predictions that might occur from outer merge if IDs are not perfectly aligned
    valid_predictions = {k: v for k, v in predictions.items() if pd.notna(v)}

    if not valid_predictions:
        return None # No valid predictions for this ID

    # Create a list of (prediction, model_index) tuples
    pred_tuples = []
    for model_col, pred_val in valid_predictions.items():
        model_index = int(model_col.split('_')[1])
        # Attempt to convert prediction to int here for consistent comparison,
        # but keep original for `Counter` if conversion fails (e.g., string labels)
        try:
            pred_tuples.append((int(pred_val), model_index))
        except (ValueError, TypeError):
            pred_tuples.append((pred_val, model_index))


    # Count votes for each unique prediction
    vote_counts = Counter([p[0] for p in pred_tuples])

    # Find the maximum number of votes
    max_votes = 0
    if vote_counts:
        max_votes = max(vote_counts.values())

    # Identify classes that achieved the maximum number of votes (potential ties)
    tied_classes = [cls for cls, count in vote_counts.items() if count == max_votes]

    final_pred_value = None

    if len(tied_classes) == 1:
        # No tie, return the majority class
        final_pred_value = tied_classes[0]
    else:
        # Handle ties: pick the class that was predicted by the highest accuracy model among the tied ones
        best_tied_class = None
        highest_tie_accuracy = -1.0

        for tied_class in tied_classes:
            # Find the maximum accuracy among models that predicted this 'tied_class'
            current_class_max_accuracy = -1.0
            for pred_val_from_tuple, model_idx in pred_tuples:
                if pred_val_from_tuple == tied_class: # Compare the already-converted-if-possible value
                    model_accuracy_col = f'Accuracy_{model_idx}'
                    if model_accuracy_col in row.index and pd.notna(row[model_accuracy_col]):
                        current_class_max_accuracy = max(current_class_max_accuracy, row[model_accuracy_col])
            
            # If this tied class has a higher 'best model accuracy' than previous ties, choose it
            if current_class_max_accuracy > highest_tie_accuracy:
                highest_tie_accuracy = current_class_max_accuracy
                best_tied_class = tied_class
            # If accuracies are equal, prefer the numerically smallest class for deterministic tie-breaking
            elif current_class_max_accuracy == highest_tie_accuracy and best_tied_class is not None:
                # Assuming 'Pred' values are numeric, if not, adjust logic (e.g., sort strings)
                try:
                    # Compare as floats to handle cases where they might be float due to NaN initially
                    if float(tied_class) < float(best_tied_class):
                        best_tied_class = tied_class
                except ValueError:
                    # Handle non-numeric class labels, e.g., by alphabetical order
                    if str(tied_class) < str(best_tied_class):
                        best_tied_class = tied_class

        final_pred_value = best_tied_class
    
    # Ensure the final prediction is an integer if it's not None
    if final_pred_value is not None:
        try:
            return int(final_pred_value)
        except (ValueError, TypeError):
            # If for some reason it cannot be converted to int (e.g., original labels were strings like 'cat'),
            # return as is. This part is primarily for numeric class labels.
            return final_pred_value
    return None


print("Applying ensemble logic...")
# Create a dictionary of model accuracies keyed by their column name suffix (e.g., '0', '1')
# This dictionary is used within the get_final_prediction function.
model_accuracies = {
    int(k.split('_')[1]): v
    for k, v in zip(
        [f'Pred_{i}' for i in range(len(dfs))],
        [input_files[list(input_files.keys())[i]] for i in range(len(dfs))]
    )
}

# Ensure accuracy columns are present in merged_df for the apply function
for i, (file_path, acc) in enumerate(input_files.items()):
    merged_df[f'Accuracy_{i}'] = acc

merged_df['Pred'] = merged_df.apply(lambda row: get_final_prediction(row, model_accuracies), axis=1)

print("Ensemble predictions generated.")

# Prepare the final submission DataFrame
# Convert the 'Pred' column to integer type, handling potential NaNs
final_submission_df = merged_df[['id', 'Pred']].copy()

# Convert to Int64 to allow for NaN values in an integer column, then fill NaNs if desired.
# Or use .astype(int) after filling NaNs if all predictions are guaranteed.
# If there are actual None values from get_final_prediction for IDs without valid predictions,
# you might want to fill them or handle them based on competition rules.
# For simplicity, we'll try to convert to 'Int64' (Pandas nullable integer type)
# if all values are convertible, otherwise we'll leave it as is if it becomes object.
try:
    final_submission_df['Pred'] = final_submission_df['Pred'].astype('Int64')
except Exception as e:
    print(f"Warning: Could not convert 'Pred' column to nullable integer type. It might contain non-integer values or NaNs that prevent this. Error: {e}")


# Save the final submission to a CSV file
try:
    final_submission_df.to_csv(output_file, index=False)
    print(f"Final submission saved to {output_file}")
except Exception as e:
    print(f"Error saving final submission to {output_file}: {e}")

