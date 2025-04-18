import json

with open("data_cleaned.json") as f:
    data = json.load(f)


# Extract data and labels
task = [d['task'] for d in data.values()]
model_type = [d['model_type'] for d in data.values()]
architecture = [d['arch'] for d in data.values()]

# Initialize dictionaries to hold counts
unique_tasks_counts = {}
unique_model_types_counts = {}
unique_architectures_counts = {}

# Count the number of occurrences for each unique task
for t in task:
    unique_tasks_counts[t] = unique_tasks_counts.get(t, 0) + 1

# Count the number of occurrences for each unique model type
for mt in model_type:
    unique_model_types_counts[mt] = unique_model_types_counts.get(mt, 0) + 1

# Count the number of occurrences for each unique architecture
for arch in architecture:
    unique_architectures_counts[arch] = unique_architectures_counts.get(arch, 0) + 1

# Print out the unique counts and their respective numbers
print(f"Unique tasks and their counts: {unique_tasks_counts}")
print(f"Unique model types and their counts: {unique_model_types_counts}")
print(f"Unique architectures and their counts: {unique_architectures_counts}")

print(f"Number of unique tasks: {len(unique_tasks_counts)}")
print(f"Number of unique model types: {len(unique_model_types_counts)}")
print(f"Number of unique architectures: {len(unique_architectures_counts)}")


# Use the already calculated counts instead of recalculating
task_counts = unique_tasks_counts
model_type_counts = unique_model_types_counts
architecture_counts = unique_architectures_counts

print(f"Task counts: {task_counts}")
print(f"Model type counts: {model_type_counts}")
print(f"Architecture counts: {architecture_counts}")

# print total number of models
print(f"Total number of models: {len(data)}")

# Update model types and architectures in 'data' based on counts being less than 20
for model in data:
    if model_type_counts[data[model]['model_type']] < 20:
        data[model]['model_type'] = 'other'
        # Update the counts since 'other' category is increased
        model_type_counts['other'] = model_type_counts.get('other', 0) + 1

for model in data:
    if architecture_counts[data[model]['arch']] < 20:
        data[model]['arch'] = 'other'
        # Update the counts since 'other' category is increased
        architecture_counts['other'] = architecture_counts.get('other', 0) + 1

# After updating data, recalculate the unique counts to reflect the changes
updated_unique_model_types = set(data[model]['model_type'] for model in data)
updated_unique_architectures = set(data[model]['arch'] for model in data)

print(f"Filtered number of unique model types: {len(updated_unique_model_types)}")
print(f"Filtered number of unique architectures: {len(updated_unique_architectures)}")

# Save the updated data to a file
with open("data_cleaned_filtered.json", "w") as f:
    json.dump(data, f)
