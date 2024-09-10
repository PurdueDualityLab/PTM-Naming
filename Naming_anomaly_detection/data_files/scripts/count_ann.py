import os


path = "/depot/davisjam/data/chingwo/PTM-v2/PTM-Naming/peatmoss_ann/rand_sample_2500"  # Ensure this path is correctly concatenated
# path = "/depot/davisjam/data/chingwo/PTM-v2/PTM-Naming/peatmoss_ann/ann"

json_file_count = 0
for root, dirs, files in os.walk(path):  # Corrected: Added colon
    for file in files:  # Corrected: Changed loop variable to 'file'
        if os.path.splitext(file)[1] == '.json':  # Corrected: Check the extension of 'file'
            json_file_count += 1  # Corrected: Increment count for each JSON file

print(json_file_count)  # Print the total count of JSON files
