import json


with open('filtered_models.json') as f:
    data = json.load(f)


model_to_architecture = {}
for architecture, models in data.items():
    for model in models:
        model_to_architecture[model] = architecture

with open('filtered_name_to_architecture.json', 'w') as f:
    json.dump(model_to_architecture, f, indent=4)
