import json

with open("arch_count_>=50.json", "r") as f:
    arch_count_50plus_list = json.load(f)


with open("arch_count_<50.json", "r") as f:
    arch_count_50minus_list = json.load(f)



with open('ModelNameArchTask.json', 'r') as f:
    data = json.load(f)

