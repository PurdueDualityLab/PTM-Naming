import json
import matplotlib.pyplot as plt


with open('ModelNameArchTask.json', 'r') as f:
    data = json.load(f)


cnt_arch = {}
for model_dic in data:
    if model_dic['arch'] == "Unknown":
        continue
    if model_dic['arch'] in cnt_arch:
        cnt_arch[model_dic['arch']
                 ] += 1
    else:
        cnt_arch[model_dic['arch']] = 1

# print the total count
print(f"Total number of architectures: {len(cnt_arch)}")
print(f"Total number of models: {sum(cnt_arch.values())}")

cnt_thresh = 50
# Only plot the architectures with count > 50 and sort the list and save the arch with count < 50 to another
cnt_arch_over50 = {k: v for k, v in cnt_arch.items() if v >= cnt_thresh}
cnt_arch_low_freq = {k: v for k, v in cnt_arch.items() if v < cnt_thresh}

print(f"Total number of architectures with count >= {cnt_thresh}: {len(cnt_arch_over50)}")
print(f"Total number of architectures with count < {cnt_thresh}: {len(cnt_arch_low_freq)}")

cnt_arch = dict(sorted(cnt_arch_over50.items(), key=lambda item: item[1], reverse=True))


# save the dict to a json file
with open(f'arch_count_>={cnt_thresh}.json', 'w') as f:
    json.dump(cnt_arch, f)

with open(f'arch_count_<{cnt_thresh}.json', 'w') as f:
    json.dump(cnt_arch_low_freq, f)


print(f"Total number of architecture that have count > {cnt_thresh}: {len(cnt_arch)}")

# Use logscale for y-axis
plt.figure(figsize=(10, 6))
plt.yscale('log')
plt.bar(cnt_arch.keys(), cnt_arch.values())
plt.xlabel('Architecture')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=12)
plt.tight_layout()
plt.savefig('arch_count.png')