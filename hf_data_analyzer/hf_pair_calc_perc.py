
import json

with open('/depot/davisjam/data/chingwo/PTM-Naming/downloaded.json', 'r') as f:
    downloaded_data = json.load(f)

with open('/depot/davisjam/data/chingwo/PTM-Naming/searched.json', 'r') as f:
    searched = json.load(f)

full_pytorch_cnt = 0
nf_pytorch_cnt = 0
nn_pytorch_cnt = 0

single_cnt = 0

ext_perc_dict = dict()
for s_k, s_v in searched.items():
    s_len = len(s_v)
    ext_dict = dict()
    if len(s_v) < 2:
        single_cnt += 1
        continue
    for m_n in s_v:

        if len(downloaded_data[m_n]) == 0:
            s_len -= 1
            continue
        
        ddmn = downloaded_data[m_n]
        if '.pt' in ddmn or '.pth' in ddmn or '.bin' in ddmn or '.pkl' in ddmn:
            if 'pytorch' not in ext_dict: ext_dict['pytorch'] = 0
            ext_dict['pytorch'] += 1

        for ex in downloaded_data[m_n]:
            if ex not in ext_dict:
                ext_dict[ex] = 0
            ext_dict[ex] += 1
    

    if 'pytorch' in ext_dict:
        if ext_dict['pytorch']/s_len == 1.0:
            full_pytorch_cnt += 1
        if ext_dict['pytorch']/s_len > 0.99:
            nn_pytorch_cnt += 1
        if ext_dict['pytorch']/s_len > 0.95:
            nf_pytorch_cnt += 1

    for k, v in ext_dict.items():
        if k not in ext_perc_dict: ext_perc_dict[k] = []
        ext_perc_dict[k].append(v/s_len)

ext_avg_perc_dict = dict()
for k, v in ext_perc_dict.items():
    ext_avg_perc_dict[k] = sum(v) / (len(searched.keys()) - single_cnt)

for k, v in ext_avg_perc_dict.items():
    print(k, v)

print('Arch Family Count:', len(searched.keys()) - single_cnt) 
print('Model Survey Count:', len(downloaded_data.keys()) - single_cnt) 
print('Full Pytorch Count:', full_pytorch_cnt)
print('99% Pytorch Count:', nn_pytorch_cnt)
print('95% Pytorch Count:', nf_pytorch_cnt)

