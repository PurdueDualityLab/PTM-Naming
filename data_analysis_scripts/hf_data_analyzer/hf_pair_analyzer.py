from huggingface_hub import HfApi
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import time

with open('/depot/davisjam/data/chingwo/PTM-Naming/ModelTypes.json') as json_file:
    data_dict = json.load(json_file)

data_set = set()
for k, v in data_dict.items():
    for n in v:
        data_set.add(n)

print(len(data_set))

api = HfApi()

limiter = 0

fw_set = set()

fw_perc_dict = dict()
ptm_picked = set()
pure_pytorch = 0
total_fw_used = 0
pure_pytorch_99 = 0
pure_pytorch_95 = 0
pure_pytorch_90 = 0
pure_pytorch_85 = 0
total_list = []

with open('/depot/davisjam/data/chingwo/PTM-Naming/LibraryToFrameworkMapping.json') as json_file:
    fw_dict = json.load(json_file)

with open('/depot/davisjam/data/chingwo/PTM-Naming/downloaded.json', 'r') as f:
    downloaded_data = json.load(f)

with open('/depot/davisjam/data/chingwo/PTM-Naming/searched.json', 'r') as f:
    searched = json.load(f)

for k, fws in fw_dict.items():
    for fw in fws:
        fw_set.add(fw)

ext_perc_dict = dict()

m_dn_cnt = 0

waittime = 0.5
batch_cnt = 500
for to_search in data_set:
    
    if limiter == 15: break
    limiter += 1

    if to_search in searched:
        limiter -= 1
        print('Skipped', to_search)
        continue

    print('Searching', to_search)

    time.sleep(1)
    models = api.list_models(search=to_search)
    tags = api.get_model_tags()

    lib_set = set()

    for option, lib in tags['library'].items():
        lib_set.add(lib)

    fw_cnt_dict_curr = dict()
    for fw in fw_set:
        fw_cnt_dict_curr[fw] = 0

    curr_model = set()
    for model in models:
        if model.downloads >= 30:
            curr_model.add(model)
            ptm_picked.add(model.id)

    if len(curr_model) == 0:
        limiter -= 1
        print('Empty search, skipped')
        continue

    print(limiter, len(curr_model))

    searched[to_search] = [model.id for model in curr_model]
    with open('/depot/davisjam/data/chingwo/PTM-Naming/searched.json', 'w') as f:
        json.dump(searched, f)

    mul_sup_lib_cnt_dict = {'spacy': 0, "transformers": 0, "adapter-transformers": 0}

    ext_dict = dict()
    for useful_model in curr_model:
        if useful_model.modelId in downloaded_data: 
            for e in downloaded_data[useful_model.modelId]:
                if e not in ext_dict:
                    ext_dict[e] = 0
                ext_dict[e] += 1
            continue

        url = "https://huggingface.co/" + str(useful_model.modelId) + "/tree/main"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a")
        ext_set = set()
        for link in links:
            href = link.get("href")
            if "/blob/main" in href:
                result = urlparse(href)
                _, ext = os.path.splitext(result.path)
                if len(ext) > 0: ext_set.add(ext)
        for e in ext_set:
            if e not in ext_dict:
                ext_dict[e] = 0
            ext_dict[e] += 1

        print(useful_model.modelId, ext_set)
        downloaded_data[useful_model.modelId] = list(ext_set)
        print(len(downloaded_data.keys()))
        with open('/depot/davisjam/data/chingwo/PTM-Naming/downloaded.json', 'w') as f:
            json.dump(downloaded_data, f)
        time.sleep(waittime)

        m_dn_cnt += 1
        if m_dn_cnt == batch_cnt:
            m_dn_cnt = 0
            i = input("Type anything to continue: \n")
            if i == "settime":
                waittime = int(input("Set time wait: \n"))
            if i == "setbcnt":
                batch_cnt = int(input("Set batch cnt: \n"))






