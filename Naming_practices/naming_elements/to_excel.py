"""
This script reads the results.json file and converts it to an excel
file with the categories as columns and the models as rows.
"""
import json
import pandas as pd
from name_analysis.get_gpt_response import CATEGORIES

if __name__ == "__main__":
    with open('name_analysis/results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"Length of results: {len(results)}")

    # The results are in the format {model_name: category_set}

    # create a df table with columns as the categories
    df = pd.DataFrame(columns=CATEGORIES)
    for model_full_name, category_set in results.items():
        # create a row with 1s and 0s
        row = [1 if category in category_set else 0 for category in CATEGORIES]
        df.loc[model_full_name] = row

    # save the dataframe to an excel file
    df.to_excel('name_analysis/results.xlsx')
