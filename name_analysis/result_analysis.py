"""
This script is used to generate a bar chart comparing the 
distribution of categories in the survey data and the practical 
data.
"""

import json
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    with open("name_analysis/results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    matplotlib.rcParams.update({'font.size': 12})

    survey_data = {
        "A": 69/108 * 100, 
        "S": 62/108 * 100,
        "T": 57/108 * 100, 
        "V": 52/108 * 100, 
        "P": 46/108 * 100,
        "L": 39/108 * 100, 
        "D": 30/108 * 100, 
        "F": 16/108 * 100, 
        "R": 8/108 * 100,
        "C": 7/108 * 100, 
        "O": 3/108 * 100, 
        "Y": 2/108 * 100
    }

    # Count the number of each category in practical data and convert to percentages
    categories_cnt = {}
    for model_name, category_set in results.items():
        for category in category_set:
            if category == 'N':
                category = 'Y'
            categories_cnt[category] = categories_cnt.get(category, 0) + 1
    total_practical = len(results)
    practical_percentages = {k: (v / total_practical) * 100 for k, v in categories_cnt.items()}

    # Create lists for plotting
    categories = list(survey_data.keys())  # Ensure consistent order of categories
    survey_percentages = [survey_data.get(cat, 0) for cat in categories]
    practical_data_percentages = [practical_percentages.get(cat, 0) for cat in categories]

    # Plotting
    x = range(len(categories))  # Category positions
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x, survey_percentages, width, label='Survey Data')
    bars2 = ax.bar(
        [p + width for p in x], practical_data_percentages, width, label='Practical Data'
    )

    # Adding text for labels, title, and axes ticks
    ax.set_ylabel('Frequency (%)')
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(categories)
    ax.legend()

    # Optional: Display percentages above bars
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        rotation=90)
    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.5, color='grey', alpha=0.7)
    # Remove the border (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()  # Adjust layout
    plt.savefig("name_analysis/survey_vs_practical.png")
