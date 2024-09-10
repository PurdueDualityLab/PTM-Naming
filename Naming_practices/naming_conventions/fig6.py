import matplotlib.pyplot as plt
import numpy as np

def plot(data, gpt, title):
    total = sum(data.values())
    percentage = [np.round(p/total*100, 1) for p in data.values()]
    
    if gpt:
        gpt_total = sum(gpt.values())
        gpt_percentage = [np.round(p/gpt_total*100, 1) for p in gpt.values()]

    N = len(data)
    ind = np.arange(0,N)
    width = 0.8
    if gpt:
        width = 0.4
    
    plt.figure(figsize=(8, 6))
    plt.box(False)
    plt.xticks(ind, data.keys(), fontsize=16)

    if gpt:
        plt.bar(ind-width/2, percentage, width=width, zorder=2, label="Survey Data")
        plt.bar(ind+width/2, gpt_percentage, width=width, zorder=2, label="Practical data")
        plt.legend(loc='best', fontsize=16)
        for i, (survey, value) in enumerate(zip(percentage, gpt_percentage)):
            plt.text(plt.xticks()[0][i]-width/2, survey+0.4, str(f'{survey}'), ha='center', va='bottom', fontsize=16, rotation=90)
            plt.text(plt.xticks()[0][i]+width/2, value+0.4, str(f'{value}'), ha='center', va='bottom', fontsize=16, rotation=90)
        max_value = max(max(gpt_percentage, percentage))
    else:
        plt.bar(ind, percentage, width=width, zorder=2)
        for i, value in enumerate(percentage):
            plt.text(plt.xticks()[0][i], value, str(f'{value}'), ha='center', va='bottom', fontsize=16)
        max_value = max(percentage)
    plt.ylabel("Frequency (%)", fontsize=20)
    plt.grid(which='major', axis='y', zorder=1)
    plt.tight_layout()
    plt.ylim(top=max_value*1.1)
    plt.savefig(f'plot/{title}.pdf')


if __name__ == "__main__":
    Q35 = {"A": 37, "S": 39, "D": 10, "V": 11, "L": 36, "T": 33, "F": 5, "R": 7, "Y": 12, "P": 30, "C": 13, "O": 2, "G": 25}
    Q23 = {"A": 69, "S": 62, "D": 30, "V": 52, "L": 39, "T": 56, "F": 16, "R": 8, "Y": 2, "P": 46, "C": 7, "O": 3, "G": 32}
    Q23_GPT = {"A": 69, "S": 62, "D": 30, "V": 52, "L": 39, "T": 56, "F": 16, "R": 8, "Y": 2, "P": 46, "C": 7, "O": 3, "G": 32}
    
    Q35_sorted = dict(sorted(Q35.items(), key=lambda item: item[1], reverse=True))
    Q23_sorted = dict(sorted(Q23.items(), key=lambda item: item[1], reverse=True))
    Q23_GPT_sorted = dict(sorted(Q23_GPT.items(), key=lambda item: item[1], reverse=True))
    
    plot(Q35_sorted, None, "Q35")
    plot(Q23_sorted, Q23_GPT_sorted, "Q23")
    