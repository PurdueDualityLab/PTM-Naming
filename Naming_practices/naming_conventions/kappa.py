from sklearn.metrics import cohen_kappa_score
import pandas as pd


def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

df = pd.read_csv('data/manual_label_data.csv')
researcher_1 = df['Researcher_1'].tolist()
researcher_2 = df['Researcher_2'].tolist()
print(kappa(researcher_1, researcher_2))
