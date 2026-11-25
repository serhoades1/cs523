import pandas as pd

csv_url = "https://www.kaggle.com/datasets/yasserh/titanic-dataset"

df = pd.read_csv(csv_url)

pivot = df.pivot_table(index='class', columns='age+group', values='survived', )


