import numpy
import pandas as pd
from sklearn.cluster import KMeans

fn = r'C:\Users\DELL I5558\Desktop\Python\ELEC5222\kMeans\NSW-ER01.csv'
dataframe = pd.read_csv(fn)
dataset = dataframe.values
X = dataset[:, 0:23].astype(float)
seasons = dataset[:, 23]

model = KMeans(n_clusters=4)
labels = model.fit_predict(X)

dff = pd.DataFrame({'labels': labels, 'seasons': seasons})
ct = pd.crosstab(dff['labels'], dff['seasons'])
print(ct)

model_inertia = model.inertia_
print('Inertia without Standardization (4 Clusters): {}'.format(model_inertia))
