from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler      



df = pd.read_csv('WineDataset.csv')

# Separates the class variable in y
X = df[["Alcohol",'Ash','Malic_Acid','Ash_Alcanity','Total_Phenols','Flavanoids','OD280','Color_Intensity','Proanthocyanins','Nonflavanoid_Phenols','Hue']].copy()

figure = plt.figure(figsize=(8,5))#set kích cở hình





# Removes the y column from df
df = pd.DataFrame(
    StandardScaler().fit_transform(df),
    columns=df.columns)
print(df)

