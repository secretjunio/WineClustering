
# from multiprocessing.sharedctypes import Value
# from optparse import Values
# from statistics import median
# from matplotlib import projections
import numpy as np
import pandas as pd                               # For dataframes
import matplotlib.pyplot as plt                   # For plotting data
import seaborn as sns                             # For plotting data
from sklearn.cluster import KMeans, kmeans_plusplus                # For k-Means
from sklearn.model_selection import GridSearchCV  # For grid search
from sklearn.metrics import silhouette_score      # For metrics and scores
from sklearn.preprocessing import StandardScaler  # For standardizing data
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn import datasets
import scikit_posthocs as sp
#from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

df = pd.read_csv('WineDataset.csv')

# Separates the class variable in y


# Removes the y column from df
#df = df.drop('STT', axis=1)


#ve so do mo ta du lieu
# figure = plt.figure(figsize=(8,5))#set kích cở hình

# for i in df:#save mà không show sẽ bị sai hình
#     median_data=np.median(df[i])
#     plt.hist(df[i],bins='auto',
#             stacked=True,
#             edgecolor="#6A9662",
#             color="#DDFFDD")

#     plt.ylabel('Frequency')
#     plt.title(i)
#     plt.xlabel("Value")
#     plt.savefig('/Document/DataMeaning/Chart/'+i+'.png')
#     plt.show(block=False)
#     # plt.pause(3)#dừng lại để coi hình
#     plt.close()#đóng hình

# Creating plot
# boxplot = X.boxplot(column=["Alcohol",'Ash','Malic_Acid','Ash_Alcanity','Total_Phenols','Flavanoids','OD280','Color_Intensity','Proanthocyanins','Nonflavanoid_Phenols','Hue'])  
# # show plot
# plt.show()

# fig=plt.figure()
# x = df['Total_Phenols']
# y = df['Flavanoids']
# plt.title("Relationship between Total_Phenols and Flavanoids") 
# plt.xlabel("Total_Phenols") 
# plt.ylabel("Flavanoids") 
# plt.plot(x,y,'ro') 
# plt.show()


#  fig = plt.figure(figsize =(10, 10))
#  corrMatrix = X.corr()
#  sns.heatmap(corrMatrix, annot=True)
#  plt.show()



# Standardizes df
df = pd.DataFrame(
    StandardScaler().fit_transform(df),
    columns=df.columns)

#print(df.head)# in 5 dòng đầu



#transform multiple dimension to 2 dimension
X = df[["Alcohol", 'Magnesium','Ash','Malic_Acid','Ash_Alcanity','Total_Phenols','Flavanoids','OD280','Color_Intensity','Proanthocyanins','Nonflavanoid_Phenols','Proline','Hue']].copy()


mds = MDS(random_state=0)
X_transform = mds.fit_transform(X)

#print X
#print(X_transform)



km = KMeans(
    n_clusters=3,
    random_state=1,
    init='k-means++',
    n_init=10)



# Fits the model to the data
km.fit(X_transform)
# Displays the parameters of the fitted model
km.get_params()


fig = plt.figure()
ax = fig.add_subplot()
palete=['green','blue']
# Creates a scatter plot
sns.scatterplot(
    x=X_transform[:,0], 
    y= X_transform[:,1],
    data=X_transform, 
    style=km.labels_)

# Adds cluster centers to the same plot
plt.scatter(
    km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    marker='x',
    s=200,
    c='red')

plt.show()




# Sets up the custom scorer

def s2(estimator,X):
    return silhouette_score(X, estimator.fit_predict(X))

# List of values for the parameter `n_clusters`
param = range(2,10)

# KMeans object
km = KMeans(random_state=1, init='k-means++',n_init=10)

# Sets up GridSearchCV object and stores in grid variable
grid = GridSearchCV(
    km,
    {'n_clusters': param},
    scoring=s2,
    cv=2)

# arr=[]
# for i in range(2,10):
#     km1 = KMeans(
#     n_clusters=i,
#     random_state=1,
#     init='k-means++',
#     n_init=10)
#     labels_pred = km1.fit_predict(X_transform)
#     print('silhouette_score is {} for {} center '.format(silhouette_score(X_transform, labels_pred),i))
#     arr.append(silhouette_score(X_transform, labels_pred))
# Fits the grid object to data
grid.fit(X_transform)
# Accesses the optimum model
best_km = grid.best_estimator_

# Displays the optimum model
best_km.get_params()


######draw plot after optimum
# Plot mean_test_scores vs. n_clusters
plt.plot(
    param,
    grid.cv_results_['mean_test_score']
    #arr
    )

print("silhouette_score 2->9")
print(grid.cv_results_['mean_test_score'])
# Draw a vertical line, where the best model is
# print(best_km.labels_)
# maxSilhouette=max(arr)
# index=arr.index(maxSilhouette)+2
plt.axvline(
    x=best_km.n_clusters, 
    color='red',
    ls='--')

# Adds labels to the plot
plt.xlabel('Total Centers')
plt.ylabel('Silhouette Score')

plt.show()

print("N_center")
print(best_km.cluster_centers_.shape) #in ra để biết số tâm vaf feature
# Creates a scatter plot
sns.scatterplot(
    x=X_transform[:,0], 
    y= X_transform[:,1],
    data=X_transform, 
    style=best_km.labels_
    )

# Adds cluster centers to the same plot
plt.scatter(
    best_km.cluster_centers_[:, 0],
    best_km.cluster_centers_[:, 1],
    marker='x',
    s=200,
    c='red')

    
plt.show()


#execute after run
df['label'] = best_km.labels_ # để biết dòng dữ liệu nào ứng với nhãn nào, ở đây có 6 cụm 0 1 2 3 4 5

print(df.groupby(['label'])['label'].count()) # in ra số lượng các điểm trong từng cụm (ví dụ cụm 0 có 498 điểm)
print(df.groupby(['label'])['label'].count()/df['label'].count()*100) # ví dụ cụm 0 có 498 điểm, lấy 498/2000(tổng data) sau đó nhân cho 100 = 24.90

a = np.array(df.groupby(['label'])['label'].count()) 
b = np.array(df.groupby(['label'])['label'])
print(a)
print(b[:,0])

fig = plt.figure(figsize = (5, 5))
# creating the bar plot
plt.bar(b[:,0], a, color ='maroon',
        width = 0.1)
plt.xlabel("Số cụm")
plt.ylabel("Số lượng")
plt.title("Biểu đồ thể hiện số lượng dữ liệu của mỗi cụm")


plt.show()