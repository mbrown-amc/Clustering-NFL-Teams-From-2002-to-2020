import os
import pandas as pd

def get_data():
    """
    Loads the data for the project.
    
    """
    
    import os
    import pandas as pd
    
    pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    datadir = pardir + "\\Data"
    datadir
    
    offense = []
    defense = []

    offensel = []
    defensel = []
    
    for f in os.listdir(datadir):
        if f[11] == "O":
            offense.append(datadir + "\\" + f)
        if f[11] == "D":
            defense.append(datadir + "\\" + f)
        
    for f in offense:
        offensel.append(pd.read_excel(f,0))
    for f in defense:
        defensel.append(pd.read_excel(f,0))
    
    return offensel, defensel
        
def clean_data(offense, defense):
    """
    Prepares the data for the project.
    :param offense: DataFrame. The offense team data.
    :param defense: DataFrame. The defense team data.
    """
    import pandas as pd
    i = 2002
    
    for f in offense:
        f.insert(0, "Year", i)
        i+= 1
    
    j = 2002

    for g in defense:
        g.insert(0, "Year", j)
        j+= 1
    for i in range(0, len(offense)):
        offense[i] = offense[i].drop([32,33,34])
        
    for i in range(0, len(defense)):
        defense[i] = defense[i].drop([32,33,34])
        
    combined = []
    i = 0
    for f in offense:
        combined.append(f.merge(defense[i], how='inner', on=["Year","Tm"]))
        i+=1
    finalframe = pd.concat(combined, ignore_index=True)
    finalframe1 = finalframe.drop(["Year", "Tm", "G_x", "G_y", "Rk_x", "Rk_y"], axis = 1)
    return finalframe, finalframe1
   
import matplotlib.pyplot as plt
import numpy as np
    
def find_clusters(ClusterTeams):
    """
    Finds the optimal number of clusters using KMeans.
    :param ClusterTeams: DataFrame. The data to find the number of clusters of.
    """
    
    from yellowbrick.cluster import KElbowVisualizer
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(random_state = 52594)

    visualizer = KElbowVisualizer(kmeans, k=(2,10), metric = 'calinski_harabasz', timings = False)

    visualizer.fit(ClusterTeams)
    visualizer.show()
    
    kmeans = KMeans(random_state = 52594)

    visualizer = KElbowVisualizer(kmeans, k=(2,10), metric = 'silhouette', timings = False)

    visualizer.fit(ClusterTeams)
    visualizer.show()
    
def scale_data(data):
    """
    Scales the data using StandardScaler
    :param data: DataFrame like object. The data to be scaled.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns = data.columns)
    return scaled_data
    
def cluster_data(data, numclusters = 3):
    """
    Clusters the data using KMeans.Returns the cluster predictions on the data.
    :param data: DataFrame like object. The data to perform clustering on.
    :param numclusters: Int. The number of clusters to make for clustering.

    """
    
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters = numclusters, random_state = 52594)

    kmeans.fit(data)

    clusters = kmeans.predict(data)
    
    return clusters
    
    
def add_clusters(data, clusters):
    """
    Adds the cluster predictions to the original data for interpretation.
    :param data: DataFrame. The data to have the cluster predictions added on to.
    :param clusters: List. The list of cluster predictions to be added to the DataFrame.
    """
    addclusters = data
    addclusters["cluster"] = clusters
    return addclusters

def pca_exp_var(data):
    """
    Charts the explained variance per component from PCA.
    :param data: DataFrame. The data for PCA to show the explained variance per component of.
    """
    from sklearn.decomposition import PCA
    pca = PCA(random_state = 52594).fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('num comp')
    plt.ylabel('cumul expl var');
    
def pca(data, numcomp = .99):
    """
    Performs PCA on the given data.
    :param data: DataFrame. The data to perform PCA on.
    :Param numcomp: Variable. As an int, the number of components to use when performing PCA. As a 2 decimal float < 1, the percentage of explained variance required when PCA is complete.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=numcomp, random_state = 52594)

    pca.fit(data)

    print("the explained variance ratio is ", pca.explained_variance_ratio_.sum())
    
    reduct = pca.transform(data)

    print("The shape of the original data is ", data.shape)
    print("The shape after pca is ", reduct.shape)
    return reduct

def plot(cluster0, cluster1, cluster2, x, y, xlabel, ylabel, l1, l2, l3):
    """
    Plots data.
    :param cluster0: DataFrame. The data belonging to the first cluster.
    :param cluster1: DataFrame. The data belonging to the second cluster.
    :param cluster2: DataFrame. The data belonging to the third cluster.
    :param x: variable. The x column to be plotted.
    :param y: variable. The y column to be plotted.
    :param xlabel: String. The x label for the plot.
    :param ylabel: String. The y label for the plot.
    :param l1: String. The label for the first cluster.
    :param l2: String. The label for the second cluster.
    :param l3: String. The label for the third cluster.
    """
    figure = plt.figure()
    plot = figure.add_subplot(111)
    
    plt.scatter(cluster0[x], cluster0[ y], s=10, c='r', cmap = "rainbow", marker = "s", label = l1)
    plt.scatter(cluster1[x], cluster1[ y], s=10, c='b', cmap = "rainbow", marker = "s", label = l2)
    plt.scatter(cluster2[x], cluster2[ y], s=10, c='g', cmap = "rainbow", marker = "s", label = l3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
def break_clusters(data):
    """
    Breaks the data into clusters for use in graphing.
    :param data: DataFrame. The data to be broken into clusters.
    """
    import pandas as pd

    c0 = data.loc[data["cluster"] == 0]
    c1 = data.loc[data["cluster"] == 1]
    c2 = data.loc[data["cluster"] == 2]   
    return c0, c1, c2