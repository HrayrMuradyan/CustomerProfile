import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


class K_Means:
    """ 
    Does clustering on the given dataset.

    This class does K-Means clustering on the given dataset and finds the optimal number of clusters based on performance metrics.

    Parameters
    ----------
    kmeans : None 
        Represents the kmeans class from sklearn that needs to be fitted.

    fitted : bool
        Whether or not the model was fitted.      

    """
    def __init__(self):
        """
        Initializes the K_Means class instance.
        """
        self.kmeans = None
        self.fitted = False
        
    def fit(self, data, n_clusters):
        """
        Fits the KMeans algorithm on the given dataset.

        Parameters
        ----------
        data : pandas dataframe
            Represents the dataset that should be fitted.
            
        n_clusters : int
            Number of clusters to specify for KMeans.

        """
        self.kmeans = KMeans(n_clusters = n_clusters, random_state = 42)
        self.kmeans.fit(data)
        self.fitted = True
        
    def predict(self, data):
        """
        Predicts the cluster of the given datapoint after fitting the model.

        Parameters
        ----------
        data : pandas dataframe
            The dataframe to be predicted.
            

        Returns
        -------
        Returns the prediction.
        """
        if self.kmeans is None:
            raise NotImplementedError('Fit the model first!')
        
        prediction = self.kmeans.predict(data)
        return prediction
    
    def best_k_elbow(self, data, max_k=10):
        """
        Finds the optimal number of clusters for KMeans using elbow method.

        Parameters
        ----------
        data : pandas dataframe
            The dataset for fitting.
            
        max_k : int
             (Default value = 10)
             Represents the maximum number of clusters to consider during searching.

        Returns
        -------
        The optimal number of clusters.
        """
        inertia_list = []

        #For each k run the kmeans and append the inertia to the list
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters = k, random_state=1)
            kmeans.fit(data)
            inertia_list.append(kmeans.inertia_)

        #Calculate the difference between X(i+1) - X(i)
        differences = [inertia_list[i-1] - inertia_list[i] for i in range(1, len(inertia_list))]
        point1 = np.array([1, differences[0]])
        point2 = np.array([max_k, differences[-1]])

        #Calculate the distance between the intertia points to the straight line connecting the start and end points
        diff_distance = [np.linalg.norm(np.cross(point2-point1, point1-np.array([i, differences[i]])))/np.linalg.norm(point2-point1) for i in range(max_k-1)]

        plt.plot(range(1, max_k + 1), inertia_list)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.show()

        plt.plot(differences)
        plt.plot([0, max_k-2], [differences[0], differences[-1]], alpha = 0.5)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia differences')
        plt.title('Elbow Method')
        plt.show()

        optimal_k = np.argmax(diff_distance) + 1
        print(f'The optimal value for K is {optimal_k}')
        return optimal_k
        
    def best_k_silhouette(self, data, max_k=10):
        """
        Finds the optimal number of clusters for KMeans using silhouette score method.

        Parameters
        ----------
        data : pandas dataframe
            The dataset for fitting.
            
        max_k : int
             (Default value = 10)
             Represents the maximum number of clusters to consider during searching.

        Returns
        -------
        The optimal number of clusters.
        """
        best_k = 0
        best_score = -1
        scores = []

        #For each k calculate the silhouette score
        for k in range(2, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            scores.append(score)

            #If new score is better than the previous one, update the best
            if score > best_score:
                best_k = k
                best_score = score
        plt.plot(range(2, max_k + 1), scores)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette Method')
        plt.show()
        return best_k
    
    def find_optimal_k(self, data, method = 'silhouette', max_k=10):
        """
        Finds the optimal number of clusters using user-defined method.

        Parameters
        ----------
        data : pandas dataframe
            The dataset for fitting.
            
        method : str (Available values: 'elbow', 'silhouette')
             (Default value = 'silhouette')
        max_k : int
             (Default value = 10)
             Represents the maximum number of clusters to consider during searching.

        Returns
        -------
        The optimal number of clusters.
        """
        if method == 'silhouette':
            function = self.best_k_silhouette
        elif method == 'elbow':
            function = self.best_k_elbow
        else:
            print(f'No method named {method}. Going with the default "silhouette"...')
            function = self.best_k_silhouette
        best_k = function(data, max_k)
        return best_k

    def evaluate(self, data):
        """
        Evaluate the KMeans clustering algorithm using silhouette score.

        Parameters
        ----------
        data : pandas dataframe
            Represents the dataset to predict and evaluate. 

        Returns
        -------
        Returns the silhouette score.
        """
        labels = self.kmeans.predict(data)
        sil_score = silhouette_score(data, labels)
        return sil_score
