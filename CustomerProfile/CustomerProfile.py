import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CustomerProfile:    
    """ 
    Creates Customer Profiles using clustering method provided.

    This class creates the customer segments and provides the report of the formed segments. Contains the basic tools for data preprocessing and issue warnings.

    Parameters
    ----------
    data : pandas dataframe 
        Represents the raw data to be fitted.

    target_column : str
        Represents the target column that the customer is interested in.       

    clustering : class
        Represents the clustering class that is used for clustering the segments.

    segmentation : str (available values: 'general', 'demographic', 'psychographic', 'behavioral', 'geographic', 'firmographic')
        Represents the type of segmentation that has to be done on the data.

    columns : str or list (either 'all' or the list of the columns to be used in segmentation)
        Represents the columns that should be used in the clustering.


    Attributes
    ----------
    preprocessed : bool
        Whether the data is preprocessed using .preprocess() method or not

    
    Notes
    ----------
    The data should not contain missing values or they are going to be automatically removed from the dataset. If the data contains many of them, handle them beforehand.
    """
    def __init__(self, data, target_column, clustering, segmentation = 'general', columns = 'all'):
        """
        Initializes a CustomerProfile instance with the given parameters.
        """
        #Instanciate the clustering class
        self.clustering = clustering()

        #If the segmentation type in valid values, then take it. Else take general
        self.segmentation = segmentation.lower() if segmentation.lower() in ['demographic', 'psychographic', 'behavioral', 'geographic', 'firmographic'] else 'general' 

        #Check if the provided columns attribute is str, list. 
        if columns.lower() == 'all':

            #if string and 'all', then take all the columns
            self.columns = list(data.columns)

        #Else if it is list and all elements are in data columns, then it is valid.
        elif isinstance(columns, list) and all(elem in data.columns for elem in columns):

            #Assign columns to the given columns
            self.columns = columns

        #Else value error    
        else:
            raise ValueError('columns parameter is assigned incorrectly. Provide "all" expression if you want to take all the columns or provide valid list of columns that exist in your data.')

        #Take the chosen columns from data 
        self.data = data[self.columns]

        #Check if target column in data
        self.target_column = target_column if target_column in self.data.columns else None

        #Yet not preprocessed
        self.preprocessed = False

        #Run greetings
        self.greeting()

        #Run warnings
        self.warnings()
        
    def greeting(self):
        """ 
        Greets the user by providing the information of the segmentation, which includes the clustering method used, the segmentation type, the columns used and the target column.
        """
        print('\033[1m>>>> Customer Segmentation instance created with the following parameters:\033[0m\n')
        print(f'- Clustering: {self.clustering.__class__.__name__}')
        print(f'- Segmentation type: {self.segmentation}')
        print(f'- Columns used: {self.columns}')
        print(f'- Target column: {self.target_column}\n')
        
    def warnings(self):
        """ 
        Warns the user if it managed to automatically find some issues connected with the segmentation, the data and/or the provided information.
        """
        warning_count=0
        print("\033[1m>>>> Warnings <<<<\033[0m")

        #If there are more than 10 features, warn. 
        if len(self.columns) > 10:
            warning_count+=1
            print("- Warning: high number of dimensions detected (>10), it is recommended to have less than 10 features. If you think all the features used are highly informative, continue confidently.")
        
        #If there are too many categorical features, warn.
        if len(self.data.select_dtypes(include=['object']).columns)/len(self.columns) > 0.5:
            warning_count+=1
            print("- Warning: more than 50% of the features are categorical features.")
        
        #If target column is not in data, raise an error.
        if self.target_column is None:
            raise ValueError("Data issue: the target column is specified incorrectly. No such column in the data.")

        #If target column is not numeric, raise an error
        if self.data[self.target_column].dtype not in ['float', 'int']:
            raise ValueError("Data issue: the target column is not numeric. Provide a numeric column.")
        
        #If no warnings caught, print no warnings.
        if warning_count==0:
            print('No warnings...')
            
        
    def preprocess(self, scaler, outlier):
        """
        Preprocesses the data by the following four ways:

        - Dropping the missing values if any.
        - One-hot encoding the categorical variables.
        - Handling outliers using the provided class.
        - Transforming the features into the same scale.

        Parameters
        ----------
        scaler : class
            Represents the scaler class to be used for transforming the features.
            
        outlier : class
            Represents the outlier class to be used for handling outliers.
            
        
        Notes
        -------
        Assignes true to preprocessed attribute.

        Returns
        -------
        Nothing

        """
        
        #Get the scaler class
        self.scaler = scaler

        #Get the outlier class
        self.outlier = outlier

        #If there are missing values, drop them
        if self.data.isnull().sum().sum() != 0:
            print('Missing values found. Dropping... - Done!')
            self.data.dropna(inplace=True)

        #One-Hot encode the data    
        self.data = pd.get_dummies(self.data, dtype=int)
        print('One-hot encoding - Done!')

        #Handle the outliers
        self.data = self.outlier.handle(self.data)
        print('Outlier handling - Done!')

        #Scale the data
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        print(f'Scaling using {self.scaler.__class__.__name__} - Done!')

        #Transformed columns are
        self.fit_columns = self.data.columns

        #The data is preprocessed from now on
        self.preprocessed = True
        
    def fit(self, method = 'silhouette', max_k=50):
        """
        Fits the clustering algorithm to create the clusters.

        Parameters
        ----------
        method : str (available values: 'silhouette', 'elbow')
             (Default value = 'silhouette')
             Represents the method to find the optimal number of clusters of the clustering algorithm.
        max_k :
             (Default value = 50)
             Represents the number of clusters to include in finding the optimal number of clusters.

        Returns
        -------
        Nothing
        """

        #If not preprocessed, warn.
        if self.preprocessed is False:
            print('- Warning: the data is not preprocessed. If you fill your data meets the assumptions, ignore this warning.')

        #Find the numbr of optimal clusters
        self.optimal_k = self.clustering.find_optimal_k(self.data, method = method, max_k=max_k)

        #Fit the clustering algorithm using the optimal number of clusters.
        self.clustering.fit(self.data, self.optimal_k)
        print(f'The data is fitted. The optimal number of clusters found using {method} method is {self.optimal_k}.')
        
    def predict(self, data):
        """
        Predicts the data points based on the fitted clustering model.

        Parameters
        ----------
        data : pandas dataframe or array-like
            Represents the data to be predicted.

        Returns
        -------
        The predicted output.

        """
        return self.clustering.predict(data)
    
    def evaluate(self):
        """ 
        Used to evaluate the fitted clustering model (Simply calculates a silhouette score).
        """
        score = self.clustering.evaluate(self.data)
        print(self.clustering.kmeans.n_clusters)
        print(f'\nThe silhouette score of the clustering is: {score:.4f}')
    
    def __prepare_report(self):

        #If the clustering not fitted, raise an error.
        if not self.clustering.fitted:
            raise NotImplementedError('The clustering is not fitted. Fit the model first!')

        #Make a dataframe from centroids
        self.centroids = pd.DataFrame(self.clustering.kmeans.cluster_centers_, columns = self.fit_columns)

        #Inverse transform the data
        self.centroids = self.scaler.inverse_transform(self.centroids).round(6)

        #Sort by the target column
        self.centroids = self.centroids.sort_values(self.target_column, ascending=False)

        #Reset the index
        self.centroids.reset_index(drop=True, inplace=True)

    def __repr__(self):
        return f'A customer profile creator class.'
        
    def report(self, save=False, save_path=''):
        """
        Prepartes the report of the segmentation model.

        Parameters
        ----------
        save : bool
             (Default value = False)
             Whether to save the report or not.
        save_path : str
             (Default value = '')
             The path to save the report. Works if save==True.

        Returns
        -------
        Nothing
        """

        #Prepare the report
        self.__prepare_report()

        #If save
        if save:

            #Check if save path is string
            if isinstance(save_path, str):

                #Check if it is csv
                if save_path[-4:] == '.csv':

                    #Save
                    self.centroids.to_csv(save_path)

                #If not csv, raise an error
                else:
                    raise ValueError('The data path should end with extension .csv to be saved in a csv file.')
            
            #Raise a value error, the save path is not a string.
            else:
                raise ValueError('The save path is not a string.')
            
        print("="*100)
        print(' '*47 + f'\033[1mReport \033[0m - {self.segmentation} segmentation.')
        print("="*100)

        for cluster_index in range(len(self.centroids)):
            print(f"\n\033[1mCluster {cluster_index}\033[0m - {self.target_column} = {self.centroids.loc[cluster_index, self.target_column]}\n")
            print(self.centroids.iloc[cluster_index])
            print('-'*100)

        #Evalute    
        self.evaluate()
        print("="*100)