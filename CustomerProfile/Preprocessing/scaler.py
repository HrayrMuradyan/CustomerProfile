"""
Scaler module that contains all the scaling algorithms.
For now MinMaxScaler and StandardScaler are implemented.
"""

import numpy as np 
import pandas as pd
from abc import ABC, abstractmethod


class Scaler(ABC):
    """ 
    Base Scaler class.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        """
        Fits the dataset for acquiring the statistics.

        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be fitted.
        """
        pass

    @abstractmethod
    def fit_transform(self, X):  
        """
        Fits the dataset and then transforms it.
    
        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be fitted and transformed.
        """
        pass

    @abstractmethod
    def transform(self, X):  
        """
        Transforms the dataset to the same scale of features.

        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be transformed.
        """
        pass

    @abstractmethod
    def inverse_transform(self, X):
        """
        Inverse transforms the dataset to the original feature scale.

        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be inversely transformed.
        """
        pass


class MinMaxScaler(Scaler):
    """ 
    The MinMaxScaler class used for transforming the features into the same scale.

    Attributes
    ----------
    min : None
        To store the minimum of the features of the data.

    max : None
        To store the maximum of the features of the data.

    """
    def __init__(self):
        """
        Initializes the MinMaxScaler class instance.
        """
        self.min = None
        self.max = None
        
    def fit(self, X):
        """
        Fits the dataset.
    
        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be fitted and transformed.
        """
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
    
    def fit_transform(self, X):      
        """
        Fits the dataset and then transforms it.
    
        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be fitted and transformed.

        Returns
        ----------
        The transformed data.
        """
        self.fit(X) 
        transformed_data = (X - self.min)/(self.max-self.min)
        
        return transformed_data
    
    def transform(self, X):     
        """
        Transforms the dataset to the same scale of features.

        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be transformed.

        Returns
        ----------
        The transformed data.
        """
        if self.min is None or self.max is None:
            raise NotImplementedError('The MinMaxScaler should be fitted first. Run .fit() on the train data before running transform!')
        
        transformed_data = (X - self.min)/(self.max-self.min)
        
        return transformed_data

    def __repr__(self):
        return f'A Min Max Scaler class.'
    
    def inverse_transform(self, X):
        """
        Inverse transforms the dataset to the original feature scale.

        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be inversely transformed.

        Returns
        ----------
        The unnormalized data.
        """
        if self.min is None or self.max is None:
            raise NotImplementedError('The MinMaxScaler should be fitted first. Run .fit() on the train data before running transform!')
        
        #If the maximum is higher than 1, the features most probably are not scaled.
        if X.max().max() > 1.0 + 1e-10:
            raise ValueError('It seems the features are not scaled. Since the maximum of the data is higher than 1...')
        
        #For dataframe compute this way
        if isinstance(X, pd.DataFrame):
            unnormalized_data = X*(self.max - self.min) + self.min
        
        #For numpy array compute this way
        elif isinstance(X, np.ndarray):
            unnormalized_data = X*(np.expand_dims(self.max, 0) - np.expand_dims(self.min, 0)) + np.expand_dims(self.min, 0)
        
        return unnormalized_data
        
        
class StandardScaler(Scaler):
    """ 
    The StandardScaler class used for transforming the features into the same scale by transforming the feature to normal ditribution with mean 0 and std 1.

    Attributes
    ----------
    std : None
        To store the std of the features of the data.

    mean : None
        To store the mean of the features of the data.

    """
    def __init__(self):
        """
        Initializes the StandardScaler class instance.
        """
        self.std = None
        self.mean = None
        
    def fit(self, X):
        """
        Fits the dataset.
    
        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be fitted and transformed.
        """
        self.std = np.std(X, axis=0)
        self.mean = np.mean(X, axis=0)
        
    def fit_transform(self, X):      
        """
        Fits the dataset and then transforms it.
    
        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be fitted and transformed.

        Returns
        ----------
        The transformed data.
        """
        self.fit(X) 
        transformed_data = (X - self.mean)/self.std
        
        return transformed_data
    
    def transform(self, X):     
        """
        Transforms the dataset to the same scale of features.

        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be transformed.

        Returns
        ----------
        The transformed data.
        """
        if self.mean is None or self.std is None:
            raise NotImplementedError('The MinMaxScaler should be fitted first. Run .fit() on the train data before running transform!')
        
        transformed_data = (X - self.mean)/self.std
        
        return transformed_data

    def inverse_transform(self, X):
        """
        Inverse transforms the dataset to the original feature scale.

        Parameters
        ----------
        X : pandas dataframe
            Represents the dataframe to be inversely transformed.

        Returns
        ----------
        The unnormalized data.
        """

        #If not fitted yet raise an error
        if self.std is None or self.mean is None:
            raise NotImplementedError('The MinMaxScaler should be fitted first. Run .fit() on the train data before running transform!')
        
        #For dataframe compute this way
        if isinstance(X, pd.DataFrame):
            unnormalized_data = X*self.std + self.mean
        
        #For numpy array compute this way
        elif isinstance(X, np.ndarray):
            unnormalized_data = X*np.expand_dims(self.std, 0) + np.expand_dims(self.mean, 0)
        
        return unnormalized_data

    def __repr__(self):
        return f'A Standard Scaler class.'
        