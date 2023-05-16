"""
Outlier module that contains all the outlier handling algorithms.
For now ZScore outlier handling algorithm is implemented.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod



class Outlier(ABC):
	""" 
	Base Outlier class.
	"""
	@abstractmethod
	def __init__(self):
		pass

	def handle(self, data):
		"""
		Handles the dataset outliers using the columns provided.

		Parameters
		----------
		data : pandas dataframe
			Represents the dataset to be handled
		"""
		pass



class ZScore(Outlier):
	""" 
	Creates a Z-Score outlier handler class.

	Parameters
    ----------
    column : str or list ('all' or list of available columns in the data)
        Represents the columns that need to be considered.

	"""
	def __init__(self, column='all'):
		"""
		Initializes ZScore class instance. 
		"""
		self.column = column

	def handle(self, data):
		"""
		Handles the dataset outliers using the columns provided.

		Parameters
		----------
		data : pandas dataframe
			Represents the dataset to be handled

		Returns
		-------
		The corrected dataset.
		"""

		#Check if the column is all
		if self.column == 'all':
			cols = data.columns

		#Else if it is string
		elif isinstance(self.column, str):
			cols = [self.column]

		#Else take the list
		elif isinstance(self.column, list):
			cols = self.column
	        
		data = data.copy()

		#For column in the columns
		for col in cols:

			#Calculate the upper and lower whiskers
			Q1 = data[col].quantile(0.25)  # Lower whisker
			Q3 = data[col].quantile(0.75)  # Upper whisker

			#Get the interquartile range
			IQR = Q3 - Q1
			lower_whisker = Q1 - 1.5 * IQR
			upper_whisker = Q3 + 1.5 * IQR

			#Find the number of outliers
			n_upper_outliers = int((data[col] > upper_whisker).sum())
			n_lower_outliers = int((data[col] < lower_whisker).sum())

			if any([n_upper_outliers, n_lower_outliers]):
				print(f'There were {n_upper_outliers} upper outliers and {n_lower_outliers} lower outliers found in column "{col}". Clipping...')

			#Cip the outliers
			data[col] = np.clip(data[col], lower_whisker, upper_whisker)

		return data