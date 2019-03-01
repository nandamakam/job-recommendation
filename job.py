"""imported the libraries needed"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast 

"""imported the required machine learning helpers"""
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')



"""loading the datasets"""
ls ./data/*.tsv
apps = pd.read_csv('./data/apps.tsv', delimiter='\t',encoding='utf-8')
user_history = pd.read_csv('./data/user_history.tsv', delimiter='\t',encoding='utf-8')
jobs = pd.read_csv('./data/jobs.tsv', delimiter='\t',encoding='utf-8', error_bad_lines=False)
users = pd.read_csv('./data/users.tsv' ,delimiter='\t',encoding='utf-8')
test_users = pd.read_csv('./data/test_users.tsv', delimiter='\t',encoding='utf-8')

#apps dataset
""" header of apps dataset""" 
apps.head()

""" columns of apps dataset"""
apps.columns


"""shape of apps dataset"""
apps.shape

""" info of apps dataset"""
apps.info()





#user_history dataset
""" header of user_history dataset""" 
user_history.head()

""" columns of user_history dataset"""
user_history.columns


"""shape of user_history dataset"""
user_history.shape

""" info of user_history dataset"""
user_history.info()





#jobs dataset
""" header of jobs dataset""" 
jobs.head()

""" columns of jobs dataset"""
jobs.columns


"""shape of jobs dataset"""
jobs.shape

""" info of jobs dataset"""
jobs.info()


#users dataset
""" header of users dataset""" 
users.head()

""" columns of users dataset"""
users.columns

"""shape of users dataset"""
users.shape

""" info of users dataset"""
users.info()



#test_users dataset
""" header of test_users dataset""" 
test_users.head()

""" columns of test_users dataset"""
test_users.columns

"""shape of test_users dataset"""
test_users.shape

""" info of test_users dataset"""
test_users.info()





"""Exploratory Data Analysis (EDA) 
Split training and testing data based on column split
Here, there are three datafiles/dataframes are having attribute split.
1.apps
2.user_history
3.users
This data attribute indicates that whether the data record can be used for training or testing so we need to filter out based on that.
We are generating training and testing dataframes
"""

#apps_training
apps_training = apps.loc[apps['Split'] == 'Train']
apps_training.shape  #shaping of apps_training set
apps_training.head()   #header information

#apps_testing
apps_testing = apps.loc[apps['Split'] == 'Test']
apps_testing.shape      #shaping of apps_testing set
apps_testing.head()     #header information


#user_history_training
user_history_training = user_history.loc[user_history['Split'] =='Train']
user_history_training.shape
user_history_training.head()

#user_history_testing
user_history_testing = user_history.loc[user_history['Split'] =='Test']
user_history_testing.shape
user_history_testing.head()



#users_training
users_training = users.loc[users['Split']=='Train']
users_training.shape
users_training.head()

#users_testing
users_testing = users.loc[users['Split']=='Test']
users_testing.shape
users_testing.head()


"""EDA for job openings based on their location information"""
jobs.groupby(['City','State','Country']).size().reset_index(name='Locationwise')  #grouping the columns
jobs.groupby(['Country']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False).head()
Country_wise_job = jobs.groupby(['Country']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False)

"""ploting the graph for the jobs"""
plt.figure(figsize=(12,12))
ax = sns.barplot(x="Country", y="Locationwise", data=Country_wise_job)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_title('Country wise job openings')
plt.tight_layout()
plt.show()


"""pre-processing
Now we will perform following pre-processing steps:
We will consider only US region for building this recommendation engine
We will be removing data records where state is blank or state data attribute is having numerical value.(If needed)"""

"""state wise"""
jobs_US = jobs.loc[jobs['Country']=='US']
#grouping
jobs_US.groupby(['City','State','Country']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False).head()
#grouping state wise
State_wise_job_US = jobs_US.groupby(['State']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False)
#plotting for the state 
plt.figure(figsize=(12,12))
ax = sns.barplot(x="State", y="Locationwise",data=State_wise_job_US)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_title('State wise job openings')
plt.tight_layout()
plt.show()

"""city wise"""
jobs_US.groupby(['City']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False)
#grouping citywise
City_wise_location = jobs_US.groupby(['City']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False)
City_wise_location_th = City_wise_location.loc[City_wise_location['Locationwise']>=12]
plt.figure(figsize=(12,12))
ax = sns.barplot(x="City", y="Locationwise",data=City_wise_location_th.head(50))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_title('City wise job openings')
plt.tight_layout()
plt.show()


"""EDA for User profiles based on their location information"""
users_training.groupby(['Country']).size().reset_index(name='Locationwise').sort_values('Locationwise',ascending=False).head()
user_training_US = users_training.loc[users_training['Country']=='US']
#grouping state wise
user_training_US.groupby(['State']).size().reset_index(name='Locationwise_state').sort_values('Locationwise_state',ascending=False)
user_training_US_state_wise = user_training_US.groupby(['State']).size().reset_index(name='Locationwise_state').sort_values('Locationwise_state',ascending=False)
user_training_US_th = user_training_US_state_wise.loc[user_training_US_state_wise['Locationwise_state']>=12]
#plot the graph
plt.figure(figsize=(12,12))
ax = sns.barplot(x="State", y="Locationwise_state",data=user_training_US_th.head(50))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_title('State wise job seekers')
plt.tight_layout()
plt.show()

#grouping citywise
user_training_US.groupby(['City']).size().reset_index(name='Locationwise_city').sort_values('Locationwise_city',ascending=False)
user_training_US_city_wise = user_training_US.groupby(['City']).size().reset_index(name='Locationwise_city').sort_values('Locationwise_city',ascending=False)
user_training_US_City_th = user_training_US_city_wise.loc[user_training_US_city_wise['Locationwise_city']>=12]
#plot the graph
plt.figure(figsize=(12,12))
ax = sns.barplot(x="City", y="Locationwise_city",data=user_training_US_City_th.head(50))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_title('State wise job seekers')
plt.tight_layout()
plt.show()



