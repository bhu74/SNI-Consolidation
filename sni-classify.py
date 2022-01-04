#!/usr/bin/env python
# coding: utf-8

# # SNI Consolidation Algorithm
# 
# ## Introduction - About the challenge
# SNI stands for server name indicator. For HTTPS Connection between Mobile App Client and the server it enables the same IP and Port to be used for multiple Web services (each differentiated by Unique Server Name indicator) as part of Digital certificate. 
# 
# SNIs are one of the biggest contributors to EDR Aggregations which occupy approximately 10% of entire footprint of the client’s Hadoop Cluster, consolidating SNIs would result in significant reduction in cost, storage and related efficiencies. 
# 
# The purpose of this challenge is to identify clusters and group the SNIs in those clusters. Output would be a csv file indicating the cluster names and the number of records in each cluster.
# 
# ## Solution
# The solution is based on anlysis of the SNI information of provided sample data. Two approaches of clustering has been provided.
# 
# ** Approach 1 - Machine Learning Approach
# 
# As this problem involves grouping of text data, Unsupervised clustering method of machine learning has been considered.  
# 
# ~ The domain information from SNI name is extracted and unique 'short' sni-names are identified.
# ~ Levenshtein distance is calculated between the short sni-names
# ~ AffinityPropoagation Algorithm is used to group the short sni-names into clusters. 
# ~ IPv4 and IPv6 sni-names are grouped in separate clusters
# ~ The cluster names are mapped back to the SNI names and the number of records in each cluster is calculated.
# ~ Null records, records with junk characters and not in standard format are classified as 'Invalid' cluster
# 
# ** Approach 2 - Text grouping Approach
# 
# ~ The domain information from SNI name is extracted and unique 'short' sni-names are identified.
# ~ The short sni-names are grouped based on the alphabetical order.
# ~ IPv4 and IPv6 sni-names are grouped in separate clusters
# ~ The cluster names are mapped back to the SNI names and the number of records in each cluster is calculated.
# ~ Null records, records with junk characters and not in standard format are classified as 'Invalid' cluster
# 

# In[1]:


# Import the required modules
import numpy as np
import pandas as pd
import re

from sni_utils import *

from sklearn import cluster
from sklearn.preprocessing import StandardScaler


# Read the input file. Apply the getstr() function to extract short sni-names and the category. The category column is used for the categorization as per Approach 2.
# 
# Segregate the unique short sni-names alone, exclusing the invalid, ipv4 and ipv6 names.

# In[2]:


words = pd.read_csv("sni trends.csv", encoding='latin-1')

words["short"], words["category"] = zip(*words['sni'].apply(getstr))
wd = words[(words.category != "Invalid") & (words.category != "ipv4") & (words.category != "ipv6")].short.unique()


# Call the get_clusters() function to calculate text similarity using Levenshtein distance and apply Affinity Propagation algorithm. Map the clusters to the Sni names using pd.merge command.

# In[3]:


grp = pd.DataFrame()

for i in range(ord('a'), ord('z')+1):
    w = pd.Series(wd)
    wrd1 = np.array(w.loc[w.str.startswith(chr(i))])
    retcls = get_clusters(wrd1)
    grp = grp.append(retcls)

wrd1 = np.array(w.loc[w.str.startswith(('0','1','2','3','4'))])
retcls = get_clusters(wrd1)
grp = grp.append(retcls)

wrd1 = np.array(w.loc[w.str.startswith(('5','6','7','8','9'))])
retcls = get_clusters(wrd1)
grp = grp.append(retcls)

result = pd.merge(words,grp,how='outer',left_on='short',right_on='short_sni')


# Call getgrp() function to get group1 cluster values based on the category and cluster name
# Rename category to group2 - This is the clustering based on approach2.

# In[4]:


result['group1'] = result.apply(lambda x: getgrp(x.short, x.category, x.group), axis=1)
result.rename(columns={'category':'group2'}, inplace=True)


# Generate the output files 

# In[5]:


# Sum the values of group1 column values and write to Approach1.csv. 
# This gives the records for each cluster identified by approach 1(machine learning) 
option1 = result['group1'].value_counts()
option1.to_csv("Approach1.csv")

# Sum the values of group2 column values and write to Approach2.csv. 
# This gives the records for each cluster identified by approach 2(text grouping) 
option2 = result['group2'].value_counts()
option2.to_csv("Approach2.csv")

