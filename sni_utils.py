import numpy as np
import pandas as pd
import re
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

def getstr(inp):
    """
    Fetches domain names (short SNI names) from the input SNI name
    
    Arguments:
    inp -- input SNI name
        
    Returns:
    s -- short SNI, ipv4, ipv6 names
    catg -- Category (a-group, b-group...z-group, num-group, Invalid). This category value is also the group names for approach 2 
    """
    #Convert input string to lower case
    x = str(inp).lower()
    
    # Use regular expression to identify ipv4(p1) and ipv6(p2) names
    p1 = re.findall(r"^([0-9]*\.[0-9]*\.[0-9]*)\.[0-9]*$", x)
    p2 = re.findall(r"^(.*)::", x)
    
    # Set category as "ipv4" or "ipv6" if regular expression return value is not 0
    # Else, extract short sni names and set category based on starting letter of short sni-name
    if len(p1) > 0:
        s = p1[0]
        catg = "ipv4"
    elif len(p2) > 0:
        s = p2[0]
        catg = "ipv6"
    elif len(x.split('.')) > 1:
        s = x.split('.')[len(x.split('.'))-2]
        if len(s) > 0:
            if s[0] >= '0' and s[0] <= '9':
                catg = 'num-group'
            else: catg = s[0]+'-group'
        else: catg = "Invalid"
    else:
        s = ""
        catg = "Invalid"
    
    return s, catg
    
def levenshtein(s1, s2):
    """
    Computes Levenshtein distance between two input strings using dynamic programming technique.
    Levenshtein distance (LD) is a measure of the similarity between two strings. 
    It is the number of deletions, insertions, or substitutions required to transform s1 into s2.
    
    Arguments:
    s1 -- input string 1
    s2 -- input string 2
        
    Returns:
    leven -- Levenshtein distance
    """
    # Compute length of input strings
    i = len(str(s1))
    j = len(str(s2))
    
    d = np.zeros((i+1, j+1))
    levn = 0
    
    # If length of any string is 0, distance = length of other string
    # If the strings are same, then distance = 0
    # If the strings are not same,
    #      Construct a matrix d of rows = s1 length+1 and columns = s2 length+1
    #      Initialize first row and column to values from 0 to row length and column length
    #      If the row letter = column letter, then d[i,j] = diagonally opposite value (d[i-1][j-1])
    #      if row letter <> column letter, then d[i,j] is minimum of d[row-1], d[column-1], d[row-1, column-1] values +1
    #      The value of last row, column value of d (d[i][j]) gives the distance 
    
    if min(i,j) == 0:
        levn = max(i, j)
    elif s1 == s2:
        levn = 0
    else:
        for r in range(i+1):
            d[r][0] = r
        for c in range(j+1):
            d[0][c] = c
        
        for c in range(1, j+1):
            for r in range(1, i+1):
                if s1[r-1] == s2[c-1]:
                    d[r][c] = d[r-1][c-1]
                else:
                    d[r][c] = min(d[r-1][c], d[r-1][c-1], d[r][c-1]) + 1
        levn = d[i][j]
    
    return levn

def get_clusters(wdlst):
    """
    Computes text similarity between the words of the input words list and applies Affinity Propagation algorithm to identify clusters
    
    Arguments:
    wdlst -- input list of words that needs to be clustered
            
    Returns:
    itr -- Dataframe of short sni names and the corresponding cluster
    """
    # Compute text similarity array - Call Levenshtein function for each word in the list to the other words in the list. 
    # Normalize the similarity array using Scikit-learn's Standard Scalar function
    lev_similarity = -1 * np.array([[levenshtein(w1,w2) for w1 in wdlst] for w2 in wdlst])
    
    lev_similarity = StandardScaler().fit_transform(lev_similarity)
    
    # Apply Scikit-learn's AffinityPropagation algorithm. Get Cluster indices, number of clusters and labels  
    pref = np.min(lev_similarity)
    affprop = cluster.AffinityPropagation(affinity="precomputed", preference=pref, damping=0.5)
    result = affprop.fit(lev_similarity)
    
    cluster_centers_indices = affprop.cluster_centers_indices_
    labels = affprop.labels_
    n_clusters = len(cluster_centers_indices)
    
    # Get the corresponding cluster name for the word and create dataframe to be returned
    itr = pd.DataFrame()
    s = pd.Series(wdlst)
    itr['short_sni'] = s
    
    x = []
    for i in range(len(wdlst)):
        x.append(wdlst[labels[i]])
        
    itr['group'] = x
    
    return(itr)
    
def getgrp(ssni, ctg, tmpg):
    """
    Identifies the group name based on identified cluster / ipv4 / ipv6 / Invalid group based on previously computed values
    
    Arguments:
    ssni -- short sni name
    catg -- category name
    tmpg -- group name 
            
    Returns:
    ret -- group name (short sni name / cluster name / Invalid)
    """
    
    # If category name starts with "ip", return short sni name
    # if category is "Invalid", return "Invalid"
    # Else, return the group name as is
    ret = tmpg
    if str(ctg)[0:2] == "ip":
        ret = ssni
    elif str(ctg) == "Invalid":
        ret = "Invalid"
    
    return ret
