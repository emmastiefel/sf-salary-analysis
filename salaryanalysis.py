
# coding: utf-8

# TECHNICAL APPENDIX

# In[2]:


#import all libraries needed and read in csv file
import scipy.stats as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Salaries.csv')


# In[3]:


#Cell 1: Confidence interval and distribution for all total salaries

#plots a frequency distribution for all total salaries recorded, showing that the distribution is skewed to the right
#the skewed distribution means that the confidence interval is inaccurate
plt.hist(data['TotalPay'])

#calculates z*standard error for a 95% confidence interval
#multiplies the standard deviation for all total salary data by 1.96, the z-score for a two-tailed 95% interval
zse = np.std(data['TotalPay']) * 1.96

#uses np.mean to find the center of the confidence interval
print("Mean: ", np.mean(data['TotalPay']))

#constructs the confidence interval by adding and subtracting z* standard error to the mean
print("Confidence interval: (", np.mean(data['TotalPay']) - zse, ", ", np.mean(data['TotalPay']) + zse, ")")


# In[4]:


#Cell2: Minimum and maximum total salaries recorded

#finds the minimum total salary recorded, which was actually negative
print(data.loc[data['TotalPay'] == min(data['TotalPay'])])

#finds the maximum total salary recorded
print(data.loc[data['TotalPay'] == max(data['TotalPay'])])


# In[6]:


#Cell 3: Confidence interval and distribution for only full-time total salaries

#filters data so that only rows where status is "FT" (full-time) are included
fulltime = data.loc[data['Status'] == 'FT']

#plots a frequency distribution for full-time total salaries recorded, showing that the distribution is skewed to the right
#again, the skewed distribution means that the confidence interval is inaccurate
plt.hist(fulltime['TotalPay'])

#calculates z*standard error for a 95% confidence interval
#multiplies the standard deviation for full-time total salary data by 1.96, the z-score for a two-tailed 95% interval
zse = np.std(fulltime['TotalPay']) * 1.96

#uses np.mean to find the center of the confidence interval
print("Mean:", np.mean(fulltime['TotalPay']))

#constructs the confidence interval by adding and subtracting z* standard error to the mean
print("Confidence interval: (", np.mean(fulltime['TotalPay']) - zse, ", ", np.mean(fulltime['TotalPay']) + zse, ")")


# In[8]:


#Cell 4: Confidence interval and distribution for only part-time total salaries

#filters data so that only rows where status is "PT" (part-time) are included
parttime = data.loc[data['Status'] == 'PT']

#plots a frequency distribution for part-time total salaries recorded, showing that the distribution is skewed to the right
#again, the skewed distribution means that the confidence interval is inaccurate
plt.hist(parttime['TotalPay'])

#calculates z*standard error for a 95% confidence interval
#multiplies the standard deviation for part-time total salary data by 1.96, the z-score for a two-tailed 95% interval
zse = np.std(parttime['TotalPay']) * 1.96

#uses np.mean to find the center of the confidence interval
print("Mean:", np.mean(parttime['TotalPay']))

#constructs the confidence interval by adding and subtracting z* standard error to the mean
print("Confidence interval: (", np.mean(parttime['TotalPay']) - zse, ", ", np.mean(parttime['TotalPay']) + zse, ")")


# In[9]:


#Cell 5: demonstrates that 2014 was the only year where employee status was recorded

#iterates through each year recorded
for x in [2011, 2012, 2013, 2014]:
    #creates a separate dataset for all values recorded for that year
    this_year = data2011 = data.loc[data['Year'] == x]
    #prints the number of unique values for 'Status' recorded in that year
    print(x, this_year['Status'].unique())


# In[10]:


#Cell 6: Begins to examine salary increases across years

#creates a separate dataset for each year
data2011 = data.loc[data['Year'] == 2011]
data2012 = data.loc[data['Year'] == 2012]
data2013 = data.loc[data['Year'] == 2013]
data2014 = data.loc[data['Year'] == 2014]

#iterates through data for each year and prints the mean of each
datas = [data2011, data2012, data2013, data2014]
for i in datas:
    print(i.iloc[0]['Year'], "mean salary:", np.mean(i['TotalPay']))

#calculates the average increase between two years 
print("Mean year-to-year increase:", (np.mean(data2014['TotalPay']) - np.mean(data2011['TotalPay'])) / 3)

#prints the total mean salary increase between 2011 and 2014
print("2011 to 2014 increase:", np.mean(data2014['TotalPay']) - np.mean(data2011['TotalPay']))


# In[11]:


#Cell 7: difference of sample means hypothesis test and effect size

#defines a function calculates a p-value and effect size for the difference of means between two provided total salary samples
def dif_means_test(data1, data2):

    #calculates the mean of each provided sample
    mean1 = np.mean(data1['TotalPay'])
    mean2 = np.mean(data2['TotalPay'])
    
    #calculates the size of each provided sample
    n1 = len(data1['TotalPay'])
    n2 = len(data2['TotalPay'])
    
    #calculates the standard deviation of each provided sample
    sd1 = np.std(data1['TotalPay'])
    sd2 = np.std(data2['TotalPay'])
    
    #uses the pooled standard deviation formula, factoring both sample sizes and standard deviations into the final calculations
    sdpooled =  (((sd1 ** 2) / n1) + ((sd2 ** 2) / n2)) ** 0.5 
    
    #calculates a z-score for the difference between the two means
    #a z-score (rather than a t-score) is appropriate because both sample sizes are very large
    zscore = (mean1 - mean2) / sdpooled
    
    #uses the scipy.stats library to convert the zscore to a one-tailed probability
    p = st.norm.cdf(zscore)
    
    #converts the one-tailed probability to a two-tailed probability
    #because we are interested in any change, not just an increase or decrease
    if p > 0.5: 
        pvalue = (1 - p) * 2
    else:
        pvalue = p * 2
        
    #the null hypothesis is that the difference between the two means is 0
    #the alternative hypothesis is that the difference between the two means is not equal to zero
    #to reject or fail to reject the null hypothesis, the outputted p-value must be compared to the 0.05 alpha level
        
    #calculates the effect size for the difference of means using the equation for Cohen's D
    #the outputted effect size must be interpreted according to the standard 0.2 = small, 0.5 = moderate, 0.8 = large scale
    es = (mean1 - mean2) / ((sd1 + sd2) / 2)
             
    print('P-value:', pvalue, "\nEffect size:", es)


# In[12]:


#Cell 8: performs the hypothesis test for total salaries in 2011 and 2014 (see cell 7)

dif_means_test(data2014, data2011)

#the p-value is below the 0.05 cutoff, so the null hypothesis is rejected
#the effect size is below 0.2, so it is quite small, but shows that 2014 salaries were higher on average


# In[13]:


#Cell 9: performs the hypothesis test for the highest and lowest total salaries (see cell 7)

#calculates the lowest and highest 10th percentile cutoffs
print("Lowest 10% below:", data['TotalPay'].quantile(0.1))
print("Highest 10% above:", data['TotalPay'].quantile(0.9))

#creates a dataset that contains only the top 10% of salaries
highest = data.loc[data['TotalPay'] > data['TotalPay'].quantile(0.9)]

#filters the top 10% salary dataset by the year in which the salaries were earned
highest2011 = highest.loc[highest['Year'] == 2011]
highest2014 = highest.loc[highest['Year'] == 2014]

#creates a dataset that contains only the bottom 10% of salaries
lowest = data.loc[data['TotalPay'] < data['TotalPay'].quantile(0.1)]

#filters the bottom 10% salary dataset by the year in which the salaries were earned
lowest2011 = lowest.loc[lowest['Year'] == 2011]
lowest2014 = lowest.loc[lowest['Year'] == 2014]

#calculates the p-value and effect size for the change between 2011 and 2014 in the upper and lower 10% of salaries

print("Highest 10%:")
dif_means_test(highest2014, highest2011)
#the p-value is below the alpha level, so the change was significant
#the effect size is still small, but less so than that for all salaries, and shows there was an increase

print("Lowest 10%:")
dif_means_test(lowest2014, lowest2011)
#the p-value is above the 0.05 alpha level, so the change was not significant
#the effect is size is very small, but shows that there was a slight decrease

#calculates the mean for the lowest salaries in 2011 and 2014, showing that there was a decrease
print("Lowest 2011 mean:", np.mean(lowest2011['TotalPay']))
print("Lowest 2014 mean:", np.mean(lowest2014['TotalPay']))


# In[14]:


#Cell 10: conditional probability of being in the top 10% of salaries given that a salary was paid in 2014

#the probability of being in the top 10% is, by definition, 0.1
prich = 0.1

#calculates the probability of a salary being from 2014 (~.26)
p2014 = len(data2014) / len(data)

#calculates the probability of a salary being in both the top 10% and from 2014, using the dataset from the previous cell
pboth = len(highest2014) / len(data)

#applies conditional probability formula ( p(a|b) = p(a, b) / p(b) ) to find p(top10% | 2014)
cp = pboth / p2014

print("P(top 10% | 2014) =", cp)


# In[15]:


#Cell 11: conditional probability of being in the top 10% of salaries given that a salary was paid in 2011

#the probability of being in the top 10% is, by definition, 0.1
prich = 0.1

#calculates the probability of a salary being from 2011 (~.24)
p2011 = len(data2011) / len(data)

#calculates the probability of a salary being in both the top 10% and from 2011
pboth = len(highest2011) / len(data)

#applies conditional probability formula ( p(a|b) = p(a, b) / p(b) ) to find p(top10% | 2011)
cp = pboth / p2011

print("P(top 10% | 2011) =", cp)


# In[16]:


#Cell 12: conditional probability of being in the bpttom 10% of salaries given that a salary was paid in 2014

#the probability of being in the bottom 10% is, by definition, 0.1
ppoor = 0.1

#calculates the probability of a salary being from 2014 (~.26)
p2014 = len(data2014) / len(data)

#calculates the probability of a salary being in both the bottom 10% and from 2014
pboth = len(lowest2014) / len(data)

#applies conditional probability formula ( p(a|b) = p(a, b) / p(b) ) to find p(bottom10% | 2014)
cp = pboth / p2014

print("P(bottom 10% | 2014) =", cp)


# In[17]:


#Cell 13: conditional probability of being in the bpttom 10% of salaries given that a salary was paid in 2011

#the probability of being in the bottom 10% is, by definition, 0.1
ppoor = 0.1

#calculates the probability of a salary being from 2011 (~.24)
p2011 = len(data2011) / len(data)

#calculates the probability of a salary being in both the bottom 10% and from 2011
pboth = len(lowest2011) / len(data)

#applies conditional probability formula ( p(a|b) = p(a, b) / p(b) ) to find p(bottom10% | 2011)
cp = pboth / p2011

print("P(bottom 10% | 2011) =", cp)

