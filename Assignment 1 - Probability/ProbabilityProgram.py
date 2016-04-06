
# coding: utf-8

# In[344]:

#get_ipython().magic(u'matplotlib inline')
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import numpy as np
import scipy as ss
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab
import sys


# In[345]:

filename = sys.argv[1]
#input = pd.read_csv("2015_CHR_Analytic_Data.csv")
input = pd.read_csv(filename)
input = input[input['COUNTYCODE']!=0]


# ### 1) Column Headers which end in Value

# In[346]:

all_columns = input.columns
value_column_names = []
print("(1) COLUMN HEADERS: ")
for col_name in all_columns:
    if str(col_name).endswith("Value"):
        print col_name
        value_column_names.append(col_name)


# ### 2) Total Counties in File 

# In[347]:

distinct_counties = pd.unique(input.County)
all_counties = input['County']
print "(2) TOTAL COUNTIES IN FILE: " + str(len(distinct_counties))


# ### 3) Total Ranked Counties in File

# In[348]:

ranked_counties = input[input['County that was not ranked'] != 1]
print("(3) TOTAL RANKED COUNTIES: " + str(len(ranked_counties)))


# ### 4) Histogram of Population

# In[349]:

pop = input["2011 population estimate Value"]
population_list = []

for p in pop:
    population_list.append(p.replace(',',''))


# In[350]:

pop = pd.Series(population_list)
pop = pd.to_numeric(pop)


# In[351]:

plt.hist(pop,bins=10)
plt.ylabel("Number of Counties")
plt.xlabel("Population Estimate Values")
plt.title("Histogram of Population")
plt.show()


# ### 5) Histogram of log population

# In[352]:

log_pop = np.log(pop)


# In[353]:

plt.hist(log_pop,bins=150)
plt.ylabel("Number of records in log scale")
plt.xlabel("Log Population Estimate Value")
plt.title("Histogram of Log Population")
plt.show()


# In[354]:

input['log_pop'] = log_pop
input.head()


# ### 6) KERNEL DENSITY ESTIMATES 

# In[355]:

def kde_plot(kernel, X, legend,color="#aaaaff", bw = 0.5):
    #create the estimator:
    kde_X = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
              
    #setup range:
    range = np.linspace(X.min()-bw*3, X.max()+bw*3, 1000)[:,np.newaxis]
    #plot:
    plt.fill(range[:,0], np.exp(kde_X.score_samples(range)), fc=color, alpha=.6,label=legend)
    dots = [y-np.random.rand()*.005 for y in np.zeros(X.shape[0])] #all points, randomly jitte 
    plt.plot(X[:,0], dots, '+k', color=color)

    return kde_X


# In[356]:

log_pop_counties_not_ranked = input[input['County that was not ranked'] == 1]['log_pop']
log_pop_counties_not_ranked = log_pop_counties_not_ranked.reshape(-1,1)

log_pop_counties_ranked = input[input['County that was not ranked'] != 1]['log_pop']
log_pop_counties_ranked = log_pop_counties_ranked.dropna()
log_pop_counties_ranked = log_pop_counties_ranked.reshape(-1,1)

plt.figure(figsize=(10,10))

# https://en.wikipedia.org/wiki/Kernel_density_estimation - Used this formula for bandwidth selection

bwr = (1.06)*(float(log_pop_counties_ranked.std()))*(len(log_pop_counties_ranked)**(-0.2))
bwnr = (1.06)*(float(log_pop_counties_not_ranked.std()))*(len(log_pop_counties_not_ranked)**(-0.2))

kde_not_ranked_model = kde_plot('gaussian', log_pop_counties_not_ranked,'Counties Not Ranked','#22aa22',bwnr)
kde_ranked_model = kde_plot('gaussian',log_pop_counties_ranked,'Counties Ranked','#aaaaff',bwr)

plt.legend(loc='best',markerscale= 1.5,fontsize=12)
plt.title("KERNEL DENSITY ESTIMATES OF COUNTIES")
plt.ylabel("Normalized Density")
plt.xlabel("Log Population Estimate Values")
plt.show()



# ### 7) PROBABILITY RANKED GIVEN POP

# In[357]:

new_pop = 300
print np.exp(kde_ranked_model.score(np.log(new_pop)))

new_pop = 3100
print np.exp(kde_ranked_model.score(np.log(new_pop)))

new_pop = 5000
print np.exp(kde_ranked_model.score(np.log(new_pop)))


# ### 8) LIST MEAN AND STD_DEV PER COLUMN

# In[358]:

dict_col_mean_std = {}

for val in value_column_names:
    col_values = input[val]
    values = []

    for p in col_values:
        values.append(float(str(p).replace(',','')))
    
    values_pd = pd.Series(values)
    values_pd.dropna()
    dict_col_mean_std[val] = (values_pd.mean(),values_pd.std())

print dict_col_mean_std
    


# ### 9) PSEUDO POP DEPENDENT COLUMNS

# In[359]:

log_pop = input['log_pop']
log_pop_mean = log_pop.mean()


# In[364]:

counties_lessthan_mean = input[input['log_pop'] < log_pop_mean]
counties_morethan_mean = input[input['log_pop'] > log_pop_mean]


# In[382]:

pseudo_pop_dependent_cols = []
for val in value_column_names:
    
    col_values = input[val]
    values = []

    for p in col_values:
        values.append(float(str(p).replace(',','')))
    
    values_pd = pd.Series(values)
    values_pd.dropna()
    A_std = values_pd.std()
    
    col_values1 = counties_morethan_mean[val]
    values1 = []

    for p in col_values1:
        values1.append(float(str(p).replace(',','')))
    
    values_pd1 = pd.Series(values1)
    values_pd1.dropna()
    
    A_more_mean = values_pd1.mean()
    
    col_values2 = counties_lessthan_mean[val]
    values2 = []

    for p in col_values2:
        values2.append(float(str(p).replace(',','')))
    
    values_pd2 = pd.Series(values2)
    values_pd2.dropna()
    
    A_less_mean = values_pd2.mean()
        
    if(abs(A_more_mean - A_less_mean) < 0.5*A_std):
        pseudo_pop_dependent_cols.append(val)


# In[383]:

print pseudo_pop_dependent_cols


# In[ ]:



