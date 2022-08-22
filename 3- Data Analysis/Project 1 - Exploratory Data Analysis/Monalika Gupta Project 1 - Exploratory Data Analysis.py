#!/usr/bin/env python
# coding: utf-8

# # Dataset Description
# - The dataset was released by Aspiring Minds from the Aspiring Mind Employment Outcome 2015 (AMEO). 
# - The dataset contains the employment outcomes of engineering graduates as dependent variables (Salary, Job Titles, and Job Locations) along with the standardized scores from three different areas – cognitive skills, technical skills and personality skills. The dataset also contains demographic features. 
# - The dataset contains around 40 independent variables and 4000 data points. The independent variables are both continuous and categorical in nature.

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats

import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_excel('aspiring_minds_employability_outcomes_2015.xlsx')
df.head()


# In[4]:


df.shape


# - There is 3998 rows and 39 columns are present in dataset.

# In[5]:


df.info()


# In[6]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_df = df.select_dtypes(include=numerics)
len(numeric_df.columns)


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.drop(['Unnamed: 0','ID','10board','12board','CollegeID','CollegeCityID'],axis=1,inplace=True)
df.head()


# In[10]:


df.shape


# # Exploratory Data Analysis and Visualization:

# In[13]:


# PDF distribution of Salary:
sns.set_style("darkgrid")
plt.figure(figsize=(8,6))
sns.distplot(df['Salary'])
plt.axvline(df['Salary'].mean(),color='black', label='Mean')
plt.title("Pdf distribution of Salary")
plt.legend(shadow=True,fontsize="larger")

skew = df['Salary'].skew()
kurt = df['Salary'].kurt()
print('Skewness:{}'.format(round(skew,2)))
print('Kurtosis:{}'.format(round(kurt,2)))


# - we can see that it's having a long tail at the right side , looks like log normal distribution.

# In[12]:


# PDF distribution of 10percentage:
sns.set_style("darkgrid")
plt.figure(figsize=(8,6))
sns.distplot(df['10percentage'])
plt.axvline(df['10percentage'].mean(), color="blue", label="Mean")
plt.title("Pdf distribution of 10percentage")
plt.legend(shadow=True,fontsize="larger")

skew = df['10percentage'].skew()
kurt = df['10percentage'].kurt()
print('Skewness:{}'.format(round(skew,2)))
print('Kurtosis:{}'.format(round(kurt,2)))


# - It's having a thick tail towards the left side looks like a negatively skewed distribution.

# In[14]:


# PDF distribution of 12percentage:
sns.set_style("darkgrid")
plt.figure(figsize=(8,6))
sns.distplot(df['12percentage'])
plt.axvline(df['12percentage'].mean(), color="brown", label="Mean")
plt.title("Pdf distribution of 12percentage")
plt.legend(shadow=True,fontsize="larger")

skew = df['12percentage'].skew()
kurt = df['12percentage'].kurt()
print('Skewness:{}'.format(round(skew,2)))
print('Kurtosis:{}'.format(round(kurt,2)))


# - It's having a tail towards left side looks like a negatively skewed distribution.

# In[15]:


# PDF distribution of collegeGPA:
sns.set_style("darkgrid")
plt.figure(figsize=(8,6))
sns.distplot(df['collegeGPA'])
plt.axvline(df['collegeGPA'].mean(), color="black", label="Mean")
plt.title("Pdf distribution of collegeGPA")
plt.legend(shadow=True,fontsize="larger")

skew = df['collegeGPA'].skew()
kurt = df['collegeGPA'].kurt()
print('Skewness:{}'.format(round(skew,2)))
print('Kurtosis:{}'.format(round(kurt,2)))


# - It's having a long tail towards the left looks like highly negativly skwed.

# In[16]:


# Plotting Gender Wise Distribution:
sns.countplot(df['Gender'])


# - We can clearly see that number of males are much more than number of females.

# In[24]:


# Designation oﬀered in the job and it's count:
designations = df['Designation'].unique()
len(designations) 


# In[25]:


designation_count = df['Designation'].value_counts()
designation_count


# In[26]:


sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
designation_count[:15].plot(kind='bar' , width=0.9)
plt.xlabel('Designation')
plt.ylabel('Count')
plt.title('Job Designation of a CSE Fresher')


# - We can see that majority of candidates have chosen Software Engineering.

# In[96]:


# Location of the job (city) and it's count:
jobcities = df['JobCity'].unique()
len(jobcities)


# In[28]:


jobcities_count = df['JobCity'].value_counts()
jobcities_count


# In[29]:


df['JobCity'].replace(-1,np.nan,inplace=True)
df['JobCity'].dropna(inplace=True)


# In[30]:


jobcities_count = df['JobCity'].value_counts()
jobcities_count


# In[31]:


sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
jobcities_count[:10].plot(kind='bar' , width=0.9)
plt.xlabel('Cities')
plt.ylabel('Count')
plt.title('Preferred Cities')


# - Bangalore is the most preferred city among all than Noida, Hyderabad and so on.

# In[32]:


# Degree obtained/pursued by the candidate and it's count:
uniq_degree = df['Degree'].unique()
len(uniq_degree)


# In[33]:


degree_count = df['Degree'].value_counts()
degree_count


# In[34]:


sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
degree_count.plot(kind='bar' , width=0.7 , logy=True)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Degree obtained/pursued by the candidate')


# - We can say that most of the candidates are from B.Tech/B.E.

# In[35]:


# Specialization pursued by the candidate and it's count:
uniq_specialization = df['Specialization'].unique()
len(uniq_specialization)


# In[36]:


specialization_count = df['Specialization'].value_counts()
specialization_count


# In[37]:


sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
specialization_count[:10].plot(kind='bar' , width=0.9)
plt.xlabel('Specialization')
plt.ylabel('Count')
plt.title('Specialization pursued by the candidate')


# - We can see that most choosed specialization is Electronics and communication engineering.

# In[38]:


# College State and it's count:
uniq_collegestate = df['CollegeState'].unique()
len(uniq_collegestate)


# In[39]:


college_state_count = df['CollegeState'].value_counts()
college_state_count


# In[40]:


sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
college_state_count[:15].plot(kind='bar' , width=0.9)
plt.xlabel('College_State')
plt.ylabel('Count')
plt.title('College State Distribution')


# - We can see that Uttar Pradesh is having most of the preferrence colleges.

# In[41]:


# Yearwise Employment groupby Gender:
sns.catplot(x = "GraduationYear",hue="Gender",data = df,kind='count')
plt.xticks(rotation=90)
plt.show()


# - From the above plot we can say that in 2013 joining of employee is higher , in 2017 joining of employee is lower and in all the years joining of male emplyoee is higher then female employees.

# ### Detect the outliers in each numerical column:

# In[42]:


# 10percentage column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['10percentage'])


# In[43]:


# Outliers
df['10percentage'][df['10percentage']<52].reset_index()


# In[44]:


# 12percentage column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['12percentage'])


# In[45]:


# outliers
df['10percentage'][df['10percentage']<44].reset_index()


# In[46]:


# Salary column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['Salary'])


# In[47]:


# collegeGPA column
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['collegeGPA'])


# In[48]:


# Outliers
df['collegeGPA'][(df['collegeGPA']<53) | (df['collegeGPA']>93)].reset_index()


# In[49]:


# Finding the outliers in English column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['English'])


# In[50]:


# Outliers
df['English'][(df['English']<220) | (df['English']>790)].reset_index()


# In[51]:


# Finding the outliers in Logical column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['Logical'])


# In[52]:


# Outliers
df['Logical'][(df['Logical']<260) | (df['Logical']>790)].reset_index()


# In[53]:


# Finding the outliers in Quant column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['Quant'])


# In[54]:


# Outliers
df['Quant'][(df['Quant']<190) | (df['Quant']>850)].reset_index()


# In[55]:


# Finding the outliers in Domain column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['Domain'])


# In[56]:


# Outliers
df['Domain'][df['Domain'] <= -1].value_counts().reset_index()


# In[57]:


# ComputerProgramming
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['ComputerProgramming'])


# In[58]:


# Outliers
df['ComputerProgramming'][df['ComputerProgramming'] > 800].reset_index()


# In[59]:


# ElectronicsAndSemicon column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['ElectronicsAndSemicon'])


# In[60]:


# Outliers
df['ElectronicsAndSemicon'][df['ElectronicsAndSemicon'] > 600].reset_index()


# In[61]:


# ComputerScience column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['ComputerScience'])


# In[62]:


df['ComputerScience'][df['ComputerScience'] > 100].value_counts().reset_index()


# In[63]:


# MechanicalEngg column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['MechanicalEngg'])


# In[65]:


# outliers
df['MechanicalEngg'][df['MechanicalEngg'] > 170].value_counts().reset_index()


# In[66]:


# ElectricalEngg column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['ElectricalEngg'])


# In[67]:


df['ElectricalEngg'][df['ElectricalEngg'] > 200].value_counts().reset_index()


# In[68]:


# TelecomEngg column:
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(df['TelecomEngg'])


# In[69]:


# outliers
df['TelecomEngg'][df['TelecomEngg'] > 150].value_counts().reset_index()


# # Bivariate Analysis:

# In[70]:


#correlation
df.corr()


# In[71]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr() , annot=True)
plt.show()


# # Interpretation:
# - From the above plot we can say that 10percentage and 12percentage is highly positively correlated with each other means if a student got a better marks in 10th standard then it is highly possible that he or she can get the better marks in 12th standard also
# - We can also say that the subjects English,Logical,Quant are also positively correlated with each other means if a student get better marks in one subject then it is highly possible that he or she will get better marks in other subjects also
# - As we can see that Conscientiousness, Agreeableness, Extraversion, and Openess_to_experience are also highly correlated with each other

# In[73]:


plt.figure(figsize=(20,15))
data = df[['Salary','10percentage','12percentage','collegeGPA', 'conscientiousness','agreeableness','extraversion','nueroticism','openess_to_experience']]
sns.pairplot(data)
plt.show()


# In[75]:


# Salary by Specialization
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.boxplot(x=df['Salary'].iloc[:15], y = df['Specialization'].iloc[:15])
plt.suptitle('Salary by specialization')
plt.show()


# ## Research Questions
# - Times of India article dated Jan 18, 2019 states that “After doing your Computer Science
# Engineering if you take up jobs as a Programming Analyst, Software Engineer,
# Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh
# graduate.” Test this claim with the data given to you.
# - Is there a relationship between gender and specialisation? (i.e. Does the preference of
# Specialisation depend on the Gender?)

# In[74]:


# Normalize Salary for Better Visualization
df['n_sal']=df['Salary']/100000


# In[76]:


df[['Designation','Specialization']][df['Designation']=='hardware engineer']


# In[78]:


print('Average Salary :')
print('Programmer Analyst :',round(df['n_sal'][(df['GraduationYear']==2014) & (df['Designation']=='programmer analyst') & (df['Specialization']=='computer science & engineering')].mean(), 2))
print('Software Engineer :',round(df['n_sal'][(df['GraduationYear']==2014) & (df['Designation']=='software engineer')  & (df['Specialization']=='computer science & engineering')].mean(),2))
print('Hardware Engineer :',round(df['n_sal'][(df['GraduationYear']==2014) &(df['Designation']=='hardware engineer')  & (df['Specialization']=='computer science & engineering')].mean(), 2))
print('Associate Engineer :',round(df['n_sal'][(df['GraduationYear']==2014) &(df['Designation']=='associate engineer')  & (df['Specialization']=='computer science & engineering')].mean(), 2))


# In[79]:


# Sample Data for Required Employees
sample = [3.16,3.6,0,3.5]
sample = np.array(sample)


# In[80]:


# Necessary variables initialization ex- sample mean
sample_size = len(sample)
sample_mean = np.mean(sample)
sample_mean


# In[81]:


# Sample Standard Devation
import math
sample_std = math.sqrt(sum([(i-sample_mean)**2 for i in sample]) / 3)
print('Sample Standard Deviation :', sample_std)


# In[82]:


# Calulating T-Score
def t_score(pop_mean, sample_mean, sample_std, sample_size):
    numerator = sample_mean - pop_mean
    denomenator = sample_std / (sample_size**0.5)
    return numerator / denomenator


# In[83]:


# Necessary variables initialization ex- sample mean,population mean
pop_mean = 2.75
sample_mean = 3.34
sample_std = 0.21
sample_size = 4


# In[84]:


# Calling T-score Function
t_sc = t_score(pop_mean, sample_mean, sample_std, sample_size)
print('t-score :', t_sc)


# In[85]:


# Setting the Confidence Level

# Two Tail - Deciding the Significance Level & Calculating the t-critical value
from scipy.stats import t
confidence_level = 0.95
alpha = 1 - confidence_level
t_critical = t.ppf(1-alpha/2, df = 3)
print('t_critical :', t_critical)


# In[86]:


# Visualizing the Sampling Distribution with Rejection Regions
from scipy.stats import norm
# Defining the x min & x max
x_min = 2
x_max =6

# Defining the Sampling Distribution mean & std
mean = pop_mean
std = sample_std / (sample_size**0.5)

# Ploting the graph and setting the x limits
x = np.linspace(x_min, x_max, 100)
y = norm.pdf(x, mean, std)
plt.xlim(x_min, x_max)
plt.plot(x, y)

# Computing the left and right critical values of Two tailed Test
t_critical_left = pop_mean + (-t_critical * std)
t_critical_right = pop_mean + (t_critical * std)

print('t_critical_left :', t_critical_left)
print('t_critical_right :', t_critical_right)

# Shading the left rejection region
x_left = np.linspace(x_min, t_critical_left, 100)
y_left = norm.pdf(x_left, mean, std)
plt.fill_between(x_left, y_left, color='red')

# Shading the right rejection region
x_right = np.linspace(t_critical_right, x_max, 100)
y_right = norm.pdf(x_right, mean, std)
plt.fill_between(x_right, y_right, color='red')

plt.scatter(sample_mean, 0)
plt.annotate("x_bar", (sample_mean, 0.1))


# In[87]:


# Compairing the Table Value and T-score value

# Conclusion using t-test

if np.abs(t_sc) > t_critical:
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# In[88]:


# Conclusion using p-test
p_value = 2 * (1.0 - norm.cdf(np.abs(t_sc)))

print("p_value = ", p_value)

if p_value < alpha:
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# ## Feature Transformation:

# #### Column Standardization for Numerical Features:

# In[89]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[90]:


# Standardizing Salary Column
scaled_sal = scaler.fit_transform(data['Salary'].values.reshape(-1,1))

# First 20 Scaled Values (Salary column)
print(scaled_sal[:20])


# In[91]:


# Standardizing 10th percent Column
scaled_10 = scaler.fit_transform(data['10percentage'].values.reshape(-1,1))

# First 20 Scaled Values (Salary column)
print(scaled_10[:20])


# In[92]:


# Standardizing 12th percent Column
scaled_12 = scaler.fit_transform(data['12percentage'].values.reshape(-1,1))

# First 20 Scaled Values (Salary column)
print(scaled_12[:20])


# #### Column Standardization for Categorical Features:

# In[93]:


# One-hot Encoding of Gender column
dummies = pd.get_dummies(df[['Gender']])
dummies


# In[94]:


df1 = pd.concat([df,dummies],axis='columns')
df1.head()


# In[95]:


finaldf = df1.drop(['Gender','Gender_f'],axis='columns')
finaldf


# # Conclusion
# - In this dataset majority of people earns between 35,000 to 5,00,000 and their are so many outliers in dataset. People who earns mre than 10,00,000 has been removed from the dataset because they are less than 1% of dataset.
# - No. of males in the given dataset is more than doble of No. of females.
# - Male earns more than females but their median is same.
# - Their is positive relation between score and Salary or grades and salary.
# - People with Computer Science background earns the most.
# - After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh graduate. This is a wrong statement because our null hypothesis got rejected.

# In[ ]:




