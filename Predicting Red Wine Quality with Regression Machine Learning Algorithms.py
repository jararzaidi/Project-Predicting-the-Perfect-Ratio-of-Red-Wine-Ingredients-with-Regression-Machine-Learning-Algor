#!/usr/bin/env python
# coding: utf-8

# # Project: Predicting Red Wine Quality with Regression Machine Learning Algorithms
# 

# Table of Contents
# 
# 1. Introduction:
#     Scenario
#     Goal
#     Features & Predictor
# 
# 2. Data Wrangling:
#     Missing Values,
#     Detecting/Handling Outliers with a Z-Score
#     
# 3. Exploratory Data Analysis:
#     Correlations
#     Pairplots
#     Filtering data by Tasty & Non-Tasty 
#     Kernel Density Estimation (KDE)
#     Regression Joint Plot
#     Comparing Tasty & Non-Tasty Red Wine¶
#     
# 
# 4. Machine Learning + Predictive Analytics:
#     Prepare Data for Modeling
#     Modeling/Training
#     Predictions
#     K-Fold Cross Validation
#   
# 5. Conclusions
# 
# *** Note include link to kaggle data
# https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009?select=winequality-red.csv
# 

# # 1. Introduction

# # Scenario:

# You walk into a Pub in northern Paris, France. Where you order vodka and mention to the person sitting next to you that your a Data Scientist. The owner of the Pub, who happens to be a vintner lets you know that he is hiring a Data Scientist. He says he is loosing customers & needs help coming up with a new formula for his red wine collection. The vintner hands you this data to peform Data Analysis and predict the perfect ratio of ingredients to maximize his profit. 

# # Goal:
# 

# - Predict the perfect ratio of ingredients for red wine. This is a numerical discrete outcome. 
# 
# - Explore with various Regression Models & see which yields  greatest accuracy. 
# - Examine trends & correlations within our data
# - Determine which features are important in determing the quality of red wine
# 
# Note: Due to the fact that we are predicting a numerical discreet value, we will be training various Regression Models 

# # Features & Predictor:

# Our Predictor (Y, Wine Quality) is determined by 11 Features (X):
# 
#     1.Fixed acidity - most acids involved with wine or fixed or nonvolatile (do not evaporate readily)
#     2.Volatile acidity - amount of acetic acid ( which at too high of levels can lead to an unpleasant, vinegar taste)
#     3.Citric acid - found in small quantities, citric acid can add 'freshness' and flavor to wines
#     4.Residual sugar - amount of sugar remaining after fermentation stops
#     5.Chlorides - amount of salt 
#     6.Free sulfur dioxide - free form SO2 exists equilibrium between molecular SO2 (dissolved gas) and bisulfite ion
#     7.Total sulfur dioxide - amount of free and bound forms of S02
#     8.Density - density of water is close to that of water depending on the percent alcohol and sugar content
#     9.pH - describes how acidic or basic a wine is on a scale from 0(very acidic) to 14(very basic); most wines are between 3-4
#     10.Sulfates - additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial 
#     11.Alcohol - percent alcohol content 

# Note: Our data has only 1 type of data:
#         Continuous (#): which is quantitative data that can be measured
#       

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt


# # 2. Data Wrangling

# In[2]:


filePath = '/Users/jarar_zaidi/Downloads/wineQuality.csv'

data = pd.read_csv(filePath)

data.head()


# In[3]:


print("(Rows, columns): " + str(data.shape))
data.columns 


# In[4]:


data.nunique(axis=0)# returns the number of unique values for each variable.


# In[5]:


#summarizes the count, mean, standard deviation, min, and max for numeric variables.
data.describe()


# The mean quality was 5.6, with its Max (best quality score) being 8.0 & its Min (worst quality score) being 3.0. Now lets see if we have any missing values we need to take care of. 
# 

# # Missing Values

# In[6]:


# Display the Missing Values

print(data.isna().sum())


# Display the Number of Missing Values for each column. We luckily have none.

# # Detecting/Handling Outliers with Z-Score

# A Z-Score is a measure of position that indicates the number of standard deviations a data value lies from the mean.
# Any z-score less than -3 or greater than 3, is an outlier.
# 
# Note: From the empirical rule we see that 99.7% of our data should be within three standard deviations from the mean.
# 

# In[7]:


from scipy import stats
import numpy as np
z = np.abs(stats.zscore(data))
print(z)


# In[8]:


threshold = 3
print(np.where(z > 3))


# The first array is the list of row numbers & the 2nd array is the corresponding column number of the outlier. For example, the first outlier is in row 13, column 9. Once we calclulated the Z-score, we can remove the outlier to clean our data, by peforming the action below. 

# In[9]:


Newdata = data[(z < 3).all(axis=1)]
Newdata


# We have now succesffully removed 148+ rows which were outliers!
# 
# Note: Other outlier tools could have been used like IQR Score, Scatter Plots, Box Plots.
# 
# Lets take a look at our new cleaned up data

# In[10]:


Newdata.describe()


# In[ ]:





# # 3. Exploratory Data Analysis

# # Correlations

# Correlation Matrix aka Heat Maps- let’s you see correlations between all variables.
# Within seconds, you can see whether something is positively or negatively correlated with our predictor (target).

# In[11]:


# calculate correlation matrix

corr = Newdata.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
sns.heatmap(corr, xticklabels=corr.columns,
            yticklabels=corr.columns, 
            annot=True,
            cmap=sns.diverging_palette(220, 20, as_cmap=True))


# We can see there is a strong positive correlation between alcohol & target (our predictor). In fact, this is our most correlated feature in our datast, with a value of 0.5! Note that our alcohol feature was the percent alcohol content in a drink. This makes sense that a higher percent of alcohol content would yield a greater satsification for a customer purchasing red wine!
# 
# Next, we can see the second strongest positive correlation, 0.39, between sulphates & our quality predictor. It seems that people rate the quality higher when an additive (S02) is contributed to the drink. Sulphates acts as an antimicrobial.
# 
# Lastly, the strongest negative correlation is the volatile acidity, with a correlation of -0.35! This is as expected because too high acetic acid levels can lead to an unpleasant, vinegar taste!
# 

# # Pairplots

# Pairplots are also a great way to immediately see the correlations between all variables. But you will see me make it with only continuous columns from our data, because with so many features, it can be difficult to see each one. So instead I will make a pairplot with only our continuous features.
# 

# Because we have 11 Features, lets select only significant features that correlate to our predictor, to further examine their correlation on a pairplot

# In[12]:


subData = data[['volatile acidity','citric acid','sulphates','alcohol']]
sns.pairplot(subData)


# Chose to make a smaller pairplot with only the the strongest predictors, to dive deeper into the relationships. Also a great way to see if theirs a positive or negative correlation!
# Note: This plot supports the correlations stated in the Heat Map.
# 

# # Filtering data by Tasty & Non-Tasty 

# We will now do a form of Feature Engineering, where we create a new column classifiying if the  the wine quality is tasty or not based on its quality score!
# 
# This new column will be a binary Categorical Data where 0 or 1 indicates if the wine was considered "tasty".

# In[13]:


Newdata['tasty'] = [0 if x < 6 else 1 for x in Newdata['quality']]


# In[14]:


# DONT INCLUDE THIS IN ARTICLE!!!!!!!!!!
#Newdata.drop(columns=['goodquality']).head()


# # Kernel Density Estimation (KDE)

# A Kernel Density Estimation (KDE) estimate the probability density function (PDF) of a continuous random variable.

# In[15]:


g = sns.jointplot("quality", "volatile acidity", data=Newdata,
           kind="kde", space=0, color="red")


# A Kernel Density Estimation allows us to visualize the distribution of data over a continuous interval. From this plot, we can conclude that lower quality Wine is heavily inclined to higher levels of volatile acidity. This is as we expected because, large quantities of acetic acid yield an unpleasant vinegar taste!

# # Regression Joint Plot

# In[16]:


g = sns.jointplot(x= 'fixed acidity',y= 'pH', data=Newdata,
             kind = 'reg',height=15,color='blue')

plt.xlabel('Fixed acidity',size=30)
plt.ylabel('pH',size=40)
plt.title('Fixed acidity vs. pH',size=30)


# From interpreting the Regression Join plot above, we can see a STRONG Negative correlation between pH level & fixed acidity. According to the HeatMap, these two features have a correlation coefficent of -0.71! 

# # Comparing Tasty & Non-Tasty Red Wine

# In[17]:


sns.catplot(x="quality", y="volatile acidity", hue="tasty", kind="bar", data=Newdata);

plt.title('Tasty & Non-Tasty Wine with Volatile Acidity Level',size=19)
plt.xlabel('Wine Quality',size=16)
plt.ylabel('Volatile Acidity Level',size=16)


# This plot illustrates that high volatile acidity levels yields a terrible tasting wine. This is as we expected because, large quantities of acetic acid yield an unpleasant vinegar taste!

# The advantages of showing the Box & Violin plots is that it shows the basic statistics of the data, as well as its distribution. These plots are often used to compare the distribution of a given variable across some categories.
# It shows the median, IQR, & Tukey’s fence. (minimum, first quartile (Q1), median, third quartile (Q3), and maximum).
# In addition it can provide us with outliers in our data.

# In[18]:



plt.figure(figsize=(12,8))
sns.boxplot(x="quality", y="sulphates", hue="tasty", data=Newdata )
plt.title("Tasty & Non-Tasty Wine with Sulphate Level", fontsize=20)
plt.xlabel("Wine Quality Level",fontsize=16)
plt.ylabel("Sulphate Level", fontsize=16)


# Next, we examine this boxplot which helps further conclude that tasty red wine exhibit a heightened median for Sulphate Levels. If we recall back from our chemistry lessons, we remember that sulfates act as a additive which can contribute to sulfur dioxide gas (S02) levels, which is an antimicrobial!
# 
# Note: A antimicrobial is an agent that kills microorganisms & stops their growth. 
# 
# We can see now why these high sulfate levels would enhance a customers preference!
# 

# In[19]:


plt.figure(figsize=(12,8))
sns.violinplot(x="quality", y="alcohol", hue="tasty", inner='quartile',data= Newdata )
plt.title("Tasty & Non-Tasty Wine with Percent alcohol content",fontsize=20)
plt.xlabel("Wine Quality Level", fontsize=16)
plt.ylabel("Percent alcohol content ", fontsize=16)


# After analyzing this violin plot we can conclude that the overall shape & distribution for tasty & non-tasty wine differ vastly. Tasty red wine exhibit a surplus median for percent alcohol content & thus a great distribution of their data is between 10 & 13,  while non-tasty red wine consists of a lower median Alcohol level content between 9.5 & 11.  

# In[20]:


yummy = Newdata[Newdata['tasty']==1]
yummy.describe()


# In[21]:


notYummy = Newdata[Newdata['tasty']==0]
notYummy.describe()


# In[22]:


print("(Tasty Wine Sulphates level): " + str(yummy['sulphates'].mean()))
print("(Non-Tasty Wine Sulphates level): " + str(notYummy['sulphates'].mean()))


# In[23]:


print("(Tasty Wine Alcohol content level): " + str(yummy['alcohol'].mean()))
print("(Non-Tasty Wine Alcohol content level): " + str(notYummy['alcohol'].mean()))


# In[24]:


print("(Tasty Wine Total Sulfur Dioxide level): " + str(yummy['total sulfur dioxide'].mean()))
print("(Non-Tasty Wine Total Sulfur Dioxide level): " + str(notYummy['total sulfur dioxide'].mean()))


# Lastly, the averages between the tasty & non - tasty wine features differ vastly. For example, Tasty Red Wine consists of minimized levels of total sulfur dioxide. In addition, Tasty Red Wine consists of sulphate & alcohol levels that are substantially higher. 
# 

# 
# If I was to tell the vinter the perfect ratio for each ingredient to maximize his Red Wine sales, I would tell him to include low levels of sulfur dioxide, and high levels of sulphates & alcohol.

# # 4. Machine Learning + Predictive Analytics
# 

# # Prepare Data for Modeling

# To prepare data for modeling, just remember ASN (Assign,Split, Normalize).

# To swap the the indexes of last 2 columns:

# In[63]:


df = Newdata[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol','tasty','quality']]


# In[64]:


df


# Assign the 11 features to X, & the last column to our continuous  predictor, y
# 

# In[84]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# Split: the data set into the Training set and Test set

# In[85]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)


# Normalize: Standardizing the data will transform the data so that its distribution will have a mean of 0 and a standard deviation of 1.

# In[86]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# # Modeling /Training
# 

# Now we’ll Train various Regression Models on the Training set & see which yields the highest accuracy. We will compare the accuracy of Multiple Linear Regression, Polynomial Linear Regression, SVR (Support Vector Regression), Decision Tree Regression, Random Forest Regression, and XGBoost.These are all supervised learning models, for predicting continous values. 
# 
# Note: There are a few metrics for measuring accuracy for a regression model like Root Mean Squared Error (RMSE), Residual Standard Error (RSE), and Mean Absolute Error (MAE). But we will be measuring our models with R-squared. 

#  Model 1: Multiple Linear Regression

# In[185]:


# Train model on whole dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting Test Set Results
y_pred = regressor.predict(x_test)

# Evaluating Model Performance
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# Model 2: Polynomial Linear Regression

# #assign
# X2 = df.iloc[:, :-1].values
# y2 = df.iloc[:, -1].values
# 
# #split
# from sklearn.model_selection import train_test_split
# x_train2, x_test2, y_train2, y_test2 = train_test_split(X2,y2,test_size = 0.2, random_state = 1)
# 
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# poly_reg = PolynomialFeatures(degree = 4)
# x_poly = poly_reg.fit_transform(x_train2)
# lin_reg_2 = LinearRegression()
# lin_reg_2.fit(x_poly,y)
# 
# y_pred2 = lin_reg_2.predict(poly_reg.transform(x_test2))
# 
# from sklearn.metrics import r2_score
# r2_score(y_test2,y_pred2)

# Model 3: SVR (Support Vector Regression)

# In[187]:


#assign
X3 = df.iloc[:, :-1].values
y3 = df.iloc[:, -1].values

#split
from sklearn.model_selection import train_test_split
x_train3, x_test3, y_train3, y_test3 = train_test_split(X3,y3,test_size = 0.2, random_state = 1)

from sklearn.svm import SVR
regressor3 = SVR(kernel='rbf')
regressor3.fit(x_train3,y_train3) # replace by x_train , y_train if we split

regressor3.predict(x_test3)

from sklearn.metrics import r2_score
r2_score(y_test3,y_pred3)


# Model 4: Decision Tree Regression
# 

# In[188]:


# Assign
X4 = df.iloc[:, :-1].values
y4 = df.iloc[:, -1].values

# Split
from sklearn.model_selection import train_test_split
x_train4, x_test4, y_train4, y_test4 = train_test_split(X4,y4,test_size = 0.2, random_state = 4)


from sklearn.tree import DecisionTreeRegressor
regressor4 = DecisionTreeRegressor(random_state = 0)
regressor4.fit(x_train4,y_train4) # replace by x_train , y_train if we split

y_pred4 = regressor4.predict(x_test4)

from sklearn.metrics import r2_score
r2_score(y_test4,y_pred4)


# Model 5: Random Forest Regression

# In[201]:


# Assign
X5 = df.iloc[:, :-1].values
y5 = df.iloc[:, -1].values

# Split
from sklearn.model_selection import train_test_split
x_train5, x_test5, y_train5, y_test5 = train_test_split(X5,y5,test_size = 0.2, random_state = 6)


from sklearn.ensemble import RandomForestRegressor
regressor5 = RandomForestRegressor(n_estimators = 10, random_state=0)
regressor5.fit(x_train5,y_train5) # replace by x_train , y_train if we split

y_pred5= regressor5.predict(x_test5)

from sklearn.metrics import r2_score
r2_score(y_test5,y_pred5)


# Model 6: XGboost
# 

# In[199]:


from xgboost import XGBClassifier

model7 = XGBClassifier(random_state=1)
model7.fit(x_train, y_train)
y_pred7 = model7.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test5,y_pred7)


# Intepreting R^2 Values

# R^2 is a statistical measure of how close the data are to the fitted regression line.
# Higher R squared, the better. The best R^2 value is 1.0
# 
# From examining our regression models, we can conclude that Model 5: Random Forest Regression yields the highest accuracy, with an accuracy of 83%!

# In[244]:


dataa = df.drop(['quality','tasty'], axis=1)
# DONT include


# In[256]:


df


# # Predictions

# Scenario: Lets predict the quality of a red wine a company makes based on its ratio of each ingredient. We will now input the ratio of each ingredient into our Machine Learning Algorithm.

# In[276]:


print(regressor5.predict(([[7.9,0.59,0.004,1.9,0.062,49.0,33.0,0.99513,3.21,0.53,8.9,1]])))


# The red wine ingredients consisting of...
# 
# 7.9 of Fixed Acidity, 
# 
# 0.59 of Volatile Acidity,
# 
# 0.004 of Citric Acid,
# 
# 1.9 of Residual Sugar after fermentation stops,
# 
# 0.062 of Chlorides Salt, 
# 
# 49.0 of Free Sulfur Dioxide,
# 
# 33.0 of Total Sulfur Dioxide,
# 
# 0.9915 Density of water,
# 
# pH of 3.21 (acidic), 
# 
# 0.53 grams of sulfates,
# 
# and 8.9 percent of alcohol content
# 
# yields a Red Wine Quality Score of 6.5!

# # K-Fold Cross Validation

# K-Fold Cross Validation is a statistical method that ensures us to have a better measure of our model pefromance. We run our modeling process on different subsets of the data to get multiple measures of model quality. We divide our data by a specific amount of "folds". 
# 
# K-Fold Cross Validation allows every observation from the original data set to appear in our train & test set.
# 
# When we create 20 different Test folds, we reduce risk of getting lucky. Final accuracy we get will be the average of the 20 test folds!

# In[284]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor5, X = x_train5, y = y_train5, cv = 20)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100)) # float w 2 decimals after comma
print("Accuracy: {:.2f} %".format(accuracies.std()*100)) # float w 2 decimals after comma
# the 20 accruaceis lie within  % , we have a high std. 
#(Mean - Std, Mean + Std) 


# The 20 accruaceis lie within (80.44-4.48,80.44+4.48)% = (75.96,84.92)% Confidence Interval. We have a high standard deviation, which means that our numbers are spread out.

# # Conclusions

# 1. Out of the 11 features we examined, the top 3 significant features that help the vinter brew a delicious Red Wine are low levels of sulfur dioxide, and high levels of sulphates & alcohol.
# 
# 2. Our Random Forest algorithm yields the highest R2 value, 83%! Any R2 above 70% is considered good, but be careful because if your accuracy is extremly high, it may be too good to be true (an example of Overfitting). Thus, 83% is the ideal accuracy!
# 
# 3. Our machine learning algorithm can now predict the red wine quality given it's ingredients. By detecting these important features, we may prevent our vintner from going out of business or loosing any profit! This is extremly powerful because now we can properly see which components of wine people prefer & thus maximize our profit!

# Here is access to the data set & code from my GitHub page:
# https://github.com/jzaidi143/Project-Predicting-Heart-Disease-with-Classification-Machine-Learning-Algorithms
# Recommendations & comments are welcomed!

# In[ ]:




