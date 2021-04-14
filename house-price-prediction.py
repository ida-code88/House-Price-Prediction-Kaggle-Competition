#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)

# In[54]:


# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import math
from scipy.stats import norm, skew


# In[55]:


# Load the data
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[56]:


print("The train data has Rows {}, Columns {}".format(train_df.shape[0], train_df.shape[1]))
print("The test data has Rows {}, Columns {}".format(test_df.shape[0], test_df.shape[1]))


# Let's take a look into our training and testing dataset

# In[57]:


train_df.head()


# In[58]:


test_df.head()


# In[59]:


train_df.info()


# From the list above, features like MiscFeature, FireplaceQu, Fence, PoolQC, and Alley has excessive amount of null values. Let's see their description.
# 
# * Alley: Type of alley access to property
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories
# * FireplaceQu: Fireplace quality
# 
# We can see that not only they have so many null, but also they don't look important for our analysis. So I assume it is safe if we want to drop them. For now I am not going to drop them yet.

# In[60]:


to_drop = train_df[['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']]
to_drop


# ### Understanding the variables
# 
# ### First is the target feature: SalePrice

# In[61]:


train_df['SalePrice'].describe()


# There is a high jump in the sales price from min to the first quartile and from 75% to max. Let's see the distribution of it to find out whether it has skewness or not.

# In[62]:


plt.figure(figsize=(8,6))
sns.distplot(train_df['SalePrice'], color='y');


# From the graph above, we can conclude that the sales price distribution is skewed right (it doesn't follow normal distribution). I will convert it into normal distribution later.
# 
# ### Second: quantitative variables

# In[63]:


num_feat = train_df.select_dtypes(include = ['float64', 'int64'])
num_feat.head()                             


# In[64]:


num_feat = num_feat.drop(['Id'], axis=1)


# In[65]:


print(num_feat.shape)


# #### Relation between sales price and numerical features:

# In[66]:


#correlation matrix
num_corr = num_feat.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(num_corr, vmax=.8, square=True);


# Let's see which numerical features whose strong correlation with sales price

# In[67]:


# saleprice correlation matrix
k = 11 #number of variables for heatmap
cols = num_corr.nlargest(k, 'SalePrice')['SalePrice'].index   # nlargest : pick the most powerfull correlation
cm = np.corrcoef(train_df[cols].values.T)
f, ax = plt.subplots(figsize=(20, 9))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[68]:


num_corrlist = num_feat.corr()['SalePrice'][:-1]
numeric_list = num_corrlist[abs(num_corrlist) > 0.5].sort_values(ascending=False)
numeric_list


# Let examine each of them:
# * OverallQual: Overall Quality
# * GrLivArea: Above grade (ground) living area square feet
# * GarageCars: Size of garage in car capacity 
# * GarageArea: Size of garage in square feet
# * TotalBsmtSF: Total square feet of basement area
# * 1stFlrSF: First Floor square feet
# * FullBath: Full bathrooms above grade
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
# 
# GragaeCars and GarageArea not only they have the same meaning, but they also have similar correlation number (almost). Hence, we can choose one and it will be GarageCars due to its higher correlation number.
# 
# TotalBsmtSF and 1stFlrSF both have a high correlation to each other. We can keep the former because of the higher correlation to the sales price.
# 
# GrLivArea and TotRmsAbvGrd have the same case as well, so TotRmsAbvGrd can be ignored.
# 
# The rest we can keep.

# In[69]:


columns = ['SalePrice', 'OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd']
num_feat = num_feat[columns]

sns.set()
sns.pairplot(num_feat, height = 3)
plt.show();


# Let see closer for some features

# In[70]:


# OverallQual, GrLivArea, SalePrice
sns.scatterplot(data=num_feat, x='GrLivArea', y='SalePrice', hue = 'OverallQual');


# So we can see that the bigger area above ground and the better overall quality of the house, the higher sales price will be.

# In[71]:


# YearBuilt, OverallQual, SalePrice
sns.scatterplot(data=num_feat, x='YearBuilt', y='SalePrice', hue = 'OverallQual');


# Newer houses tend to have a better quality and more expensive price

# In[72]:


# GarageCars, FullBath, SalePrice
sns.scatterplot(data=num_feat, x='GarageCars', y='SalePrice', hue = 'FullBath');


# Fore the same size of cars garage, the more full bathrooms it has, the sales price will increase as expected.

# ### Third: Qualitative Variables

# In[73]:


cat_feat = train_df.select_dtypes(['object'])
cat_feat.head()   


# In[74]:


# Their distribution
fig, axes = plt.subplots(15, 3, figsize=(20, 40))

for i, ax in enumerate(fig.axes):
    if i < len(cat_feat.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=cat_feat.columns[i], alpha=0.7, data=cat_feat, ax=ax)

fig.tight_layout()


# In[75]:


cat_feat_price = pd.concat([cat_feat, num_feat['SalePrice']], axis=1)

fig, axes = plt.subplots(22, 2, figsize=(20, 70))

for i, ax in enumerate(fig.axes):
    if i < len(cat_feat_price.columns)-1:
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.boxplot(data = cat_feat_price, x=cat_feat_price.columns[i], y='SalePrice', ax = ax)

fig.tight_layout()


# # Data Cleaning: 
# 
# ### Handling Missing Values

# In[76]:


all_data = pd.concat([train_df, test_df], axis=0, sort=False)
all_data = all_data.drop(['Id', 'SalePrice'], axis=1)
all_data


# It's time to handle missing data

# In[77]:


null_val = pd.DataFrame(all_data.isna().sum(), columns = ['Nan_sum'])
null_val = null_val[null_val['Nan_sum']>0]
null_val['Percentage'] = (null_val['Nan_sum']/len(all_data))*100
null_val = null_val.sort_values(by=['Nan_sum'], ascending=False)
null_val


# If we look at what those features above reprsent, we will know that none of them are considered important for sales price analysis. Addition to their large number of missing values, it will be better to drop them.

# In[78]:


# drop the missing data
all_data = all_data.drop((null_val[null_val['Nan_sum']>5]).index,1)


# In[79]:


num_null = ['GarageArea', 'GarageCars', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']
cat_null = ['Functional', 'Utilities', 'Electrical', 'KitchenQual', 'MSZoning', 'Exterior2nd', 'Exterior1st', 'SaleType']

for x in num_null:
    all_data[x] = all_data[x].fillna(0)
    
for x in cat_null:
    all_data[x] = all_data[x].fillna(all_data[x].mode()[0])


# In[80]:


# check null values after treatment
all_data.isnull().sum().max()


# ### Dealing with skewness

# In[81]:


# code source: https://www.kaggle.com/adamml/how-to-be-in-top-10-for-beginner
non_object_feat = all_data.dtypes[all_data.dtypes != 'object']. index
skewed_feat = all_data[non_object_feat].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feat[abs(skewed_feat)>0.5]
high_skew


# In[82]:


# change skewed distribution into normal distribution
for x in high_skew.index:
    all_data[x] = np.log1p(all_data[x])


# In[83]:


# for sales prediction
train2_df = train_df.copy()
train2_df['SalePrice'] = np.log1p(train2_df['SalePrice'])
y_log = train2_df['SalePrice']

plt.figure(figsize=(8,6))
sns.distplot(train2_df['SalePrice'], color='y');


# In[84]:


# if we compare it to distribution of sales price earlier (below), it shows that the distribution of sales price is now normal
plt.figure(figsize=(8,6))
sns.distplot(train_df['SalePrice'], color='y');


# ### Dummy for categorical variables

# In[85]:


all_data = pd.get_dummies(all_data)
all_data.head()


# # Apply ML

# In[86]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# In[87]:


# separate data for training and testing
X = all_data[:len(y_log)]
X_test_data = all_data[len(y_log):]


# In[88]:


X.shape, X_test_data.shape


# In[89]:


# Creation of the RMSE metric:
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, X, y_log, scoring = 'neg_mean_squared_error', cv = kf))
    return (rmse)


# In[90]:


# K Fold Cross validation
kf = KFold(n_splits = 7, random_state = 234, shuffle = True)

cv_scores = []
cv_std = []

baseline_models = ['Linear_reg', 'Random_Forest_reg', 'Grad_Boost_reg', 'XGB_reg']


# In[91]:


# Linear Regression

lr = LinearRegression()
score_lr = cv_rmse(lr)
cv_scores.append(score_lr.mean())
cv_std.append(score_lr.std())


# Random Forest Regressor

rf = RandomForestRegressor()
score_rf = cv_rmse(rf)
cv_scores.append(score_rf.mean())
cv_std.append(score_rf.std())


# Gradient Boost Regressor

gb = GradientBoostingRegressor()
score_gb = cv_rmse(gb)
cv_scores.append(score_gb.mean())
cv_std.append(score_gb.std())


# XGB Regressor

xgb = XGBRegressor()
score_xgb = cv_rmse(xgb)
cv_scores.append(score_xgb.mean())
cv_std.append(score_xgb.std())


# In[92]:


cv_score_results = pd.DataFrame(baseline_models, columns = ['Regressors'])
cv_score_results['RMSE_mean'] = cv_scores
cv_score_results['RMSE_std'] = cv_std


# In[93]:


cv_score_results


# In[94]:


# visualization of cv_score_results
plt.figure(figsize = (12,8))
sns.barplot(cv_score_results['Regressors'],cv_score_results['RMSE_mean'])
plt.xlabel('Regressors', fontsize = 12)
plt.ylabel('CV_Mean_RMSE', fontsize = 12)
plt.xticks(rotation=40)
plt.show()


# In[95]:


# Linear Regression
linear_model = lr.fit(X,y_log)
test_lr = lr.predict(X_test_data)
result = pd.DataFrame(test_df, columns = ['Id'])
test_pre = np.expm1(test_lr)
result['SalePrice'] = test_pre

result.to_csv("linear_result.csv", index = False, header = True)


# In[96]:


# Random Forest
forest_model = rf.fit(X,y_log)
test_rf = rf.predict(X_test_data)
result = pd.DataFrame(test_df, columns = ['Id'])
test_pre = np.expm1(test_rf)
result['SalePrice'] = test_pre

result.to_csv("rforest_result.csv", index = False, header = True)


# In[97]:


# XGB
xgb_model = xgb.fit(X,y_log)
test_xgb = xgb.predict(X_test_data)
result = pd.DataFrame(test_df, columns = ['Id'])
test_pre = np.expm1(test_xgb)
result['SalePrice'] = test_pre

result.to_csv("xgb_result.csv", index = False, header = True)


# In[98]:


# Gradient Boosting
gb_model = gb.fit(X,y_log)
test_gb = gb.predict(X_test_data)
result = pd.DataFrame(test_df, columns = ['Id'])
test_pre = np.expm1(test_gb)
result['SalePrice'] = test_pre

result.to_csv("gradient_result.csv", index = False, header = True)


# Further analysis can be done if we feel courious about it. We can drop the features we think are not significant to sales prediction, combine features that have similar purpose, or using only the important variables in building our regression models.
# 
# I might be do some mistakes here and there in this project. Feel free to tell me or comment on anything you think necessary.
