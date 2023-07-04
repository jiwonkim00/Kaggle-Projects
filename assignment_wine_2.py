#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

wine_df_org = pd.read_csv('wine_train.csv')
wine_df = wine_df_org.copy()
wine_df.head(5)


# In[2]:


wine_df.tail(5)


# In[3]:


#타깃 값의 분포도 확인 (정규 분포인지)

plt.title('Original Points Histogram')
sns.distplot(wine_df['points'])


# In[4]:


#결괏값을 로그 변환하고 다시 분포도 살펴보기

# plt.title('Log Transformed Points Histogram')
# log_Points = np.log1p(wine_df['points'])
# sns.distplot(log_Points)


# In[5]:


wine_df.info()


# In[6]:


#데이터 세트의 전체 크기, 칼럼의 타입, null이 있는 칼럼과 건수를 내림차순으로 출력

print('데이터 세트의 Shape : ', wine_df.shape)
print('\n전체 피처의 type \n', wine_df.dtypes.value_counts())
isnull_series = wine_df.isnull().sum()
print('\nNull 칼럼과 그 건수 : \n', isnull_series[isnull_series > 0].sort_values(ascending=False))


# In[7]:


wine_df['country'].describe()


# In[8]:


wine_df['designation'].describe()


# In[9]:


wine_df['price'].describe()


# In[10]:


wine_df['province'].describe()


# In[11]:


wine_df['region_1'].describe()


# In[12]:


wine_df['region_2'].describe()


# In[13]:


wine_df['variety'].describe()


# In[14]:


wine_df['winery'].describe()


# In[15]:


wine_df['points'].describe()


# # Pipeline 

# In[63]:


from sklearn.model_selection import train_test_split

y = wine_df['points']
X = wine_df.drop(['points', 'taster_twitter_handle', 'id', 'region_2', 'country'], axis=1)


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2)

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 1500 and X_train_full[cname].dtype == "object"]

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# In[64]:


X_train.head()


# In[65]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy = 'constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown= 'ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[29]:


# from sklearn.ensemble import RandomForestRegressor   # MAE :  2.322780699375733

# model = RandomForestRegressor(n_estimators = 100)


# In[71]:


from xgboost import XGBRegressor

model = XGBRegressor(n_estimators = 1000)


#cols = country, province, region_1, region_2, variety, price)
#100 : MAE = 2.2510787042693092
#500 : MAE = 2.219148113567828
#1000 : MAE = MAE :  2.2160245414679562

#cols = country, province, region_1, taster_name, variety, price
#500 : MAE = 2.143475989653365
#1000 : MAE :  2.1421490104321035

#cols = province, region_1, taster_name, variety, price
#1000 : MAE = 2.1408334693768842

#cols = country, province, taster_name, price
#500 : MAE = 2.2651362130224153


# In[72]:


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)

mse = mean_squared_error(y_valid, preds)
score = np.sqrt(mse)

print('MAE : ', score)


# In[1]:


from sklearn.model_selection import GridSearchCV

xgb_params = {'n_estimators' : [1000]}
xgb_reg = XGBRegressor(n_estimators=1000, learning_rate = 0.05, colsample_bytree = 0.5, subsample=0.8)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', xgb_reg)])

def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params,
                             scoring='neg_mean_squared_error', cv=5)
    grid_model.fit(X_train, y_train)
    rmse = np.sqrt(-1 * grid_model.best_score_)
    print('{0} 5 CV 시 최적 평균 RMSE 값:{1}, 최적 alpha:{2}'.format(model.__class__.__name__,
                                                            np.round(rmse,4), grid_model.best_params_))

print_best_params(pipeline, xgb_params)


# In[ ]:





# # Cross Validation

# In[73]:


from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring = 'neg_root_mean_squared_error')
print("MAE scores: \n", scores)
print("Average MAE score :")
print(scores.mean())


# In[ ]:





# # Preprocessing

# In[94]:


#points -> log points, null많은 칼럼 drop, null 값 대체

y = wine_df['points']
wine_df = wine_df[['province', 'region_1', 'taster_name', 'variety', 'price']]


# X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2)


# wine_df.drop(['taster_name', 'taster_twitter_handle', 'title', 'description', 'id'], axis=1, inplace=True)
# wine_df.fillna(wine_df.mean(), inplace=True)
# original_Points = wine_df['points']
# wine_df['points'] = np.log1p(wine_df['points'])


# In[95]:


wine_df.isna().sum()

# OR
# null_column_count = wine_df.isnull().sum()[wine_df.isnull().sum() > 0]
# print('## Null Feature Type : \n',wine_df.dtypes[null_column_count.index])


# In[96]:


wine_df.head(5)


# In[97]:


feature_columns_categorical = ['province', 'region_1', 'taster_name', 'variety']
wine_df[feature_columns_categorical] = wine_df[feature_columns_categorical].fillna('None')


# In[98]:


wine_df.head(5)


# In[99]:


wine_df['price'] = wine_df['price'].fillna(wine_df['price'].mean())


# In[100]:


wine_df.head(10)


# In[101]:


wine_df.isna().sum()   # no null values ~~


# In[14]:


# One-Hot Encoding

# print('get_dummies() 수행 전 데이터 Shape : ', wine_df.shape)
# wine_df_ohe = pd.get_dummies(wine_df)
# print('get_dummies() 수행 후 데이터 Shape : ', wine_df_ohe.shape)


# In[102]:


print('get_dummies() 수행 전 데이터 Shape : ', X.shape)
wine_df_ohe = pd.get_dummies(X, columns = ['province', 'region_1', 'taster_name', 'variety', 'price'])
print('get_dummies() 수행 후 데이터 Shape : ', wine_df_ohe.shape)


# # Functions

# In[92]:


def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print(model.__class__.__name__, '로그 변환된 RMSE:', np.round(rmse, 3))
    return rmse

def get_rmses(models):
    rmses = []
    for model in models :
        rmse = get_rmse(model)
        rmses.append(rmse)
        return rmses
    


# In[108]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# y = wine_df_ohe['points']
# X = wine_df_ohe.drop('points', axis=1, inplace=False)
X_train, X_valid, y_train, y_valid = train_test_split(wine_df_ohe, y, test_size=0.2)


# In[106]:


# lr_reg = LinearRegression()
# lr_reg.fit(X_train, y_train)
# ridge_reg = Ridge()
# ridge_reg.fit(X_train, y_train)
# lasso_reg = Lasso()
# lasso_reg.fit(X_train, y_train)

# models = [ridge_reg, lasso_reg]
# get_rmses(models)


# In[110]:


# xgb_params = {'n_estimators':[1000]}
# xgb_reg = XGBRegressor(n_estimators = 1000, learning_rate = 0.05, 
#                       colsample_bytree = 0.5, subsample = 0.8)
# print_best_params(xgb_reg, xgb_params)


# In[ ]:




