#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error


# In[3]:


wine_df_org = pd.read_csv('wine_train.csv')
wine_df = wine_df_org.copy()
wine_df.head(5)


# In[4]:


wine_test = pd.read_csv('wine_test.csv')


# In[4]:


#타깃 값의 분포도 확인 (정규 분포인지)

plt.title('Original Points Histogram')
sns.distplot(wine_df['points'])


# In[10]:


#p.307

# fig, axs = plt.subplots(figsize=(16,8), ncols=2, nrows=1)
# lm_features = ['price', 'description_length']
# for i, feature in enumerate(lm_features):
#     row = int(i/1)
#     col = i%2
#     sns.regplot(x=feature, y='points', data=wine_df, ax=axs[row][col])


# In[5]:


wine_df.info()


# In[6]:


#데이터 세트의 전체 크기, 칼럼의 타입, null이 있는 칼럼과 건수를 내림차순으로 출력

print('데이터 세트의 Shape : ', wine_df.shape)
print('\n전체 피처의 type \n', wine_df.dtypes.value_counts())
isnull_series = wine_df.isnull().sum()
print('\nNull 칼럼과 그 건수 : \n', isnull_series[isnull_series > 0].sort_values(ascending=False))


# # Preprocessing

# In[19]:


y = wine_df['points']
X = wine_df.drop(['points'], axis=1)
X['description_length'] = X.description.str.len()
wine_df['description_length'] = wine_df.description.str.len()
X = X.drop(['taster_twitter_handle', 'id', 'region_2' 'country', 'description', 'winery'], axis=1)
X.head()

#drop
#'taster_twitter_handle', 'id', 'region_2', 'country', 'description' : MAE =  1.847030179941818
# + 'province' : MAE = 1.8720274596941904
# + 'title' : MAE = 1.8645017663312122
# + 'winery' : MAE = MAE :  1.8402934788424206

# + 'winery' - 'region_2' : MAE = 1.8496255971459103


# In[20]:


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2)

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 1500 and X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

numerical_transformer = SimpleImputer(strategy = 'constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train.head()


# # XGBoost

# In[42]:


model = XGBRegressor(n_estimators = 1000, learning_rate = 0.2)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)

mse = mean_squared_error(y_valid, preds)
score = np.sqrt(mse)

print('MAE : ', score)

#n_estimators = 1000, max_depth = 9, min_child_weight = 1 : MAE :  1.8597984884050986

#Learning_rate
#0.3 : 1.8471743226988189
#0.25 : 1.847030179941818
#0.2 : 1.8447899853492296
#0.15 : 1.8489612992889137
#0.1 : 1.8551066040566266


# # Linear Regression

# In[18]:


lr = LinearRegression()
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                             ('model', lr)])
my_pipeline.fit(X_train, y_train)
y_preds = my_pipeline.predict(X_valid)
mse = mean_squared_error(y_valid, y_preds)
rmse = np.sqrt(mse)

print('MSE : {0:.3f}, RMSE : {1: .3f}'.format(mse, rmse))
print('Variance score : {0:.3f}'.format(r2_score(y_test, y_preds)))

print('절편 값 : ', lr.intercept_)
print('회귀 계수값 : ', np.round(lr.coef_, 1))

#linear regression
#RMSE : 2.030


# # LightGBM

# In[38]:


from lightgbm import LGBMRegressor

lgbm_params = {'n_estimators' : [1000]}
lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.2, num_leaves=18,
                        subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', lgbm_reg)])
my_pipeline.fit(X_train, y_train)
y_preds = my_pipeline.predict(X_valid)

mse = mean_squared_error(y_valid, y_preds)
rmse = np.sqrt(mse)

print('MSE : {0:.3f}, RMSE : {1: .3f}'.format(mse, rmse))

#RMSE :  1.952
#(learning_rate=0.2) : RMSE :  1.898
#(num_leaves =6) : RMSE :  1.878
#(num_leaves =7) : RMSE :  1.869
#(num_leaves =9) : RMSE :  1.860
#(num_leaves =11) : RMSE :  1.856
#(num_leaves =14) : RMSE :  1.848
#(num_leaves =16) : RMSE :  1.846
#(num_leaves =18) : RMSE :  1.845

#(colsample_bytree =0.2) : RMSE : 


# # Assemble

# In[45]:


assemble_preds = 0.5 * preds + 0.5 * y_preds
#xgb, lgbm assemple predictions
mse = mean_squared_error(y_valid, assemble_preds)
rmse = np.sqrt(mse)

print(rmse)

#0,5, 0.5 : 1.8310808351678143
#0,4, 0.6 : 1.832171476375979


# In[38]:


param_test1 = {'model__n_estimators':[1,10,100,1000]}

param_test2 = { 'model__max_depth':range(3,10,3), 'model__min_child_weight':range(1,6,2)}

param_test3 = { 'model__gamma':[i/10.0 for i in range(0,5)] }

param_test4 = { 'model__subsample':[i/10.0 for i in range(6,10)], 'model__colsample_bytree':[i/10.0 for i in range(6,10)] }



grid_pipeline_1 = GridSearchCV(my_pipeline, param_test2)
grid_pipeline_1.fit(X_train, y_train)

grid_pipeline_1.best_score_
grid_pipeline_1.best_params_


# In[ ]:


# def print_best_params(model, params):
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                               'model', model])
#     grid_pipeline = GridSearchCV(pipeline, param_grid=params,
#                              scoring='neg_mean_squared_error', cv=5)
#     grid_pipeline.fit(X_train, y_train)
#     rmse = np.sqrt(-1 * grid_pipeline.best_score_)
#     print('{0} 5 CV 시 최적 평균 RMSE 값:{1}, 최적 alpha:{2}'.format(model.__class__.__name__,
#                                                             np.round(rmse,4), grid_pipeline.best_params_))


# # Test

# In[48]:


X_test = wine_test.copy()

X_test['description_length'] = X_test.description.str.len()
X_test = X_test.drop(['taster_twitter_handle', 'id', 'region_2', 'country', 'description', 'winery'], axis=1)

preprocessed_X_test = preprocessor.transform(X_test)
preprocessed_X_test.index = wine_test.id

test_preds_xgb = model.predict(preprocessed_X_test)
test_preds_lgbm = lgbm_reg.predict(preprocessed_X_test)

test_preds = 0.5 * preds + 0.5 * y_preds
#xgb, lgbm assemple predictions


# In[53]:


my_submission = pd.DataFrame({'id': preprocessed_X_test.index, 'points': test_preds})
my_submission.to_csv('wine_submission_4.csv', index=False)


# In[ ]:




