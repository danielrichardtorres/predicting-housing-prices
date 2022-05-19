# Predicting House Sale Prices

> In this course, we started by building intuition for model based learning, explored how the linear regression model worked, understood how the two different approaches to model fitting worked, and some techniques for cleaning, transforming, and selecting features. In this guided project, you can practice what you learned in this course by exploring ways to improve the models we built.

>You'll work with housing data for the city of Ames, Iowa, United States from 2006 to 2010. You can read more about why the data was collected here. You can also read about the different columns in the data here.

>Let's start by setting up a pipeline of functions that will let us quickly iterate on different models.

- train
- transform_features()
- select_features()
- train_and_test()
- view resulting rmse_values and avg_mse



```python
import pandas as pd
pd.options.display.max_columns = 999
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import KFold

%matplotlib inline
import seaborn as sns
```


```python
df = pd.read_csv('AmesHousing.tsv', delimiter = '\t')
```


```python
def transform_features(df):
    return df

def select_features(df):
    return df[['Gr Liv Area', "SalePrice"]]

def train_and_test(df):
    train = df[:1460]
    test = df[1460:]
    
    numeric_train = train.select_dtypes(include = ['integer', 'float'])
    numeric_test = test.select_dtypes(include = ['integer', 'float'])
    
    features = numeric_train.columns.drop('SalePrice')
    
    model = linear_model.LinearRegression()
    model.fit(train[features], train['SalePrice'])
    
    predictions = model.predict(test[features])
    mse = mean_squared_error(test['SalePrice'], predictions)
    rmse = np.sqrt(mse)
    
    return rmse

transform_df = transform_features(df)
filtered_df = select_features(transform_df)
rmse = train_and_test(filtered_df)

rmse
```




    57088.25161263909



# Feature Engineering
- Handle missing values:
    - All columns:
        - Drop any with 5% or more missing values for now.
    - Text columns:
        - Drop any with 1 or more missing values for now.
    - Numerical columns:
        - For columns with missing values, fill in with the most common value in -that column

1: All columns: Drop any with 5% or more missing values for now.


```python
num_missing = df.isnull().sum()
#num_missing
#series obj: col name - num null val in col
```


```python
drop_missing_cols = num_missing[(num_missing > len(df)/20)].sort_values()
#type(drop_missing_cols)
df = df.drop(drop_missing_cols.index, axis=1)
```

2: Text columns: Drop any with 1 or more missing values for now.


```python
text_mv_count = df.select_dtypes(include = ['object']).isnull().sum().sort_values(ascending=False)
```


```python
#text_mv_count
```


```python
drop_missing_cols_2 = text_mv_count[text_mv_count>0]
df = df.drop(drop_missing_cols_2.index, axis = 1)
```


```python
drop_missing_cols_2
```




    Bsmt Exposure     83
    BsmtFin Type 2    81
    BsmtFin Type 1    80
    Bsmt Qual         80
    Bsmt Cond         80
    Mas Vnr Type      23
    Electrical         1
    dtype: int64



3: Numerical columns: For columns with missing values, fill in with the most common value in -that column


```python
#compute columnswise missing numeric value counts
num_missing = df.select_dtypes(include = ['integer', 'float']).isnull().sum().sort_values()
fixable_numeric_cols = num_missing[(num_missing < len(df)/20) & (num_missing > 0)].sort_values()
fixable_numeric_cols
```




    Garage Area        1
    BsmtFin SF 2       1
    Bsmt Unf SF        1
    Total Bsmt SF      1
    BsmtFin SF 1       1
    Garage Cars        1
    Bsmt Full Bath     2
    Bsmt Half Bath     2
    Mas Vnr Area      23
    dtype: int64




```python

## Compute the most common value for each column in `fixable_nmeric_missing_cols`.
replacement_values_dict = df[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
replacement_values_dict
```




    {'Bsmt Full Bath': 0.0,
     'Bsmt Half Bath': 0.0,
     'Bsmt Unf SF': 0.0,
     'BsmtFin SF 1': 0.0,
     'BsmtFin SF 2': 0.0,
     'Garage Area': 0.0,
     'Garage Cars': 2.0,
     'Mas Vnr Area': 0.0,
     'Total Bsmt SF': 0.0}




```python
## Use `pd.DataFrame.fillna()` to replace missing values.
df = df.fillna(replacement_values_dict)
```


```python
## Verify that every column has 0 missing values
df.isnull().sum().value_counts()
```




    0    64
    dtype: int64



What new features can we create, that better capture the information in some of the features?




```python
years_sold = df['Yr Sold'] - df['Year Built']
years_sold[years_sold < 0]
```




    2180   -1
    dtype: int64




```python
years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
years_since_remod[years_since_remod < 0]
```




    1702   -1
    2180   -2
    2181   -1
    dtype: int64




```python
## Create new columns
df['Years Before Sale'] = years_sold
df['Years Since Remod'] = years_since_remod

## Drop rows with negative values for both of these new features
df = df.drop([1702, 2180, 2181], axis=0)

## No longer need original year columns
df = df.drop(["Year Built", "Year Remod/Add"], axis = 1)
```


```python

## Drop columns that aren't useful for ML
df = df.drop(["PID", "Order"], axis=1)

## Drop columns that leak info about the final sale
df = df.drop(["Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis=1)
```

Let's update our functions



```python
def transform_features(df):
    #makes series obj to get num missing for each column
    num_missing = df.isnull().sum()
    #creates series that has info on if missing more than 5% 
    drop_missing_cols = num_missing[(num_missing > len(df)/20)].sort_values()
    #drops those columns missing more than 5% data
    df = df.drop(drop_missing_cols.index, axis=1)
    
    #next drop text columns with more than 1 missing value
    text_mv_count = df.select_dtypes(include = ['object']).isnull().sum().sort_values(ascending=False)
    drop_missing_cols_2 = text_mv_count[text_mv_count>0]
    df = df.drop(drop_missing_cols_2.index, axis = 1)
    
    #compute columnswise missing numeric value counts
    num_missing = df.select_dtypes(include = ['integer', 'float']).isnull().sum().sort_values()
    fixable_numeric_cols = num_missing[(num_missing < len(df)/20) & (num_missing > 0)].sort_values()
    ## Compute the most common value for each column in `fixable_nmeric_missing_cols`.
    replacement_values_dict = df[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
    ## Use `pd.DataFrame.fillna()` to replace missing values.
    df = df.fillna(replacement_values_dict)
    
    #create some features
    years_sold = df['Yr Sold'] - df['Year Built']
    years_sold[years_sold < 0]
    years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
    years_since_remod[years_since_remod < 0]
    ## Create new columns
    df['Years Before Sale'] = years_sold
    df['Years Since Remod'] = years_since_remod

    ## Drop rows with negative values for both of these new features
    df = df.drop([1702, 2180, 2181], axis=0)

    ## No longer need original year columns, or columns not useful for ML or columns that leak final sale data
    df = df.drop(["Year Built", "Year Remod/Add","PID", "Order","Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis = 1)
    
    return df

def select_features(df):
    return df[['Gr Liv Area', "SalePrice"]]

def train_and_test(df):
    train = df[:1460]
    test = df[1460:]
    
    numeric_train = train.select_dtypes(include = ['integer', 'float'])
    numeric_test = test.select_dtypes(include = ['integer', 'float'])
    
    features = numeric_train.columns.drop('SalePrice')
    
    model = linear_model.LinearRegression()
    model.fit(train[features], train['SalePrice'])
    
    predictions = model.predict(test[features])
    mse = mean_squared_error(test['SalePrice'], predictions)
    rmse = np.sqrt(mse)
    
    return rmse

df = pd.read_csv('AmesHousing.tsv', delimiter = '\t')
transform_df = transform_features(df)
filtered_df = select_features(transform_df)
rmse = train_and_test(filtered_df)

rmse
```




    55275.36731241307



# Feature Selection 
> Now that we have cleaned and transformed a lot of the features in the data set, it's time to move on to feature selection for numerical features.


```python
#generate a correlation heatmap of numerical features in the training data set
numerical_df = transform_df.select_dtypes(include=['int', 'float'])
numerical_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MS SubClass</th>
      <th>Lot Area</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Mas Vnr Area</th>
      <th>BsmtFin SF 1</th>
      <th>BsmtFin SF 2</th>
      <th>Bsmt Unf SF</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>2nd Flr SF</th>
      <th>Low Qual Fin SF</th>
      <th>Gr Liv Area</th>
      <th>Bsmt Full Bath</th>
      <th>Bsmt Half Bath</th>
      <th>Full Bath</th>
      <th>Half Bath</th>
      <th>Bedroom AbvGr</th>
      <th>Kitchen AbvGr</th>
      <th>TotRms AbvGrd</th>
      <th>Fireplaces</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Misc Val</th>
      <th>SalePrice</th>
      <th>Years Before Sale</th>
      <th>Years Since Remod</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>31770</td>
      <td>6</td>
      <td>5</td>
      <td>112.0</td>
      <td>639.0</td>
      <td>0.0</td>
      <td>441.0</td>
      <td>1080.0</td>
      <td>1656</td>
      <td>0</td>
      <td>0</td>
      <td>1656</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>2.0</td>
      <td>528.0</td>
      <td>210</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215000</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>11622</td>
      <td>5</td>
      <td>6</td>
      <td>0.0</td>
      <td>468.0</td>
      <td>144.0</td>
      <td>270.0</td>
      <td>882.0</td>
      <td>896</td>
      <td>0</td>
      <td>0</td>
      <td>896</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1.0</td>
      <td>730.0</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>105000</td>
      <td>49</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>14267</td>
      <td>6</td>
      <td>6</td>
      <td>108.0</td>
      <td>923.0</td>
      <td>0.0</td>
      <td>406.0</td>
      <td>1329.0</td>
      <td>1329</td>
      <td>0</td>
      <td>0</td>
      <td>1329</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>1.0</td>
      <td>312.0</td>
      <td>393</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12500</td>
      <td>172000</td>
      <td>52</td>
      <td>52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>11160</td>
      <td>7</td>
      <td>5</td>
      <td>0.0</td>
      <td>1065.0</td>
      <td>0.0</td>
      <td>1045.0</td>
      <td>2110.0</td>
      <td>2110</td>
      <td>0</td>
      <td>0</td>
      <td>2110</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>2.0</td>
      <td>522.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>244000</td>
      <td>42</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>13830</td>
      <td>5</td>
      <td>5</td>
      <td>0.0</td>
      <td>791.0</td>
      <td>0.0</td>
      <td>137.0</td>
      <td>928.0</td>
      <td>928</td>
      <td>701</td>
      <td>0</td>
      <td>1629</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2.0</td>
      <td>482.0</td>
      <td>212</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>189900</td>
      <td>13</td>
      <td>12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>9978</td>
      <td>6</td>
      <td>6</td>
      <td>20.0</td>
      <td>602.0</td>
      <td>0.0</td>
      <td>324.0</td>
      <td>926.0</td>
      <td>926</td>
      <td>678</td>
      <td>0</td>
      <td>1604</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>2.0</td>
      <td>470.0</td>
      <td>360</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>195500</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>120</td>
      <td>4920</td>
      <td>8</td>
      <td>5</td>
      <td>0.0</td>
      <td>616.0</td>
      <td>0.0</td>
      <td>722.0</td>
      <td>1338.0</td>
      <td>1338</td>
      <td>0</td>
      <td>0</td>
      <td>1338</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>2.0</td>
      <td>582.0</td>
      <td>0</td>
      <td>0</td>
      <td>170</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>213500</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>120</td>
      <td>5005</td>
      <td>8</td>
      <td>5</td>
      <td>0.0</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>1017.0</td>
      <td>1280.0</td>
      <td>1280</td>
      <td>0</td>
      <td>0</td>
      <td>1280</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>2.0</td>
      <td>506.0</td>
      <td>0</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
      <td>144</td>
      <td>0</td>
      <td>0</td>
      <td>191500</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>120</td>
      <td>5389</td>
      <td>8</td>
      <td>5</td>
      <td>0.0</td>
      <td>1180.0</td>
      <td>0.0</td>
      <td>415.0</td>
      <td>1595.0</td>
      <td>1616</td>
      <td>0</td>
      <td>0</td>
      <td>1616</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2.0</td>
      <td>608.0</td>
      <td>237</td>
      <td>152</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>236500</td>
      <td>15</td>
      <td>14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>60</td>
      <td>7500</td>
      <td>7</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>994.0</td>
      <td>994.0</td>
      <td>1028</td>
      <td>776</td>
      <td>0</td>
      <td>1804</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>2.0</td>
      <td>442.0</td>
      <td>140</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>189000</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>10</th>
      <td>60</td>
      <td>10000</td>
      <td>6</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>763.0</td>
      <td>763.0</td>
      <td>763</td>
      <td>892</td>
      <td>0</td>
      <td>1655</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>2.0</td>
      <td>440.0</td>
      <td>157</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>175900</td>
      <td>17</td>
      <td>16</td>
    </tr>
    <tr>
      <th>11</th>
      <td>20</td>
      <td>7980</td>
      <td>6</td>
      <td>7</td>
      <td>0.0</td>
      <td>935.0</td>
      <td>0.0</td>
      <td>233.0</td>
      <td>1168.0</td>
      <td>1187</td>
      <td>0</td>
      <td>0</td>
      <td>1187</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>2.0</td>
      <td>420.0</td>
      <td>483</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>500</td>
      <td>185000</td>
      <td>18</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>60</td>
      <td>8402</td>
      <td>6</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>789.0</td>
      <td>789.0</td>
      <td>789</td>
      <td>676</td>
      <td>0</td>
      <td>1465</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>2.0</td>
      <td>393.0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>180400</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20</td>
      <td>10176</td>
      <td>7</td>
      <td>5</td>
      <td>0.0</td>
      <td>637.0</td>
      <td>0.0</td>
      <td>663.0</td>
      <td>1300.0</td>
      <td>1341</td>
      <td>0</td>
      <td>0</td>
      <td>1341</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2.0</td>
      <td>506.0</td>
      <td>192</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>171500</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>14</th>
      <td>120</td>
      <td>6820</td>
      <td>8</td>
      <td>5</td>
      <td>0.0</td>
      <td>368.0</td>
      <td>1120.0</td>
      <td>0.0</td>
      <td>1488.0</td>
      <td>1502</td>
      <td>0</td>
      <td>0</td>
      <td>1502</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>2.0</td>
      <td>528.0</td>
      <td>0</td>
      <td>54</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>212000</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <th>15</th>
      <td>60</td>
      <td>53504</td>
      <td>8</td>
      <td>5</td>
      <td>603.0</td>
      <td>1416.0</td>
      <td>0.0</td>
      <td>234.0</td>
      <td>1650.0</td>
      <td>1690</td>
      <td>1589</td>
      <td>0</td>
      <td>3279</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>3.0</td>
      <td>841.0</td>
      <td>503</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>210</td>
      <td>0</td>
      <td>0</td>
      <td>538000</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>16</th>
      <td>50</td>
      <td>12134</td>
      <td>8</td>
      <td>7</td>
      <td>0.0</td>
      <td>427.0</td>
      <td>0.0</td>
      <td>132.0</td>
      <td>559.0</td>
      <td>1080</td>
      <td>672</td>
      <td>0</td>
      <td>1752</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>2.0</td>
      <td>492.0</td>
      <td>325</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>164000</td>
      <td>22</td>
      <td>5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>20</td>
      <td>11394</td>
      <td>9</td>
      <td>2</td>
      <td>350.0</td>
      <td>1445.0</td>
      <td>0.0</td>
      <td>411.0</td>
      <td>1856.0</td>
      <td>1856</td>
      <td>0</td>
      <td>0</td>
      <td>1856</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>3.0</td>
      <td>834.0</td>
      <td>113</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>394432</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>20</td>
      <td>19138</td>
      <td>4</td>
      <td>5</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>744.0</td>
      <td>864.0</td>
      <td>864</td>
      <td>0</td>
      <td>0</td>
      <td>864</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>2.0</td>
      <td>400.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>141000</td>
      <td>59</td>
      <td>59</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>13175</td>
      <td>6</td>
      <td>6</td>
      <td>119.0</td>
      <td>790.0</td>
      <td>163.0</td>
      <td>589.0</td>
      <td>1542.0</td>
      <td>2073</td>
      <td>0</td>
      <td>0</td>
      <td>2073</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>2.0</td>
      <td>500.0</td>
      <td>349</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>210000</td>
      <td>32</td>
      <td>22</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>11751</td>
      <td>6</td>
      <td>6</td>
      <td>480.0</td>
      <td>705.0</td>
      <td>0.0</td>
      <td>1139.0</td>
      <td>1844.0</td>
      <td>1844</td>
      <td>0</td>
      <td>0</td>
      <td>1844</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>2.0</td>
      <td>546.0</td>
      <td>0</td>
      <td>122</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>190000</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>21</th>
      <td>85</td>
      <td>10625</td>
      <td>7</td>
      <td>6</td>
      <td>81.0</td>
      <td>885.0</td>
      <td>168.0</td>
      <td>0.0</td>
      <td>1053.0</td>
      <td>1173</td>
      <td>0</td>
      <td>0</td>
      <td>1173</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>2.0</td>
      <td>528.0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>170000</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>22</th>
      <td>60</td>
      <td>7500</td>
      <td>7</td>
      <td>5</td>
      <td>0.0</td>
      <td>533.0</td>
      <td>0.0</td>
      <td>281.0</td>
      <td>814.0</td>
      <td>814</td>
      <td>860</td>
      <td>0</td>
      <td>1674</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>2.0</td>
      <td>663.0</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>216000</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>23</th>
      <td>20</td>
      <td>11241</td>
      <td>6</td>
      <td>7</td>
      <td>180.0</td>
      <td>578.0</td>
      <td>0.0</td>
      <td>426.0</td>
      <td>1004.0</td>
      <td>1004</td>
      <td>0</td>
      <td>0</td>
      <td>1004</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2.0</td>
      <td>480.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>700</td>
      <td>149000</td>
      <td>40</td>
      <td>40</td>
    </tr>
    <tr>
      <th>24</th>
      <td>20</td>
      <td>12537</td>
      <td>5</td>
      <td>6</td>
      <td>0.0</td>
      <td>734.0</td>
      <td>0.0</td>
      <td>344.0</td>
      <td>1078.0</td>
      <td>1078</td>
      <td>0</td>
      <td>0</td>
      <td>1078</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2.0</td>
      <td>500.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>149900</td>
      <td>39</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>20</td>
      <td>8450</td>
      <td>5</td>
      <td>6</td>
      <td>0.0</td>
      <td>775.0</td>
      <td>0.0</td>
      <td>281.0</td>
      <td>1056.0</td>
      <td>1056</td>
      <td>0</td>
      <td>0</td>
      <td>1056</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1.0</td>
      <td>304.0</td>
      <td>0</td>
      <td>85</td>
      <td>184</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142000</td>
      <td>42</td>
      <td>42</td>
    </tr>
    <tr>
      <th>26</th>
      <td>20</td>
      <td>8400</td>
      <td>4</td>
      <td>5</td>
      <td>0.0</td>
      <td>804.0</td>
      <td>78.0</td>
      <td>0.0</td>
      <td>882.0</td>
      <td>882</td>
      <td>0</td>
      <td>0</td>
      <td>882</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>2.0</td>
      <td>525.0</td>
      <td>240</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>126000</td>
      <td>40</td>
      <td>40</td>
    </tr>
    <tr>
      <th>27</th>
      <td>20</td>
      <td>10500</td>
      <td>4</td>
      <td>5</td>
      <td>0.0</td>
      <td>432.0</td>
      <td>0.0</td>
      <td>432.0</td>
      <td>864.0</td>
      <td>864</td>
      <td>0</td>
      <td>0</td>
      <td>864</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>115000</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>28</th>
      <td>120</td>
      <td>5858</td>
      <td>7</td>
      <td>5</td>
      <td>0.0</td>
      <td>1051.0</td>
      <td>0.0</td>
      <td>354.0</td>
      <td>1405.0</td>
      <td>1337</td>
      <td>0</td>
      <td>0</td>
      <td>1337</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2.0</td>
      <td>511.0</td>
      <td>203</td>
      <td>68</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>184000</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>29</th>
      <td>160</td>
      <td>1680</td>
      <td>6</td>
      <td>5</td>
      <td>504.0</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>327.0</td>
      <td>483.0</td>
      <td>483</td>
      <td>504</td>
      <td>0</td>
      <td>987</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1.0</td>
      <td>264.0</td>
      <td>275</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96000</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2900</th>
      <td>20</td>
      <td>13618</td>
      <td>8</td>
      <td>5</td>
      <td>198.0</td>
      <td>1350.0</td>
      <td>0.0</td>
      <td>378.0</td>
      <td>1728.0</td>
      <td>1960</td>
      <td>0</td>
      <td>0</td>
      <td>1960</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>3.0</td>
      <td>714.0</td>
      <td>172</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>320000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2901</th>
      <td>20</td>
      <td>11443</td>
      <td>8</td>
      <td>5</td>
      <td>208.0</td>
      <td>1460.0</td>
      <td>0.0</td>
      <td>408.0</td>
      <td>1868.0</td>
      <td>2028</td>
      <td>0</td>
      <td>0</td>
      <td>2028</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>3.0</td>
      <td>880.0</td>
      <td>326</td>
      <td>66</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>369900</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2902</th>
      <td>20</td>
      <td>11577</td>
      <td>9</td>
      <td>5</td>
      <td>382.0</td>
      <td>1455.0</td>
      <td>0.0</td>
      <td>383.0</td>
      <td>1838.0</td>
      <td>1838</td>
      <td>0</td>
      <td>0</td>
      <td>1838</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>3.0</td>
      <td>682.0</td>
      <td>161</td>
      <td>225</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>359900</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2903</th>
      <td>20</td>
      <td>31250</td>
      <td>1</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1600</td>
      <td>0</td>
      <td>0</td>
      <td>1600</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>1.0</td>
      <td>270.0</td>
      <td>0</td>
      <td>0</td>
      <td>135</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>81500</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>2904</th>
      <td>90</td>
      <td>7020</td>
      <td>7</td>
      <td>5</td>
      <td>200.0</td>
      <td>1243.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>1288.0</td>
      <td>1368</td>
      <td>0</td>
      <td>0</td>
      <td>1368</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>4.0</td>
      <td>784.0</td>
      <td>0</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215000</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2905</th>
      <td>120</td>
      <td>4500</td>
      <td>6</td>
      <td>5</td>
      <td>116.0</td>
      <td>897.0</td>
      <td>0.0</td>
      <td>319.0</td>
      <td>1216.0</td>
      <td>1216</td>
      <td>0</td>
      <td>0</td>
      <td>1216</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>2.0</td>
      <td>402.0</td>
      <td>0</td>
      <td>125</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>164000</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2906</th>
      <td>120</td>
      <td>4500</td>
      <td>6</td>
      <td>5</td>
      <td>443.0</td>
      <td>1201.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>1237.0</td>
      <td>1337</td>
      <td>0</td>
      <td>0</td>
      <td>1337</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>2.0</td>
      <td>405.0</td>
      <td>0</td>
      <td>199</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>153500</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2907</th>
      <td>20</td>
      <td>17217</td>
      <td>5</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1140.0</td>
      <td>1140.0</td>
      <td>1140</td>
      <td>0</td>
      <td>0</td>
      <td>1140</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>36</td>
      <td>56</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>84500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>160</td>
      <td>2665</td>
      <td>5</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>264.0</td>
      <td>264.0</td>
      <td>616</td>
      <td>688</td>
      <td>0</td>
      <td>1304</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
      <td>336.0</td>
      <td>141</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>104500</td>
      <td>29</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2909</th>
      <td>160</td>
      <td>2665</td>
      <td>5</td>
      <td>6</td>
      <td>0.0</td>
      <td>548.0</td>
      <td>173.0</td>
      <td>36.0</td>
      <td>757.0</td>
      <td>925</td>
      <td>550</td>
      <td>0</td>
      <td>1475</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1.0</td>
      <td>336.0</td>
      <td>104</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>127000</td>
      <td>29</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2910</th>
      <td>160</td>
      <td>3964</td>
      <td>6</td>
      <td>4</td>
      <td>0.0</td>
      <td>837.0</td>
      <td>0.0</td>
      <td>105.0</td>
      <td>942.0</td>
      <td>1291</td>
      <td>1230</td>
      <td>0</td>
      <td>2521</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>2.0</td>
      <td>576.0</td>
      <td>728</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>151400</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2911</th>
      <td>20</td>
      <td>10172</td>
      <td>5</td>
      <td>7</td>
      <td>0.0</td>
      <td>441.0</td>
      <td>0.0</td>
      <td>423.0</td>
      <td>864.0</td>
      <td>874</td>
      <td>0</td>
      <td>0</td>
      <td>874</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1.0</td>
      <td>288.0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>126500</td>
      <td>38</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2912</th>
      <td>90</td>
      <td>11836</td>
      <td>5</td>
      <td>5</td>
      <td>0.0</td>
      <td>149.0</td>
      <td>0.0</td>
      <td>1503.0</td>
      <td>1652.0</td>
      <td>1652</td>
      <td>0</td>
      <td>0</td>
      <td>1652</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>3.0</td>
      <td>928.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>146500</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2913</th>
      <td>180</td>
      <td>1470</td>
      <td>4</td>
      <td>6</td>
      <td>0.0</td>
      <td>522.0</td>
      <td>0.0</td>
      <td>108.0</td>
      <td>630.0</td>
      <td>630</td>
      <td>0</td>
      <td>0</td>
      <td>630</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>73000</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2914</th>
      <td>160</td>
      <td>1484</td>
      <td>4</td>
      <td>4</td>
      <td>0.0</td>
      <td>252.0</td>
      <td>0.0</td>
      <td>294.0</td>
      <td>546.0</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1.0</td>
      <td>253.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>79400</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2915</th>
      <td>20</td>
      <td>13384</td>
      <td>5</td>
      <td>5</td>
      <td>194.0</td>
      <td>119.0</td>
      <td>344.0</td>
      <td>641.0</td>
      <td>1104.0</td>
      <td>1360</td>
      <td>0</td>
      <td>0</td>
      <td>1360</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>1.0</td>
      <td>336.0</td>
      <td>160</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>140000</td>
      <td>37</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2916</th>
      <td>180</td>
      <td>1533</td>
      <td>5</td>
      <td>7</td>
      <td>0.0</td>
      <td>553.0</td>
      <td>0.0</td>
      <td>77.0</td>
      <td>630.0</td>
      <td>630</td>
      <td>0</td>
      <td>0</td>
      <td>630</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>92000</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2917</th>
      <td>160</td>
      <td>1533</td>
      <td>4</td>
      <td>5</td>
      <td>0.0</td>
      <td>408.0</td>
      <td>0.0</td>
      <td>138.0</td>
      <td>546.0</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1.0</td>
      <td>286.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>87550</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2918</th>
      <td>160</td>
      <td>1526</td>
      <td>4</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>546.0</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>79500</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2919</th>
      <td>160</td>
      <td>1936</td>
      <td>4</td>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>546.0</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>90500</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2920</th>
      <td>160</td>
      <td>1894</td>
      <td>4</td>
      <td>5</td>
      <td>0.0</td>
      <td>252.0</td>
      <td>0.0</td>
      <td>294.0</td>
      <td>546.0</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>1.0</td>
      <td>286.0</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>71000</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2921</th>
      <td>90</td>
      <td>12640</td>
      <td>6</td>
      <td>5</td>
      <td>0.0</td>
      <td>936.0</td>
      <td>396.0</td>
      <td>396.0</td>
      <td>1728.0</td>
      <td>1728</td>
      <td>0</td>
      <td>0</td>
      <td>1728</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>2.0</td>
      <td>574.0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>150900</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2922</th>
      <td>90</td>
      <td>9297</td>
      <td>5</td>
      <td>5</td>
      <td>0.0</td>
      <td>1606.0</td>
      <td>0.0</td>
      <td>122.0</td>
      <td>1728.0</td>
      <td>1728</td>
      <td>0</td>
      <td>0</td>
      <td>1728</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>2.0</td>
      <td>560.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>188000</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2923</th>
      <td>20</td>
      <td>17400</td>
      <td>5</td>
      <td>5</td>
      <td>0.0</td>
      <td>936.0</td>
      <td>0.0</td>
      <td>190.0</td>
      <td>1126.0</td>
      <td>1126</td>
      <td>0</td>
      <td>0</td>
      <td>1126</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2.0</td>
      <td>484.0</td>
      <td>295</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>160000</td>
      <td>29</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2924</th>
      <td>20</td>
      <td>20000</td>
      <td>5</td>
      <td>7</td>
      <td>0.0</td>
      <td>1224.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1224.0</td>
      <td>1224</td>
      <td>0</td>
      <td>0</td>
      <td>1224</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>2.0</td>
      <td>576.0</td>
      <td>474</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>131000</td>
      <td>46</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2925</th>
      <td>80</td>
      <td>7937</td>
      <td>6</td>
      <td>6</td>
      <td>0.0</td>
      <td>819.0</td>
      <td>0.0</td>
      <td>184.0</td>
      <td>1003.0</td>
      <td>1003</td>
      <td>0</td>
      <td>0</td>
      <td>1003</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>2.0</td>
      <td>588.0</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142500</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>20</td>
      <td>8885</td>
      <td>5</td>
      <td>5</td>
      <td>0.0</td>
      <td>301.0</td>
      <td>324.0</td>
      <td>239.0</td>
      <td>864.0</td>
      <td>902</td>
      <td>0</td>
      <td>0</td>
      <td>902</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>2.0</td>
      <td>484.0</td>
      <td>164</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>131000</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>85</td>
      <td>10441</td>
      <td>5</td>
      <td>5</td>
      <td>0.0</td>
      <td>337.0</td>
      <td>0.0</td>
      <td>575.0</td>
      <td>912.0</td>
      <td>970</td>
      <td>0</td>
      <td>0</td>
      <td>970</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>80</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>700</td>
      <td>132000</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2928</th>
      <td>20</td>
      <td>10010</td>
      <td>5</td>
      <td>5</td>
      <td>0.0</td>
      <td>1071.0</td>
      <td>123.0</td>
      <td>195.0</td>
      <td>1389.0</td>
      <td>1389</td>
      <td>0</td>
      <td>0</td>
      <td>1389</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2.0</td>
      <td>418.0</td>
      <td>240</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>170000</td>
      <td>32</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2929</th>
      <td>60</td>
      <td>9627</td>
      <td>7</td>
      <td>5</td>
      <td>94.0</td>
      <td>758.0</td>
      <td>0.0</td>
      <td>238.0</td>
      <td>996.0</td>
      <td>996</td>
      <td>1004</td>
      <td>0</td>
      <td>2000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>3.0</td>
      <td>650.0</td>
      <td>190</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>188000</td>
      <td>13</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>2927 rows × 33 columns</p>
</div>




```python
x = numerical_df.corr()
hm = sns.heatmap(x)
```


    
![png](housing_files/housing_26_0.png)
    



```python
#calc correlation coeff for cols that corr well with SalePrice
high_corr_cols = ["SalePrice",'Overall Qual', 'Total Bsmt SF', "Gr Liv Area", "Garage Area"]
hm2 = sns.heatmap(numerical_df[high_corr_cols].corr())
```


    
![png](housing_files/housing_27_0.png)
    



```python
abs_corr_coeffs = numerical_df.corr()['SalePrice'].abs().sort_values(ascending=False)
abs_corr_coeffs
```




    SalePrice            1.000000
    Overall Qual         0.801206
    Gr Liv Area          0.717596
    Garage Cars          0.648361
    Total Bsmt SF        0.644012
    Garage Area          0.641425
    1st Flr SF           0.635185
    Years Before Sale    0.558979
    Full Bath            0.546118
    Years Since Remod    0.534985
    Mas Vnr Area         0.506983
    TotRms AbvGrd        0.498574
    Fireplaces           0.474831
    BsmtFin SF 1         0.439284
    Wood Deck SF         0.328183
    Open Porch SF        0.316262
    Half Bath            0.284871
    Bsmt Full Bath       0.276258
    2nd Flr SF           0.269601
    Lot Area             0.267520
    Bsmt Unf SF          0.182751
    Bedroom AbvGr        0.143916
    Enclosed Porch       0.128685
    Kitchen AbvGr        0.119760
    Screen Porch         0.112280
    Overall Cond         0.101540
    MS SubClass          0.085128
    Pool Area            0.068438
    Low Qual Fin SF      0.037629
    Bsmt Half Bath       0.035875
    3Ssn Porch           0.032268
    Misc Val             0.019273
    BsmtFin SF 2         0.006127
    Name: SalePrice, dtype: float64




```python
## Let's only keep columns with a correlation coefficient of larger than 0.4 (arbitrary, worth experimenting later!)
abs_corr_coeffs[abs_corr_coeffs > 0.4]
```




    SalePrice            1.000000
    Overall Qual         0.801206
    Gr Liv Area          0.717596
    Garage Cars          0.648361
    Total Bsmt SF        0.644012
    Garage Area          0.641425
    1st Flr SF           0.635185
    Years Before Sale    0.558979
    Full Bath            0.546118
    Years Since Remod    0.534985
    Mas Vnr Area         0.506983
    TotRms AbvGrd        0.498574
    Fireplaces           0.474831
    BsmtFin SF 1         0.439284
    Name: SalePrice, dtype: float64




```python
## Drop columns with less than 0.4 correlation with SalePrice
transform_df = transform_df.drop(abs_corr_coeffs[abs_corr_coeffs < 0.4].index, axis=1)
```


```python
#look at categorical data convert those

## Create a list of column names from documentation that are *meant* to be categorical
nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 
                    "Misc Feature", "Sale Type", "Sale Condition"]
```


```python
## Which categorical columns have we still carried with us? We'll test tehse 
transform_cat_cols = []
for col in nominal_features:
    if col in transform_df.columns:
        transform_cat_cols.append(col)

## How many unique values in each categorical column?
uniqueness_counts = transform_df[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()
## Aribtrary cutoff of 10 unique values (worth experimenting)
drop_nonuniq_cols = uniqueness_counts[uniqueness_counts > 10].index
transform_df = transform_df.drop(drop_nonuniq_cols, axis=1)

## Select just the remaining text columns and convert to categorical
text_cols = transform_df.select_dtypes(include=['object'])
for col in text_cols:
    transform_df[col] = transform_df[col].astype('category')
    
## Create dummy columns and add back to the dataframe!
transform_df = pd.concat([
    transform_df, 
    pd.get_dummies(transform_df.select_dtypes(include=['category']))
], axis=1).drop(text_cols,axis=1)
```

Update select_features() with the above experiements


```python
def transform_features(df):
    num_missing = df.isnull().sum()
    drop_missing_cols = num_missing[(num_missing > len(df)/20)].sort_values()
    df = df.drop(drop_missing_cols.index, axis=1)
    
    text_mv_counts = df.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    df = df.drop(drop_missing_cols_2.index, axis=1)
    
    num_missing = df.select_dtypes(include=['int', 'float']).isnull().sum()
    fixable_numeric_cols = num_missing[(num_missing < len(df)/20) & (num_missing > 0)].sort_values()
    replacement_values_dict = df[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
    df = df.fillna(replacement_values_dict)
    
    years_sold = df['Yr Sold'] - df['Year Built']
    years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
    df['Years Before Sale'] = years_sold
    df['Years Since Remod'] = years_since_remod
    df = df.drop([1702, 2180, 2181], axis=0)

    df = df.drop(["PID", "Order", "Mo Sold", "Sale Condition", "Sale Type", "Year Built", "Year Remod/Add"], axis=1)
    return df

def select_features(df, coeff_threshold=0.4, uniq_threshold=10):
    numerical_df = df.select_dtypes(include=['int', 'float'])
    abs_corr_coeffs = numerical_df.corr()['SalePrice'].abs().sort_values()
    df = df.drop(abs_corr_coeffs[abs_corr_coeffs < coeff_threshold].index, axis=1)
    
    nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 
                    "Misc Feature", "Sale Type", "Sale Condition"]
    
    transform_cat_cols = []
    for col in nominal_features:
        if col in df.columns:
            transform_cat_cols.append(col)

    uniqueness_counts = df[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()
    drop_nonuniq_cols = uniqueness_counts[uniqueness_counts > 10].index
    df = df.drop(drop_nonuniq_cols, axis=1)
    
    text_cols = df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col] = df[col].astype('category')
    df = pd.concat([df, pd.get_dummies(df.select_dtypes(include=['category']))], axis=1).drop(text_cols,axis=1)
    
    return df

def train_and_test(df, k=0):
    numeric_df = df.select_dtypes(include=['integer', 'float'])
    features = numeric_df.columns.drop("SalePrice")
    lr = linear_model.LinearRegression()
    
    if k == 0:
        train = df[:1460]
        test = df[1460:]

        lr.fit(train[features], train["SalePrice"])
        predictions = lr.predict(test[features])
        mse = mean_squared_error(test["SalePrice"], predictions)
        rmse = np.sqrt(mse)

        return rmse
    
    if k == 1:
        # Randomize *all* rows (frac=1) from `df` and return
        shuffled_df = df.sample(frac=1, )
        train = df[:1460]
        test = df[1460:]
        
        lr.fit(train[features], train["SalePrice"])
        predictions_one = lr.predict(test[features])        
        
        mse_one = mean_squared_error(test["SalePrice"], predictions_one)
        rmse_one = np.sqrt(mse_one)
        
        lr.fit(test[features], test["SalePrice"])
        predictions_two = lr.predict(train[features])        
       
        mse_two = mean_squared_error(train["SalePrice"], predictions_two)
        rmse_two = np.sqrt(mse_two)
        
        avg_rmse = np.mean([rmse_one, rmse_two])
        print(rmse_one)
        print(rmse_two)
        return avg_rmse
    else:
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values = []
        for train_index, test_index, in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            lr.fit(train[features], train["SalePrice"])
            predictions = lr.predict(test[features])
            mse = mean_squared_error(test["SalePrice"], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
        print(rmse_values)
        avg_rmse = np.mean(rmse_values)
        return avg_rmse

df = pd.read_csv("AmesHousing.tsv", delimiter="\t")
transform_df = transform_features(df)
filtered_df = select_features(transform_df)
rmse = train_and_test(filtered_df, k=4)

rmse
```

    [30688.490853226755, 24783.299063000337, 37053.81398884517, 24739.1736990907]
    




    29316.19440104074




```python

```
