import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statistics as st
import scipy.stats as stats
import numpy as np


#generate a test dataframe with same statistical features to original dataset
def test_dataframe_generator(df_original):
    test_df = pd.DataFrame().reindex_like(df_original)
    for i in range(df_original.shape[1]):
        data_col = df_original.iloc[:, i]
        if ( i==0 or i==2):
            test_df.iloc[:,i] = data_col
            continue
        data_col_mean = st.mean(data_col)
        data_col_sd = st.stdev(data_col)
        data_col_min = data_col.min()
        data_col_max = data_col.max()
        dist = stats.truncnorm((data_col_min - data_col_mean) / data_col_sd, (data_col_max - data_col_mean) / data_col_sd, loc=data_col_mean, scale=data_col_sd)
        values = dist.rvs(df_original.shape[0])
        if (i ==1 or i ==3 or i==8):
            values = np.int_(values)
        test_df.iloc[:,i] = values
    return test_df


#selecting the important features for our model
def feature_sel(df):
    X = df.drop(columns=['profit'])
    y = df['profit']
    # Correlation
    corr = df.corr()
    f, ax = plt.subplots(figsize = (12,9))
    print(corr['profit'])
    sns.heatmap(corr, vmax =.8, square =True)
    plt.savefig("result/Img/correlation_heatmap")

    #select from model
    model = RandomForestRegressor(n_estimators=500, random_state=1).fit(X, y)
    names = df.columns.values[0:-1]
    ticks = [i for i in range(len(names))]
    fig = plt.figure(figsize = (12,9))
    plt.bar(ticks, model.feature_importances_)
    plt.xticks(ticks, names)
    plt.show()
    plt.savefig("result/Img/feature_importance")


#train the model and return back the predicted profit of test data frame
def train_model(df_original, df_test, method) :

    X_df_test = df_test.drop(columns=['profit','Station','new customer in %'])
    X = df_original.drop(columns=['profit','Station','new customer in %'])
    y = df_original['profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if method == "linearReg":
        model = LinearRegression().fit(X_train, y_train)
    if method == "RandomForest":
        model = RandomForestRegressor(n_estimators=500, random_state=1).fit(X_train, y_train)
    y_df_test_pred = model.predict(X_df_test)
    return y_df_test_pred

df = pd.read_csv ('data/TV_CASE.csv', sep=';')
df.dropna(inplace=True)

#generate test dataframe
test_df = test_dataframe_generator(df)

#set index for original and test data frame
df.set_index('datetime',inplace=True)
test_df.set_index('datetime', inplace= True)

# applying feature selection method on original dataframe
feature_sel(df)

#applying machine learning model on data
test_df.iloc[:,3] = train_model(df,test_df,"RandomForest")

com_profit = (df.groupby('Programme').sum())['profit'] - (test_df.groupby('Programme').sum())['profit']
succ_spot = com_profit.copy()
succ_spot[succ_spot >= 0] = 1
succ_spot[succ_spot < 0] = 0

#adding successful and compare profit cols to original data frame
df_result = pd.DataFrame( columns=['Programme','Profit_first_month','Profit_second_month','successful'])
df_result['Programme'] = df['Programme'].unique()
df_result.sort_values(by=['Programme'], inplace=True)
df_result['Profit_first_month'] = (df.groupby('Programme').sum())['profit']
df_result['Profit_second_month'] = (test_df.groupby('Programme').sum())['profit']
df_result['successful'] = succ_spot


df_result.to_csv("result/result.csv")




















