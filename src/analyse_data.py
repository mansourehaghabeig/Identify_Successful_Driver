import pandas as pd
import matplotlib.pyplot as plt

#function for ploting time series of a metric
def plot_gb_time_series(df, ts_name, gb_name, value_name, title=None):
    '''
    Runs groupby on Pandas dataframe and produces a time series chart.
    Parameters:
    ----------
    df : Pandas dataframe
    ts_name : string
        The name of the df column that has the datetime timestamp x-axis values.
    gb_name : string
        The name of the df column to perform group-by.
    value_name : string
        The name of the df column for the y-axis.
        title : string
        Optional title
    '''

    grouped = df.groupby([gb_name])
    for key, grp in grouped:
        fig, ax = plt.subplots()
        ax = grp.plot(ax=ax, kind='line', x=ts_name, y=value_name, label=0, marker='o')
        ax.autoscale_view()
        ax.legend(loc='upper left')
        _ = plt.grid()
        _ = plt.gcf().subplots_adjust(bottom=0.38)
        _ = plt.xticks(rotation=90, )
        _ = plt.xlabel('', fontsize=14)
        _ = plt.ylabel(value_name)
        if title is not None:
            _ = plt.title(title)
        _ = plt.show()








#reading dataset & and checking data types
df = pd.read_csv ('data/TV_CASE.csv', sep=';')
df_info = df.info()

#checking if dataset contains any missing values & clean it
if (df.isnull().values.any()):
    df.dropna(inplace=True)

#correlation maps bettween all features
corr = df.corr()


#Time series data
df_profit = df[['Programme','datetime','profit']]
plot_gb_time_series(df_profit, 'datetime', 'Programme', 'profit', title="profit")
df_order_sum = df[['Programme','datetime','order_sum']]
plot_gb_time_series(df_order_sum, 'datetime', 'Programme', 'order_sum', title="order_sum")
df_vis = df[['Programme','datetime','vis']]
plot_gb_time_series(df_vis, 'datetime', 'Programme', 'vis', title="visit")
df_new_customer = df[['Programme','datetime','new customer in %']]
plot_gb_time_series(df_new_customer, 'datetime', 'Programme', 'new customer in %', title="new customer in %")











