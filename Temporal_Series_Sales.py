import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.api import ExponentialSmoothing

#  MORE INFO ABOUT THE DATASET - https://www.kaggle.com/c/rossmann-store-sales

df_stores_base_sales = pd.read_csv('train.csv')
df_sales = pd.read_csv('train.csv')

df_stores_base_stores = pd.read_csv('store.csv')
df_stores = pd.read_csv('store.csv')

#  todo EXPLORATION ------------------------

'''Let's take a look at the datasets, df_sales and df_stores in that order'''

# PRINT 0

'''Right away after loading the dataset I received a warning, which indicates that we have mixed types of values'''

# PRINT 1

'''Let's check some basic info about the sales datasets'''

print(df_sales.info())
print(df_sales.describe())

# PRINT 2

'''Let's check the stores dataset'''

print(df_stores.info())
print(df_stores.describe())

# PRINT 3

'''I'm also checking for NaN values and some graphs so we can understand better distributions and other metrics'''

print(df_sales.isnull().sum())
print(df_stores.isnull().sum())

df_sales.hist(bins=30, figsize=(20, 20))
plt.show()

df_stores.hist(bins=30, figsize=(20, 20))
plt.show()

# PRINT 4

'''Let's split the days the stores were open and the days that the stores were closed, check the data and create a
dataset only for the days the stores were open, because that can affect the price we'll try to predict, after that
I'm dropping the Open column, since we're just selecting the day the stores we're open, also if we work only
with the days that the stores were open we can call the describe method again, because the data might have changed'''

closed_stores_df = df_sales[df_sales['Open'] == 0]
open_stores_df = df_sales[df_sales['Open'] == 1]

print(f'Days open: {len(open_stores_df)} -', f'{len(open_stores_df) / len(df_sales) * 100:.2f}%')
print(f'Days closed: {len(closed_stores_df)} -', f'{len(closed_stores_df) / len(df_sales) * 100:.2f}%')

df_sales = df_sales[df_sales['Open'] == 1]
df_sales.drop(columns=['Open'], inplace=True)

print(df_sales.describe())

# PRINT 5

'''We also seen before that we have NaN data and other problems with the stores dataset, and after checking it,
the most of the missing values are from stores that don't have info about the promos (because they're not
doing the promo in that store), so we'll have to replace it. And for the distance from other stores, we have
NaN values, but this ones we'll replace with the mean of the same columns'''

str_cols = ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'CompetitionOpenSinceYear',
            'CompetitionOpenSinceMonth']

for str in str_cols:
    df_stores[str].fillna(0, inplace=True)

df_stores['CompetitionDistance'] = df_stores['CompetitionDistance'].fillna(df_stores['CompetitionDistance'].mean())

sns.heatmap(df_stores.isnull())
plt.show()
df_stores.hist(bins=30, figsize=(20, 20))
plt.show()

'''Let's check our new histograms and the if still there's null values'''

# PRINT 6

'''Now I'll create a new dataframe with the stores and the sales dataset, using the Store, which is meant to
be like a primary key for union in this case, also I'll be looking at the correlation'''

df = pd.merge(df_sales, df_stores, how='inner', on='Store')
correlation = df.corr()
f, ax = plt.subplots(figsize=(25, 18))
sns.heatmap(correlation, annot=True)
plt.show()

'''We can see many infos, like the promo actually helps in the sales, so let's isolate this correlation comparison'''

print(df.corr()['Sales'].sort_values())

# PRINT 7

'''Let's split the data so we can try to see new insights, also I want to check the ScoreType so we can see
if there's difference between the types of the stores'''

df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Day'] = pd.DatetimeIndex(df['Date']).day

axis1 = df.groupby('Year')[['Sales']].mean().plot(figsize=(10, 5), marker='o')
axis1.set_title('Yearly average sales')
plt.show()
axis2 = df.groupby('Month')[['Sales']].mean().plot(figsize=(10, 5), marker='o')
axis2.set_title('Monthly average sales')
plt.show()
axis3 = df.groupby('Day')[['Sales']].mean().plot(figsize=(10, 5), marker='o')
axis3.set_title('Daily average sales')
plt.show()

fig, ax = plt.subplots(figsize=(20, 10))
df.groupby(['Date', 'StoreType']).mean()['Sales'].unstack().plot(ax=ax)
plt.show()

'''Stores of the type B sell more than the other stores, and type A stores is the ones that sells less'''

# PRINT 8

'''Since this data is seasonal and it was created thinking about holidays and weekdays, we're using Facebook
Prophet for the temporal series part. The results with multivariable recurrent network for this project was
roughly the same, at the end I'll apply a quick Holt Winters method so we can compare the results'''

#  todo PROPHET MODEL CREATION ------------------------

'''Now I'll create a function so we can wrap everything in one go, some alterations are obligatory by Prophet,
like Date column must be named "ds" and the prediction (Sales) must be named "y", we're also passing school and
state holidays so the model can give us more data about different occasions and take this info in consideration'''


def sales_pred(store_id, df_sales, holidays, periods):
    df_sales = df_sales[df_sales['Store'] == store_id]
    df_sales = df_sales[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    df_sales = df_sales.sort_values(by='ds')

    model = Prophet(holidays=holidays)
    model.fit(df_sales)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    figure1 = model.plot(forecast, xlabel='Data', ylabel='Sales')
    figure2 = model.plot_components(forecast)
    plt.show()
    return df_sales, forecast


'''Let's select the holidays dates so we can pass to the function'''

school_holidays = df_sales[df_sales['SchoolHoliday'] == 1].loc[:, 'Date'].values
state_holidays = df_sales[(df_sales['StateHoliday'] == 'a') |
                          (df_sales['StateHoliday'] == 'b') |
                          (df_sales['StateHoliday'] == 'c')].loc[:, 'Date'].values

school_holidays = pd.DataFrame({'ds': pd.to_datetime(school_holidays), 'holiday': 'school_holiday'})
state_holidays = pd.DataFrame({'ds': pd.to_datetime(state_holidays), 'holiday': 'state_holiday'})

school_state = pd.concat((state_holidays, school_holidays))

df_original, df_prediction = sales_pred(store_id=10, df_sales=df, holidays=school_state, periods=60)

'''Now let's check the dataframe with the predictions and some graphs'''

prediction_final = df_prediction.tail(60)

# PRINT 9

'''Now I'll take a step back and work in a new dataset and create a daily prediction using Holt Winters method'''

#  todo HOLT MODEL CREATION ------------------------

df_holtz = df[df['Store'] == 1]
df_holtz = df_holtz.loc[:, ['Date', 'Sales']]
df_holtz.Date = df.Date.astype(np.datetime64)
df_holtz.Date = pd.to_datetime(df_holtz.Date).dt.to_period('d')
df_holtz = df_holtz.groupby('Date').sum()
df_holtz.dropna(axis=0, inplace=True)

es = ExponentialSmoothing(df_holtz, seasonal_periods=365, trend='additive', seasonal='additive').fit()

plt.figure(figsize=(12, 8))
es.fittedvalues.plot(style='--', color='pink')
plt.legend(['Past Sales'])
plt.show()

plt.figure(figsize=(12, 8))
es.forecast(90).plot(style='--', marker='o', color='red', legend=True)
plt.legend(['Predictions'])
plt.show()

plt.figure(figsize=(12, 8))
es.fittedvalues.plot(style='--', color='pink')
es.forecast(90).plot(marker='o', color='red', legend=True)
plt.legend(['Past Sales', 'Predictions'])
plt.show()

forecast_final = es.forecast(90)

# PRINT 10

#  todo SAVE ------------------------

'''and we're also saving the datasets for a deeper analysis by the sales team'''

prediction_final.to_csv('Prediction_stores_prophet.csv')
forecast_final.to_csv('Prediction_stores_holt.csv')