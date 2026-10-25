import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_parquet('D:/VS Code/ridepulse/data/raw/yellow_tripdata_2025-01 (2).parquet')

df['duration_trip'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
df['duration_trip_in_mins'] = round(df['duration_trip'].dt.total_seconds() / 60, 2)

#print('Общее кол-во поедздок -', df.shape[0])
#print('Кол-во поездок с длительностью меньше или равно 0 (аномалии) -', df[df['duration_trip_in_mins'] <= 0].shape[0])

#print('Поездок с отрицательной длительностью -', df[df['duration_trip_in_mins'] < 0].shape[0])
#print('Поездок с нулевой длительностью -', df[df['duration_trip_in_mins'] == 0].shape[0])

#print('Поездок с длительностью больше 180 минут -', df[df['duration_trip_in_mins'] > 180].shape[0])
#print('Их процент от общего числа поездок -', round(df[df['duration_trip_in_mins'] > 180].shape[0] / df['duration_trip_in_mins'].shape[0], 5) * 100, '%')

#print('Первые 10 поездок дольше 180 минут -', df[df['duration_trip_in_mins'] > 180].head(10))
#print('Их расстояние\n', df[df['duration_trip_in_mins'] > 180]['trip_distance'])

#print('Дольше 180 минут но расстояние не пройдено -', df[(df['duration_trip_in_mins'] > 180) & (df['trip_distance'] == 0)].shape[0])
#print('Их медианная длительность -', df[df['duration_trip_in_mins'] > 180]['duration_trip_in_mins'].median())

#print('Сколько поездок длительностью больше 180 минут заканчиваются ровно в 00:00:00 -', df[(df['duration_trip_in_mins'] > 180) & (df['tpep_dropoff_datetime'].dt.strftime('%H:%M:%S') == '00:00:00')].shape[0])

#print('===================================================================')

DF = df[(df['duration_trip_in_mins'] > 0) & (df['duration_trip_in_mins'] <= 180)].copy()

row_before = df.shape[0]
row_now = DF.shape[0]
#print('Строк отфильтровано после удаления временных аномалий в датасете -', row_before - row_now)

columns = [
    'PULocationID',
    'DOLocationID',
    'tpep_pickup_datetime',
    'duration_trip_in_mins'
]

#print('Пропуски в данных для обучения модели -', DF[columns].isna().sum())

DF['pickup_hour'] = DF['tpep_pickup_datetime'].dt.hour
DF['pickup_weekday'] = DF['tpep_pickup_datetime'].dt.day_of_week

#print('Макс и мин значение часа -', DF['pickup_hour'].max(),DF['pickup_hour'].min())
#print('Макс и мин значение недели -', DF['pickup_weekday'].max(), DF['pickup_weekday'].min())

#print('Проверим пропуски в часах и неделях...')

days_and_weeks = [
    'pickup_hour',
    'pickup_weekday'
    ]

#print(DF[days_and_weeks].isna().sum())

df_places = pd.read_csv('D:/VS Code/ridepulse/data/raw/taxi_zone_lookup.csv')

#print('Уникальных LocationID -', len(df_places['LocationID'].unique()))

mask1 = DF['PULocationID'].isin(df_places['LocationID'])
mask2 = DF['DOLocationID'].isin(df_places['LocationID'])

m1 = ~mask1
m2 = ~mask2

#print(sum(m1))
#print(sum(m2))

DF = DF[DF['tpep_pickup_datetime'].dt.month < 2]
DF['tpep_pickup_datetime'] = pd.to_datetime(DF['tpep_pickup_datetime'])
DF = DF.sort_values(by='tpep_pickup_datetime').reset_index(drop=True)

#print(DF['tpep_pickup_datetime'].dt.month.max())
#print(DF['tpep_pickup_datetime'].is_monotonic_increasing)

columns_for_X = [
    'PULocationID',
    'DOLocationID',
    'pickup_hour',
    'pickup_weekday'
    ]

X = DF[columns_for_X]
y = DF['duration_trip_in_mins']

#print(X.shape)
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    shuffle=False
    )

X_val, X_test, y_val, y_test = train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    shuffle=False
    )

#print(X_train.shape)
#print(X_val.shape)
#print(X_test.shape)

baseline = DummyRegressor(strategy='median')
baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_val)
MAE = mean_absolute_error(y_true=y_val, y_pred=y_pred)

print('Baseline MAE:', MAE)

model_tree = DecisionTreeRegressor(
    max_depth=30,
    min_samples_leaf=6
    )

model_tree.fit(X_train, y_train)

y_pred = model_tree.predict(X_val)
MAE_tree = mean_absolute_error(y_true=y_val, y_pred=y_pred)
print('DecisionTreeRegressor MAE on Validation:', MAE_tree)

y_pred2 = model_tree.predict(X_train)
MAE_tree2 = mean_absolute_error(y_true=y_train, y_pred=y_pred2)
print('DecisionTreeRegressor MAE on Train:', MAE_tree2)

model_random_forest = RandomForestRegressor(
    n_estimators=30,
    bootstrap=True,
    min_samples_leaf=6,
    max_samples=1.0,
    max_features=0.75,
    max_depth=30,
    criterion='squared_error',
    random_state=42
    )

model_random_forest.fit(X_train, y_train)

y_pred_rand = model_random_forest.predict(X_val)
MAE_tree_rand = mean_absolute_error(y_true=y_val, y_pred=y_pred_rand)
RMSE_tree_rand = root_mean_squared_error(y_true=y_val, y_pred=y_pred_rand)
R2_tree_rand = r2_score(y_true=y_val, y_pred=y_pred_rand)
print('RandomForestRegressor MAE on Validation:', MAE_tree_rand)
print('RandomForestRegressor RMSE on Validation:', RMSE_tree_rand)
print('RandomForestRegressor R2 on Validation:', R2_tree_rand)

y_pred_rand2 = model_random_forest.predict(X_train)
MAE_tree_rand2 = mean_absolute_error(y_true=y_train, y_pred=y_pred_rand2)
print('RandomForestRegressor MAE on Train:', MAE_tree_rand2)


df_errors = pd.DataFrame({
    'real_duration': y_val,
    'model_predict': y_pred_rand 
    })

df_errors['absolute_error'] = (df_errors['real_duration'] - df_errors['model_predict']).abs()

print(df_errors.head(10))

min_bins = [0, 10, 30, float('inf')]
mins = ['short_trip', 'usual_trip', 'long_trip']

df_errors['category_of_trip'] = pd.cut(
    df_errors['real_duration'],
    bins=min_bins,
    labels=mins
)

grouped_erros = df_errors.groupby('category_of_trip')['absolute_error'].agg(['mean', 'count', 'max', 'min'])

print(grouped_erros.head(10))

df_errors['signed_error'] = df_errors['real_duration'] - df_errors['model_predict']
mean_of_every_value = df_errors.groupby('category_of_trip')['signed_error'].mean()

print(mean_of_every_value.head(10))

res = pd.merge(DF,
               df_places,
               left_on='PULocationID',
               right_on='LocationID',
               how='left'
               )

res = pd.merge(
    res,
    df_places,
    left_on='DOLocationID',
    right_on='LocationID',
    how='left',
    suffixes=(
        '_pickup',
        '_dropoff'
        )   
    )

DF['Borough_pickup'], DF['Borough_dropoff'] = res['Borough_pickup'], res['Borough_dropoff']

print(DF.shape)
print(res.shape)







