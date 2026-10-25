import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_data(file_type: str):
    if file_type == 'parquet':
        df1 = pd.read_parquet(BASE_DIR / 'data' / 'raw' / 'yellow_tripdata_2025-01 (2).parquet')
        return df1
    elif file_type == 'csv':
        df2 = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'taxi_zone_lookup.csv')
        return df2
    raise ValueError('Введите правильный формат файла paquet/csv')

df = load_data('parquet')

df['duration_trip'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
df['duration_trip_in_mins'] = round(df['duration_trip'].dt.total_seconds() / 60, 2)

DF = df[(df['duration_trip_in_mins'] > 0) & (df['duration_trip_in_mins'] <= 180)].copy()

DF['pickup_hour'] = DF['tpep_pickup_datetime'].dt.hour
DF['pickup_weekday'] = DF['tpep_pickup_datetime'].dt.day_of_week

DF = DF[DF['tpep_pickup_datetime'].dt.month < 2]
DF['tpep_pickup_datetime'] = pd.to_datetime(DF['tpep_pickup_datetime'])
DF = DF.sort_values(by='tpep_pickup_datetime').reset_index(drop=True)

df_places = load_data('csv')

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

DF = res

columns_for_X = [
    'PULocationID',
    'DOLocationID',
    'pickup_hour',
    'pickup_weekday',
    'Borough_pickup',
    'Borough_dropoff'
    ]

X = DF[columns_for_X]
y = DF['duration_trip_in_mins']

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

preprocessor = ColumnTransformer(
    transformers=[
        (
            'borough',
            OneHotEncoder(handle_unknown='ignore'),
            ['Borough_pickup', 'Borough_dropoff']
        )
    ],
    remainder='passthrough'
)

baseline = DummyRegressor(strategy='median')
baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_val)
MAE = mean_absolute_error(y_true=y_val, y_pred=y_pred)
print('Baseline MAE:', MAE)

model_tree = DecisionTreeRegressor(
    max_depth=30,
    min_samples_leaf=6
    )

model_tree = make_pipeline(
    clone(preprocessor),
    model_tree
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
    max_features=1.0,
    max_depth=30,
    criterion='squared_error',
    random_state=42
    )

model_random_forest = make_pipeline(
    clone(preprocessor),
    model_random_forest
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

y_pred_final = model_random_forest.predict(X_test)
MAE_tree_rand_final = mean_absolute_error(y_true=y_test, y_pred=y_pred_final)
print('RandomForestRegressor MAE on TEST:', MAE_tree_rand_final)

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

print(grouped_erros)

df_errors['signed_error'] = df_errors['real_duration'] - df_errors['model_predict']
mean_of_every_value = df_errors.groupby('category_of_trip')['signed_error'].mean()

print(mean_of_every_value)


joblib.dump(
    model_random_forest,
    BASE_DIR / 'models' / 'ridepulse_rf.joblib'
)