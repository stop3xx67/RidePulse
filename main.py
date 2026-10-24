import pandas as pd
import time

df = pd.read_parquet('D:/VS Code/ridepulse/data/raw/yellow_tripdata_2025-01 (2).parquet')

df['duration_trip'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
df['duration_trip_in_mins'] = round(df['duration_trip'].dt.total_seconds() / 60, 2)

print('Общее кол-во поедздок -', df.shape[0])
print('Кол-во поездок с длительностью меньше или равно 0 (аномалии) -', df[df['duration_trip_in_mins'] <= 0].shape[0])

print('Поездок с отрицательной длительностью -', df[df['duration_trip_in_mins'] < 0].shape[0])
print('Поездок с нулевой длительностью -', df[df['duration_trip_in_mins'] == 0].shape[0])

print('Поездок с длительностью больше 180 минут -', df[df['duration_trip_in_mins'] > 180].shape[0])
print('Их процент от общего числа поездок -', round(df[df['duration_trip_in_mins'] > 180].shape[0] / df['duration_trip_in_mins'].shape[0], 5) * 100, '%')

print('Первые 10 поездок дольше 180 минут -', df[df['duration_trip_in_mins'] > 180].head(10))
print('Их расстояние\n', df[df['duration_trip_in_mins'] > 180]['trip_distance'])

print('Дольше 180 минут но расстояние не пройдено -', df[(df['duration_trip_in_mins'] > 180) & (df['trip_distance'] == 0)].shape[0])
print('Их медианная длительность -', df[df['duration_trip_in_mins'] > 180]['duration_trip_in_mins'].median())

print('Сколько поездок длительностью больше 180 минут заканчиваются ровно в 00:00:00 -', df[(df['duration_trip_in_mins'] > 180) & (df['tpep_dropoff_datetime'].dt.strftime('%H:%M:%S') == '00:00:00')].shape[0])
