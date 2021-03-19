from pandas import read_csv
from tsfresh import extract_relevant_features


dataframe = read_csv('/home/yakov/PycharmProjects/freqtrade/.env/bin/eth_1h_safe_features.csv').reset_index().dropna()
features_filtered_direct = extract_relevant_features(dataframe, dataframe['close'],
                                                     column_id='index', column_sort='date')
print(features_filtered_direct)