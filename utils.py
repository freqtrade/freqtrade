
import sqlite3

import pandas as pd


#Pivot Points, Supports and Resistances
def PPSR(df):
    PP = pd.Series((df['high'] + df['low'] + df['close']) / 3)
    R1 = pd.Series(2 * PP - df['low'])
    S1 = pd.Series(2 * PP - df['high'])
    R2 = pd.Series(PP + df['high'] - df['low'])
    S2 = pd.Series(PP - df['high'] + df['low'])
    R3 = pd.Series(df['high'] + 2 * (PP - df['low']))
    S3 = pd.Series(df['low'] - 2 * (df['high'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = pd.DataFrame(psr)
    df = df.join(PSR)
    return df

#Camarilla Pivot Points
def camarilla_pp(df):
    PP = pd.Series((df['high'] + df['low'] + df['close']) / 3)
    R1 = pd.Series(2 * PP - df['low'])
    S1 = pd.Series(2 * PP - df['high'])
    R2 = pd.Series(PP + df['high'] - df['low'])
    S2 = pd.Series(PP - df['high'] + df['low'])
    R3 = pd.Series(df['high'] + 2 * (PP - df['low']))
    S3 = pd.Series(df['low'] - 2 * (df['high'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = pd.DataFrame(psr)
    df = df.join(PSR)
    return df


def get_best_run_by_total_profit(pkl_path):
    df = pd.read_pickle(pkl_path)

    total_profit = 0
    i = 0
    best = None
    for res in df:
        i += 1
        if res['loss'] >= 5:
            continue
        if total_profit < res['results_metrics']['total_profit']:
            total_profit = res['results_metrics']['total_profit']
            if res['results_metrics']['trade_count'] < 100:
                continue
            best = res
    return best

def get_best_from_pkl(pkl_path):
    df = pd.read_pickle(pkl_path)
    return [strg for strg in df if strg['is_best']][0]


def get_most_trades(pkl_path):
    # "/home/yakov/PycharmProjects/freqtrade/.env/bin/user_data/hyperopt_results/hyperopt_results_2021-01-30_13-15-48.pickle"

    df = pd.read_pickle(pkl_path)

    total_trades = 0
    i = 0
    best = None
    for res in df:
        i += 1
        if res['loss'] >= 5:
            continue
        if res['results_metrics']['trade_count'] < 100:
            continue
        if total_trades < res['results_metrics']['trade_count']:
            total_trades = res['results_metrics']['trade_count']

            best = res
    return best


def get_trade_from_db():
    # Read sqlite query results into a pandas DataFrame
    con = sqlite3.connect("/home/yakov/PycharmProjects/freqtrade/.env/bin/user_data/ethusdt_19022020.sqlite")
    df = pd.read_sql_query("SELECT * from trades", con)
    # Verify that result of SQL query is stored in the dataframe
    con.close()





b = get_best_run_by_total_profit(
    pkl_path="/home/yakov/PycharmProjects/freqtrade/.env/bin/user_data/hyperopt_results/ethusdt2_safe_sharpe_2021-02-28_20-54-24.pickle")
print(b)