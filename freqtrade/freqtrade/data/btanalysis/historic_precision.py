from pandas import DataFrame, Series


def get_tick_size_over_time(candles: DataFrame) -> Series:
    """
    Calculate the number of significant digits for candles over time.
    It's using the Monthly maximum of the number of significant digits for each month.
    :param candles: DataFrame with OHLCV data
    :return: Series with the average number of significant digits for each month
    """
    # count the number of significant digits for the open and close prices
    for col in ["open", "high", "low", "close"]:
        candles[f"{col}_count"] = (
            candles[col].round(14).apply("{:.15f}".format).str.extract(r"\.(\d*[1-9])")[0].str.len()
        )
    candles["max_count"] = candles[["open_count", "close_count", "high_count", "low_count"]].max(
        axis=1
    )

    candles1 = candles.set_index("date", drop=True)
    # Group by month and calculate the average number of significant digits
    monthly_count_avg1 = candles1["max_count"].resample("MS").max()
    # monthly_open_count_avg
    # convert monthly_open_count_avg from 5.0 to 0.00001, 4.0 to 0.0001, ...
    monthly_open_count_avg = 1 / 10**monthly_count_avg1

    return monthly_open_count_avg
