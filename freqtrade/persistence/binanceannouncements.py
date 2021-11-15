# -*- coding: utf-8 -*-
import pandas as pd
import pytz

from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.types import DateTime


def get_engine(uri: str):
    return create_engine(uri, pool_recycle=3600)


def get_connection(uri: str):
    return get_engine(uri).connect()


def get_df(uri, table_name):
    """Get dataframe and the first time create DB."""
    connection = get_connection(uri)
    try:
        return pd.read_sql_table(
            table_name=table_name,
            con=connection,
            index_col='index',
            columns=['Token', 'Text', 'Link', 'Datetime discover', 'Datetime announcement'],
        )
    except ValueError as e:
        return None
    finally:
        connection.close()


def save_df(df, uri, table_name):
    """Save dataframe on DB."""
    connection = get_connection(uri)
    try:
        df.to_sql(
            name=table_name,
            con=connection,
            index=True,
            index_label='index',
            if_exists='replace',
            dtype={"Datetime discover": DateTime(timezone=pytz.utc),
                   "Datetime announcement": DateTime(timezone=pytz.utc)}
        )
    finally:
        connection.close()
