from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Optional, Union

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from freqtrade.persistence.base import ModelBase, SessionType


ValueTypes = Union[str, datetime, float, int]


class ValueTypesEnum(str, Enum):
    STRING = 'str'
    DATETIME = 'datetime'
    FLOAT = 'float'
    INT = 'int'


class KeyStoreKeys(str, Enum):
    BOT_START_TIME = 'bot_start_time'
    STARTUP_TIME = 'startup_time'


class _KeyValueStoreModel(ModelBase):
    """
    Pair Locks database model.
    """
    __tablename__ = 'KeyValueStore'
    session: ClassVar[SessionType]

    id: Mapped[int] = mapped_column(primary_key=True)

    key: Mapped[KeyStoreKeys] = mapped_column(String(25), nullable=False, index=True)

    value_type: Mapped[ValueTypesEnum] = mapped_column(String(20), nullable=False)

    string_value: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    datetime_value: Mapped[Optional[datetime]]
    float_value: Mapped[Optional[float]]
    int_value: Mapped[Optional[int]]


class KeyValueStore:
    """
    Generic bot-wide, persistent key-value store
    Can be used to store generic values, e.g. very first bot startup time.
    Supports the types str, datetime, float and int.
    """

    @staticmethod
    def store_value(key: KeyStoreKeys, value: ValueTypes) -> None:
        """
        Store the given value for the given key.
        :param key: Key to store the value for - can be used in get-value to retrieve the key
        :param value: Value to store - can be str, datetime, float or int
        """
        kv = _KeyValueStoreModel.session.query(_KeyValueStoreModel).filter(
            _KeyValueStoreModel.key == key).first()
        if kv is None:
            kv = _KeyValueStoreModel(key=key)
        if isinstance(value, str):
            kv.value_type = ValueTypesEnum.STRING
            kv.string_value = value
        elif isinstance(value, datetime):
            kv.value_type = ValueTypesEnum.DATETIME
            kv.datetime_value = value
        elif isinstance(value, float):
            kv.value_type = ValueTypesEnum.FLOAT
            kv.float_value = value
        elif isinstance(value, int):
            kv.value_type = ValueTypesEnum.INT
            kv.int_value = value
        else:
            raise ValueError(f'Unknown value type {kv.value_type}')
        _KeyValueStoreModel.session.add(kv)
        _KeyValueStoreModel.session.commit()

    @staticmethod
    def delete_value(key: KeyStoreKeys) -> None:
        """
        Delete the value for the given key.
        :param key: Key to delete the value for
        """
        kv = _KeyValueStoreModel.session.query(_KeyValueStoreModel).filter(
            _KeyValueStoreModel.key == key).first()
        if kv is not None:
            _KeyValueStoreModel.session.delete(kv)
            _KeyValueStoreModel.session.commit()

    @staticmethod
    def get_value(key: KeyStoreKeys) -> Optional[ValueTypes]:
        """
        Get the value for the given key.
        :param key: Key to get the value for
        """
        kv = _KeyValueStoreModel.session.query(_KeyValueStoreModel).filter(
            _KeyValueStoreModel.key == key).first()
        if kv is None:
            return None
        if kv.value_type == ValueTypesEnum.STRING:
            return kv.string_value
        if kv.value_type == ValueTypesEnum.DATETIME and kv.datetime_value is not None:
            return kv.datetime_value.replace(tzinfo=timezone.utc)
        if kv.value_type == ValueTypesEnum.FLOAT:
            return kv.float_value
        if kv.value_type == ValueTypesEnum.INT:
            return kv.int_value
        # This should never happen unless someone messed with the database manually
        raise ValueError(f'Unknown value type {kv.value_type}')  # pragma: no cover

    @staticmethod
    def get_string_value(key: KeyStoreKeys) -> Optional[str]:
        """
        Get the value for the given key.
        :param key: Key to get the value for
        """
        kv = _KeyValueStoreModel.session.query(_KeyValueStoreModel).filter(
            _KeyValueStoreModel.key == key,
            _KeyValueStoreModel.value_type == ValueTypesEnum.STRING).first()
        if kv is None:
            return None
        return kv.string_value

    @staticmethod
    def get_datetime_value(key: KeyStoreKeys) -> Optional[datetime]:
        """
        Get the value for the given key.
        :param key: Key to get the value for
        """
        kv = _KeyValueStoreModel.session.query(_KeyValueStoreModel).filter(
            _KeyValueStoreModel.key == key,
            _KeyValueStoreModel.value_type == ValueTypesEnum.DATETIME).first()
        if kv is None or kv.datetime_value is None:
            return None
        return kv.datetime_value.replace(tzinfo=timezone.utc)

    @staticmethod
    def get_float_value(key: KeyStoreKeys) -> Optional[float]:
        """
        Get the value for the given key.
        :param key: Key to get the value for
        """
        kv = _KeyValueStoreModel.session.query(_KeyValueStoreModel).filter(
            _KeyValueStoreModel.key == key,
            _KeyValueStoreModel.value_type == ValueTypesEnum.FLOAT).first()
        if kv is None:
            return None
        return kv.float_value

    @staticmethod
    def get_int_value(key: KeyStoreKeys) -> Optional[int]:
        """
        Get the value for the given key.
        :param key: Key to get the value for
        """
        kv = _KeyValueStoreModel.session.query(_KeyValueStoreModel).filter(
            _KeyValueStoreModel.key == key,
            _KeyValueStoreModel.value_type == ValueTypesEnum.INT).first()
        if kv is None:
            return None
        return kv.int_value


def set_startup_time():
    """
    sets bot_start_time to the first trade open date - or "now" on new databases.
    sets startup_time to "now"
    """
    st = KeyValueStore.get_value('bot_start_time')
    if st is None:
        from freqtrade.persistence import Trade
        t = Trade.session.query(Trade).order_by(Trade.open_date.asc()).first()
        if t is not None:
            KeyValueStore.store_value('bot_start_time', t.open_date_utc)
        else:
            KeyValueStore.store_value('bot_start_time', datetime.now(timezone.utc))
    KeyValueStore.store_value('startup_time', datetime.now(timezone.utc))
