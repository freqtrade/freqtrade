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


class _KeyValueStoreModel(ModelBase):
    """
    Pair Locks database model.
    """
    __tablename__ = 'KeyValueStore'
    session: ClassVar[SessionType]

    id: Mapped[int] = mapped_column(primary_key=True)

    key: Mapped[str] = mapped_column(String(25), nullable=False, index=True)

    value_type: Mapped[ValueTypesEnum] = mapped_column(String(25), nullable=False)

    string_value: Mapped[Optional[str]]
    datetime_value: Mapped[Optional[datetime]]
    float_value: Mapped[Optional[float]]
    int_value: Mapped[Optional[int]]


class KeyValueStore():
    """
    Generic bot-wide, persistent key-value store
    Can be used to store generic values, e.g. very first bot startup time.
    Supports the types str, datetime, float and int.
    """

    @staticmethod
    def get_value(key: str) -> Optional[ValueTypes]:
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
    def store_value(key: str, value: ValueTypes) -> None:
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
    def delete_value(key: str) -> None:
        """
        Delete the value for the given key.
        :param key: Key to delete the value for
        """
        kv = _KeyValueStoreModel.session.query(_KeyValueStoreModel).filter(
            _KeyValueStoreModel.key == key).first()
        if kv is not None:
            _KeyValueStoreModel.session.delete(kv)
            _KeyValueStoreModel.session.commit()
