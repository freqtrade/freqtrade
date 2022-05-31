import json
import logging
from datetime import datetime
from typing import Any, List, Optional

from freqtrade.persistence.keyvalue import KeyValue


logger = logging.getLogger(__name__)


class KeyValues():
    """
    KeyValues middleware class
    Abstracts the database layer away so it becomes optional - which will be necessary to support
    backtesting and hyperopt in the future.
    """

    use_db = True
    kvals: List[KeyValue] = []
    unserialized_types = ['bool', 'float', 'int', 'str']

    @staticmethod
    def reset_keyvalues() -> None:
        """
        Resets all key-value pairs. Only active for backtesting mode.
        """
        if not KeyValues.use_db:
            KeyValues.kvals = []

    @staticmethod
    def get_kval(key: Optional[str] = None, trade_id: Optional[int] = None) -> List[KeyValue]:
        if trade_id is None:
            trade_id = 0

        if KeyValues.use_db:
            filtered_kvals = KeyValue.query_kv(trade_id=trade_id, key=key).all()
            for index, kval in enumerate(filtered_kvals):
                if kval.kv_type not in KeyValues.unserialized_types:
                    kval.kv_value = json.loads(kval.kv_value)
                filtered_kvals[index] = kval
            return filtered_kvals
        else:
            filtered_kvals = [kval for kval in KeyValues.kvals if (kval.ft_trade_id == trade_id)]
            if key is not None:
                filtered_kvals = [
                    kval for kval in filtered_kvals if (kval.kv_key.casefold() == key.casefold())]
            return filtered_kvals

    @staticmethod
    def set_kval(key: str, value: Any, trade_id: Optional[int] = None) -> None:

        value_type = type(value).__name__
        value_db = None

        if value_type not in KeyValues.unserialized_types:
            try:
                value_db = json.dumps(value)
            except TypeError as e:
                logger.warning(f"could not serialize {key} value due to {e}")
        else:
            value_db = str(value)

        if trade_id is None:
            trade_id = 0

        kvals = KeyValues.get_kval(key=key, trade_id=trade_id)
        if kvals:
            kv = kvals[0]
            kv.kv_value = value
            kv.updated_at = datetime.utcnow()
        else:
            kv = KeyValue(
                ft_trade_id=trade_id,
                kv_key=key,
                kv_type=value_type,
                kv_value=value,
                created_at=datetime.utcnow()
            )

        if KeyValues.use_db and value_db is not None:
            kv.kv_value = value_db
            KeyValue.query.session.add(kv)
            KeyValue.query.session.commit()
        elif not KeyValues.use_db:
            kv_index = -1
            for index, kval in enumerate(KeyValues.kvals):
                if kval.ft_trade_id == trade_id and kval.kv_key == key:
                    kv_index = index
                    break

            if kv_index >= 0:
                kval.kv_type = value_type
                kval.value = value
                kval.updated_at = datetime.utcnow()

                KeyValues.kvals[kv_index] = kval
            else:
                KeyValues.kvals.append(kv)

    @staticmethod
    def get_all_kvals() -> List[KeyValue]:

        if KeyValues.use_db:
            return KeyValue.query.all()
        else:
            return KeyValues.kvals
