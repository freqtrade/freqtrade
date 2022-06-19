import json
import logging
from datetime import datetime
from typing import Any, List, Optional

from freqtrade.persistence.custom_data import CustomData


logger = logging.getLogger(__name__)


class CustomDataWrapper():
    """
    CustomData middleware class
    Abstracts the database layer away so it becomes optional - which will be necessary to support
    backtesting and hyperopt in the future.
    """

    use_db = True
    custom_data: List[CustomData] = []
    unserialized_types = ['bool', 'float', 'int', 'str']

    @staticmethod
    def reset_custom_data() -> None:
        """
        Resets all key-value pairs. Only active for backtesting mode.
        """
        if not CustomDataWrapper.use_db:
            CustomDataWrapper.custom_data = []

    @staticmethod
    def get_custom_data(key: Optional[str] = None,
                        trade_id: Optional[int] = None) -> List[CustomData]:
        if trade_id is None:
            trade_id = 0

        if CustomDataWrapper.use_db:
            filtered_custom_data = CustomData.query_cd(trade_id=trade_id, key=key).all()
            for index, data_entry in enumerate(filtered_custom_data):
                if data_entry.cd_type not in CustomDataWrapper.unserialized_types:
                    data_entry.cd_value = json.loads(data_entry.cd_value)
                filtered_custom_data[index] = data_entry
            return filtered_custom_data
        else:
            filtered_custom_data = [
                data_entry for data_entry in CustomDataWrapper.custom_data
                if (data_entry.ft_trade_id == trade_id)
            ]
            if key is not None:
                filtered_custom_data = [
                    data_entry for data_entry in filtered_custom_data
                    if (data_entry.cd_key.casefold() == key.casefold())
                ]
            return filtered_custom_data

    @staticmethod
    def set_custom_data(key: str, value: Any, trade_id: Optional[int] = None) -> None:

        value_type = type(value).__name__
        value_db = None

        if value_type not in CustomDataWrapper.unserialized_types:
            try:
                value_db = json.dumps(value)
            except TypeError as e:
                logger.warning(f"could not serialize {key} value due to {e}")
        else:
            value_db = str(value)

        if trade_id is None:
            trade_id = 0

        custom_data = CustomDataWrapper.get_custom_data(key=key, trade_id=trade_id)
        if custom_data:
            data_entry = custom_data[0]
            data_entry.cd_value = value
            data_entry.updated_at = datetime.utcnow()
        else:
            data_entry = CustomData(
                            ft_trade_id=trade_id,
                            cd_key=key,
                            cd_type=value_type,
                            cd_value=value,
                            created_at=datetime.utcnow()
            )

        if CustomDataWrapper.use_db and value_db is not None:
            data_entry.cd_value = value_db
            CustomData.query.session.add(data_entry)
            CustomData.query.session.commit()
        elif not CustomDataWrapper.use_db:
            cd_index = -1
            for index, data_entry in enumerate(CustomDataWrapper.custom_data):
                if data_entry.ft_trade_id == trade_id and data_entry.cd_key == key:
                    cd_index = index
                    break

            if cd_index >= 0:
                data_entry.cd_type = value_type
                data_entry.value = value
                data_entry.updated_at = datetime.utcnow()

                CustomDataWrapper.custom_data[cd_index] = data_entry
            else:
                CustomDataWrapper.custom_data.append(data_entry)

    @staticmethod
    def get_all_custom_data() -> List[CustomData]:

        if CustomDataWrapper.use_db:
            return CustomData.query.all()
        else:
            return CustomDataWrapper.custom_data
