from datetime import datetime, timedelta, timezone

import pytest

from freqtrade.persistence.key_value_store import KeyValueStore, set_startup_time
from tests.conftest import create_mock_trades_usdt


@pytest.mark.usefixtures("init_persistence")
def test_key_value_store(time_machine):
    start = datetime(2023, 1, 1, 4, tzinfo=timezone.utc)
    time_machine.move_to(start, tick=False)

    KeyValueStore.store_value("test", "testStringValue")
    KeyValueStore.store_value("test_dt", datetime.now(timezone.utc))
    KeyValueStore.store_value("test_float", 22.51)
    KeyValueStore.store_value("test_int", 15)

    assert KeyValueStore.get_value("test") == "testStringValue"
    assert KeyValueStore.get_value("test") == "testStringValue"
    assert KeyValueStore.get_string_value("test") == "testStringValue"
    assert KeyValueStore.get_value("test_dt") == datetime.now(timezone.utc)
    assert KeyValueStore.get_datetime_value("test_dt") == datetime.now(timezone.utc)
    assert KeyValueStore.get_string_value("test_dt") is None
    assert KeyValueStore.get_float_value("test_dt") is None
    assert KeyValueStore.get_int_value("test_dt") is None
    assert KeyValueStore.get_value("test_float") == 22.51
    assert KeyValueStore.get_float_value("test_float") == 22.51
    assert KeyValueStore.get_value("test_int") == 15
    assert KeyValueStore.get_int_value("test_int") == 15
    assert KeyValueStore.get_datetime_value("test_int") is None

    time_machine.move_to(start + timedelta(days=20, hours=5), tick=False)
    assert KeyValueStore.get_value("test_dt") != datetime.now(timezone.utc)
    assert KeyValueStore.get_value("test_dt") == start
    # Test update works
    KeyValueStore.store_value("test_dt", datetime.now(timezone.utc))
    assert KeyValueStore.get_value("test_dt") == datetime.now(timezone.utc)

    KeyValueStore.store_value("test_float", 23.51)
    assert KeyValueStore.get_value("test_float") == 23.51
    # test deleting
    KeyValueStore.delete_value("test_float")
    assert KeyValueStore.get_value("test_float") is None
    # Delete same value again (should not fail)
    KeyValueStore.delete_value("test_float")

    with pytest.raises(ValueError, match=r"Unknown value type"):
        KeyValueStore.store_value("test_float", {'some': 'dict'})


@pytest.mark.usefixtures("init_persistence")
def test_set_startup_time(fee, time_machine):
    create_mock_trades_usdt(fee)
    start = datetime.now(timezone.utc)
    time_machine.move_to(start, tick=False)
    set_startup_time()

    assert KeyValueStore.get_value("startup_time") == start
    initial_time = KeyValueStore.get_value("bot_start_time")
    assert initial_time <= start

    # Simulate bot restart
    new_start = start + timedelta(days=5)
    time_machine.move_to(new_start, tick=False)
    set_startup_time()

    assert KeyValueStore.get_value("startup_time") == new_start
    assert KeyValueStore.get_value("bot_start_time") == initial_time
