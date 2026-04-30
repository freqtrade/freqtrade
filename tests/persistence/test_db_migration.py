from unittest.mock import MagicMock

from freqtrade.persistence.base import ModelBase
from freqtrade.persistence.custom_data import _CustomData
from freqtrade.persistence.db_migration import migrate_db
from freqtrade.persistence.key_value_store import _KeyValueStoreModel
from freqtrade.persistence.pairlock import PairLock
from freqtrade.persistence.trade_model import Trade
from freqtrade.persistence.wallet_history import WalletHistory


def test_migrate_db_detail(mocker):
    # Expected models to be migrated based on the registered models
    expected_models = {mapper.class_.__name__ for mapper in ModelBase.registry.mappers}
    session_target = MagicMock()

    order = MagicMock()
    trade = MagicMock(orders=[order])
    pairlock = MagicMock()
    kv = MagicMock()
    custom_data = MagicMock()
    wallet_history = MagicMock()

    kv_session = MagicMock()
    kv_session.scalars.return_value = [kv]
    custom_data_session = MagicMock()
    custom_data_session.scalars.return_value = [custom_data]
    wallet_history_session = MagicMock()
    wallet_history_session.scalars.return_value = [wallet_history]

    mocker.patch.object(Trade, "get_trades", return_value=[trade])
    mocker.patch.object(PairLock, "get_all_locks", return_value=[pairlock])
    mocker.patch.object(_KeyValueStoreModel, "session", kv_session, create=True)
    mocker.patch.object(_CustomData, "session", custom_data_session, create=True)
    mocker.patch.object(WalletHistory, "session", wallet_history_session, create=True)

    make_transient_mock = mocker.patch("freqtrade.persistence.db_migration.make_transient")
    set_sequence_ids_mock = mocker.patch("freqtrade.persistence.db_migration.set_sequence_ids")

    # max ids for Trade, Order, PairLock, KeyValueStore, CustomData, WalletHistory
    session_target.scalar.side_effect = [10, 11, 12, 13, 14, 15]
    session_target.get_bind.return_value = "bind"

    migrate_db(session_target)

    assert session_target.add.call_count == 5
    # Order objects are linked to trades, so they are not added explicitly

    assert session_target.add.call_count == len(expected_models) - 1
    session_target.add.assert_any_call(trade)
    session_target.add.assert_any_call(pairlock)
    session_target.add.assert_any_call(kv)
    session_target.add.assert_any_call(custom_data)
    session_target.add.assert_any_call(wallet_history)

    assert session_target.commit.call_count == 5
    assert make_transient_mock.call_count == 6
    make_transient_mock.assert_any_call(trade)
    make_transient_mock.assert_any_call(order)

    set_sequence_ids_mock.assert_called_once_with(
        "bind",
        trade_id=11,
        order_id=12,
        pairlock_id=13,
        kv_id=14,
        custom_data_id=15,
        wallet_history_id=16,
    )
