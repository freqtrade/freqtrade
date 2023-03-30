import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.freqai.normalization import (LegacyNormalization, MinMaxNormalization,
                                            QuantileNormalization, StandardNormalization)
from tests.freqai.conftest import make_data_dictionary


def test_default_normalization_is_legacy(mocker, freqai_conf):
    freqai_1st = make_data_dictionary(mocker, freqai_conf, normalized=False)
    data_dict_1st = freqai_1st.dk.data_dictionary
    freqai_1st.dk.normalize_data(data_dict_1st)

    freqai_conf["freqai"]["feature_parameters"]["data_normalization"] = "legacy"
    freqai_2nd = make_data_dictionary(mocker, freqai_conf, normalized=False)
    data_dict_2nd = freqai_2nd.dk.data_dictionary

    assert not freqai_1st.dk.data_dictionary['train_features'].equals(
        freqai_2nd.dk.data_dictionary['train_features']), "raw data is equal to normalized data"

    freqai_2nd.dk.normalize_data(data_dict_2nd)

    assert freqai_1st.dk.data_dictionary['train_features'].equals(
        freqai_2nd.dk.data_dictionary['train_features']), \
        "explicit\\implicit legacy normalization mismatch"


def test_legacy_normalization_add_max_min_columns(mocker, freqai_conf):
    freqai_conf["freqai"]["feature_parameters"]["data_normalization"] = "legacy"
    freqai = make_data_dictionary(mocker, freqai_conf, normalized=False)
    data_dict = freqai.dk.data_dictionary
    freqai.dk.normalize_data(data_dict)

    assert any('_max' in entry for entry in freqai.dk.data.keys())
    assert any('_min' in entry for entry in freqai.dk.data.keys())
    assert all(not entry.endswith('_scaler') for entry in freqai.dk.pkl_data.keys())


def test_standard_normalization_dont_add_max_min_columns(mocker, freqai_conf):
    freqai_conf["freqai"]["feature_parameters"]["data_normalization"] = "standard"
    freqai = make_data_dictionary(mocker, freqai_conf, normalized=False)
    data_dict = freqai.dk.data_dictionary
    freqai.dk.normalize_data(data_dict)
    assert all(not entry.endswith('_max') for entry in freqai.dk.data.keys())
    assert all(not entry.endswith('_min') for entry in freqai.dk.data.keys())
    assert any(entry.endswith('_scaler') for entry in freqai.dk.pkl_data.keys())


def test_legacy_and_standard_normalization_difference(mocker, freqai_conf):
    freqai_conf["freqai"]["feature_parameters"]["data_normalization"] = "legacy"
    freqai_1st = make_data_dictionary(mocker, freqai_conf, normalized=False)
    data_dict_1st = freqai_1st.dk.data_dictionary
    freqai_1st.dk.normalize_data(data_dict_1st)

    freqai_conf["freqai"]["feature_parameters"]["data_normalization"] = "standard"
    freqai_2nd = make_data_dictionary(mocker, freqai_conf, normalized=False)
    data_dict_2nd = freqai_2nd.dk.data_dictionary
    freqai_2nd.dk.normalize_data(data_dict_2nd)

    assert not freqai_1st.dk.data_dictionary['train_features'].equals(
        freqai_2nd.dk.data_dictionary['train_features']), \
        "legacy and standard normalization produce same features"


@pytest.mark.parametrize(
    "config_id, norm_class",
    [
        ("legacy", LegacyNormalization),
        ("standard", StandardNormalization),
        ("minmax", MinMaxNormalization),
        ("quantile", QuantileNormalization),
    ],
)
def test_normalization_class(config_id, norm_class, mocker, freqai_conf):
    freqai_conf["freqai"]["feature_parameters"]["data_normalization"] = config_id
    freqai = make_data_dictionary(mocker, freqai_conf)
    assert type(freqai.dk.normalizer) == norm_class


def test_assertion_invalid_normalization_id(mocker, freqai_conf):
    freqai_conf["freqai"]["feature_parameters"]["data_normalization"] = "not_a_norm_id"
    try:
        make_data_dictionary(mocker, freqai_conf)
        assert False, "missing expected normalization factory exception"
    except OperationalException as e_info:
        assert str(e_info).startswith("Invalid data normalization identifier"), \
            "unexpected exception string"


@pytest.mark.parametrize(
    "config_id",
    [
        "legacy",
        "standard",
        "minmax",
        "quantile",
    ],
)
def test_denormalization(config_id, mocker, freqai_conf):
    freqai_conf["freqai"]["feature_parameters"]["data_normalization"] = config_id
    freqai_1st = make_data_dictionary(mocker, freqai_conf)
    data_dict_1st = freqai_1st.dk.data_dictionary

    freqai_2nd = make_data_dictionary(mocker, freqai_conf, normalized=False)
    data_dict_2nd = freqai_2nd.dk.data_dictionary

    denorm_labels = freqai_1st.dk.denormalize_labels_from_metadata(
        data_dict_1st["train_labels"]).round(9)
    assert denorm_labels.equals(data_dict_2nd['train_labels'].round(9)), \
        "raw labels data isn't the same as denormalized labels"
