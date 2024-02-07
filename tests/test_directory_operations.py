# pragma pylint: disable=missing-docstring, protected-access, invalid-name
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from freqtrade.configuration.directory_operations import (chown_user_directory, copy_sample_files,
                                                          create_datadir, create_userdata_dir)
from freqtrade.exceptions import OperationalException
from tests.conftest import log_has, log_has_re


def test_create_datadir(mocker, default_conf, caplog) -> None:
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    md = mocker.patch.object(Path, 'mkdir', MagicMock())

    create_datadir(default_conf, '/foo/bar')
    assert md.call_args[1]['parents'] is True
    assert log_has('Created data directory: /foo/bar', caplog)


def test_create_userdata_dir(mocker, default_conf, caplog) -> None:
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    md = mocker.patch.object(Path, 'mkdir', MagicMock())

    x = create_userdata_dir('/tmp/bar', create_dir=True)
    assert md.call_count == 10
    assert md.call_args[1]['parents'] is False
    assert log_has(f'Created user-data directory: {Path("/tmp/bar")}', caplog)
    assert isinstance(x, Path)
    assert str(x) == str(Path("/tmp/bar"))


def test_create_userdata_dir_and_chown(mocker, tmp_path, caplog) -> None:
    sp_mock = mocker.patch('subprocess.check_output')
    path = tmp_path / 'bar'
    assert not path.is_dir()

    x = create_userdata_dir(str(path), create_dir=True)
    assert sp_mock.call_count == 0
    assert log_has(f'Created user-data directory: {path}', caplog)
    assert isinstance(x, Path)
    assert path.is_dir()
    assert (path / 'data').is_dir()

    os.environ['FT_APP_ENV'] = 'docker'
    chown_user_directory(path / 'data')
    assert sp_mock.call_count == 1
    del os.environ['FT_APP_ENV']


def test_create_userdata_dir_exists(mocker, default_conf, caplog) -> None:
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=True))
    md = mocker.patch.object(Path, 'mkdir', MagicMock())

    create_userdata_dir('/tmp/bar')
    assert md.call_count == 0


def test_create_userdata_dir_exists_exception(mocker, default_conf, caplog) -> None:
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    md = mocker.patch.object(Path, 'mkdir', MagicMock())

    with pytest.raises(OperationalException,
                       match=r'Directory `.{1,2}tmp.{1,2}bar` does not exist.*'):
        create_userdata_dir('/tmp/bar',  create_dir=False)
    assert md.call_count == 0


def test_copy_sample_files(mocker, default_conf, caplog) -> None:
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=True))
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    copymock = mocker.patch('shutil.copy', MagicMock())

    copy_sample_files(Path('/tmp/bar'))
    assert copymock.call_count == 3
    assert copymock.call_args_list[0][0][1] == str(
        Path('/tmp/bar') / 'strategies/sample_strategy.py')
    assert copymock.call_args_list[1][0][1] == str(
        Path('/tmp/bar') / 'hyperopts/sample_hyperopt_loss.py')
    assert copymock.call_args_list[2][0][1] == str(
        Path('/tmp/bar') / 'notebooks/strategy_analysis_example.ipynb')


def test_copy_sample_files_errors(mocker, default_conf, caplog) -> None:
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    mocker.patch('shutil.copy', MagicMock())
    with pytest.raises(OperationalException,
                       match=r"Directory `.{1,2}tmp.{1,2}bar` does not exist\."):
        copy_sample_files(Path('/tmp/bar'))

    mocker.patch.object(Path, "is_dir", MagicMock(side_effect=[True, False]))

    with pytest.raises(OperationalException,
                       match=r"Directory `.{1,2}tmp.{1,2}bar.{1,2}strategies` does not exist\."):
        copy_sample_files(Path('/tmp/bar'))
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=True))
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    copy_sample_files(Path('/tmp/bar'))
    assert log_has_re(r"File `.*` exists already, not deploying sample file\.", caplog)
    caplog.clear()
    copy_sample_files(Path('/tmp/bar'), overwrite=True)
    assert log_has_re(r"File `.*` exists already, overwriting\.", caplog)
