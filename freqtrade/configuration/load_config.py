"""
This module contain functions to load the configuration file
"""
import logging
import re
import sys
from os import environ
from pathlib import Path
from typing import Any, Dict

import rapidjson

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)

CONFIG_PARSE_MODE = rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS


class SubstitutionException(Exception):
    """
    Indicates that a variable within the configuration couldn't be substituted.
    """

    def __init__(self, key: str, offset: int):
        self.offset = offset
        self.err = f'Environment variable {key} was requested for substitution, but is not set.'
        super().__init__(self.err)


def log_config_error_range(path: str, offset: int) -> str:
    """
    Parses configuration file and prints range around the specified offset
    """
    if path != '-' and offset != -1:
        text = Path(path).read_text()
        # Fetch an offset of 80 characters around the error line
        subtext = text[offset - min(80, offset):offset + 80]
        segments = subtext.split('\n')
        if len(segments) > 3:
            # Remove first and last lines, to avoid odd truncations
            return '\n'.join(segments[1:-1])
        else:
            return subtext

    return ''


def substitute_environment_variable(match: re.Match) -> str:
    """
    Substitutes a matched environment variable with its value
    """
    key = match.group(1).strip()
    if key not in environ:
        raise SubstitutionException(key, match.start(0))

    return environ[key]


def extract_error_offset(errmsg: str) -> int:
    offsetlist = re.findall(r'(?<=Parse\serror\sat\soffset\s)\d+', errmsg)
    if offsetlist:
        return int(offsetlist[0])

    return -1


def load_config_file(path: str) -> Dict[str, Any]:
    """
    Loads a config file from the given path
    :param path: path as str
    :return: configuration as dictionary
    """
    try:
        # Read config from stdin if requested in the options
        with open(path) if path != '-' else sys.stdin as file:
            content = re.sub(r'\${(.*?)}', substitute_environment_variable, file.read())
            config = rapidjson.loads(content, parse_mode=CONFIG_PARSE_MODE)
    except FileNotFoundError:
        raise OperationalException(
            f'Config file "{path}" not found!'
            ' Please create a config file or check whether it exists.')
    except rapidjson.JSONDecodeError as e:
        err_offset = extract_error_offset(str(e))
        err_range = log_config_error_range(path, err_offset)
        raise OperationalException(
            f'{e}\n'
            f'Please verify the following segment of your configuration:\n{err_range}'
            if err_range else 'Please verify your configuration file for syntax errors.'
        )
    except SubstitutionException as e:
        err_range = log_config_error_range(path, e.offset)
        raise OperationalException(
            f'{e}\n'
            f'Please verify the following segment of your configuration:\n{err_range}'
            if err_range else 'Please verify your configuration file for syntax errors.'
        )

    return config
