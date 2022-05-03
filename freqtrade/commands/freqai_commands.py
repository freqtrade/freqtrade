import logging
from typing import Any, Dict

from freqtrade import constants
from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.misc import round_coin_value


logger = logging.getLogger(__name__)

def start_training(args: Dict[str, Any]) -> None:
    """
    Train a model for predicting signals
    :param args: Cli args from Arguments()
    :return: None
    """
    from freqtrade.freqai.training import Training

    config = setup_utils_configuration(args, RunMode.FREQAI)

    training = Training(config)
    training.start()
