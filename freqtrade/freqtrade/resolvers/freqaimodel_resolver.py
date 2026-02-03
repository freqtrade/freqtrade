# pragma pylint: disable=attribute-defined-outside-init

"""
This module load a custom model for freqai
"""

import logging
from pathlib import Path

from freqtrade.constants import USERPATH_FREQAIMODELS, Config
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.resolvers import IResolver


logger = logging.getLogger(__name__)


class FreqaiModelResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt loss class
    """

    object_type = IFreqaiModel
    object_type_str = "FreqaiModel"
    user_subdir = USERPATH_FREQAIMODELS
    initial_search_path = (
        Path(__file__).parent.parent.joinpath("freqai/prediction_models").resolve()
    )
    extra_path = "freqaimodel_path"

    @staticmethod
    def load_freqaimodel(config: Config) -> IFreqaiModel:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        """
        disallowed_models = ["BaseRegressionModel"]

        freqaimodel_name = config.get("freqaimodel")
        if not freqaimodel_name:
            raise OperationalException(
                "No freqaimodel set. Please use `--freqaimodel` to "
                "specify the FreqaiModel class to use.\n"
            )
        if freqaimodel_name in disallowed_models:
            raise OperationalException(
                f"{freqaimodel_name} is a baseclass and cannot be used directly. Please choose "
                "an existing child class or inherit from this baseclass.\n"
            )
        freqaimodel = FreqaiModelResolver.load_object(
            freqaimodel_name,
            config,
            kwargs={"config": config},
        )

        return freqaimodel
