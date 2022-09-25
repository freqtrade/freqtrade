# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom hyperopt
"""
import logging
from pathlib import Path

from freqtrade.constants import HYPEROPT_LOSS_BUILTIN, USERPATH_HYPEROPTS, Config
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.resolvers import IResolver


logger = logging.getLogger(__name__)


class HyperOptLossResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt loss class
    """
    object_type = IHyperOptLoss
    object_type_str = "HyperoptLoss"
    user_subdir = USERPATH_HYPEROPTS
    initial_search_path = Path(__file__).parent.parent.joinpath('optimize/hyperopt_loss').resolve()

    @staticmethod
    def load_hyperoptloss(config: Config) -> IHyperOptLoss:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        """

        hyperoptloss_name = config.get('hyperopt_loss')
        if not hyperoptloss_name:
            raise OperationalException(
                "No Hyperopt loss set. Please use `--hyperopt-loss` to "
                "specify the Hyperopt-Loss class to use.\n"
                f"Built-in Hyperopt-loss-functions are: {', '.join(HYPEROPT_LOSS_BUILTIN)}"
            )
        hyperoptloss = HyperOptLossResolver.load_object(hyperoptloss_name,
                                                        config, kwargs={},
                                                        extra_dir=config.get('hyperopt_path'))

        # Assign timeframe to be used in hyperopt
        hyperoptloss.__class__.timeframe = str(config['timeframe'])

        return hyperoptloss
