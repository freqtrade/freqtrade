import logging
from pathlib import Path
from typing import Any, Dict

from freqtrade.constants import USERPATH_HYPEROPTS
from freqtrade.exceptions import OperationalException
from freqtrade.state import RunMode
from freqtrade.configuration import setup_utils_configuration
from freqtrade.misc import render_template, render_template_with_fallback

logger = logging.getLogger(__name__)


def deploy_custom_hyperopt(hyperopt_name: str, hyperopt_path: Path, buy_indicators: str, sell_indicators: str) -> None:
    """
    Deploys a custom hyperopt template to hyperopt_path
    TODO make the code below more dynamic with a large list of indicators instead of a few templates
    """
    fallback = 'full'
    buy_guards = render_template_with_fallback(
        templatefile=f"subtemplates/hyperopt_buy_guards_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/hyperopt_buy_guards_{fallback}.j2",
    )
    sell_guards = render_template_with_fallback(
        templatefile=f"subtemplates/hyperopt_sell_guards_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/hyperopt_sell_guards_{fallback}.j2",
    )
    buy_space = render_template_with_fallback(
        templatefile=f"subtemplates/hyperopt_buy_space_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/hyperopt_buy_space_{fallback}.j2",
    )
    sell_space = render_template_with_fallback(
        templatefile=f"subtemplates/hyperopt_sell_space_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/hyperopt_sell_space_{fallback}.j2",
    )

    strategy_text = render_template(templatefile='base_hyperopt.py.j2',
                                    arguments={"hyperopt": hyperopt_name,
                                               "buy_guards": buy_guards,
                                               "sell_guards": sell_guards,
                                               "buy_space": buy_space,
                                               "sell_space": sell_space,
                                               })

    logger.info(f"Writing custom hyperopt to `{hyperopt_path}`.")
    hyperopt_path.write_text(strategy_text)


def start_build_hyperopt(args: Dict[str, Any]) -> None:
    """
    Check if the right subcommands where passed and start building the hyperopt
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    # check what the name of the hyperopt should
    if not 'hyperopt' in args or not args['hyperopt']:
        raise OperationalException("`build-hyperopt` requires --hyperopt to be set.")
    elif not 'buy_indicators' in args or not args['buy_indicators']:
        raise OperationalException("`build-hyperopt` requires --buy-indicators to be set.")
    elif not 'sell_indicators' in args or not args['sell_indicators']:
        raise OperationalException("`build-hyperopt` requires --sell-indicators to be set.")
    else:
        if args['hyperopt'] == 'DefaultHyperopt':
            raise OperationalException("DefaultHyperopt is not allowed as name.")

        new_path = config['user_data_dir'] / USERPATH_HYPEROPTS / (args['hyperopt'] + '.py')
        if new_path.exists():
            raise OperationalException(f"`{new_path}` already exists. "
                                       "Please choose another Hyperopt Name.")
        deploy_custom_hyperopt(args['hyperopt'], new_path,
                               args['buy_indicators'], args['sell_indicators'])
