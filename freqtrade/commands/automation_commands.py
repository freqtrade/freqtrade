import ast
import logging
from pathlib import Path
from typing import Any, Dict

from freqtrade.constants import USERPATH_HYPEROPTS
from freqtrade.exceptions import OperationalException
from freqtrade.state import RunMode
from freqtrade.configuration import setup_utils_configuration
from freqtrade.misc import render_template

logger = logging.getLogger(__name__)

'''
    TODO 
    -make the code below more dynamic with a large list of indicators and aims
    -buy_space integer values variation based on aim(later deep learning)
    -add --mode , see notes
    -when making the strategy reading tool, make sure that the populate indicators gets copied to here
'''

POSSIBLE_GUARDS = ["rsi", "mfi", "fastd"]
POSSIBLE_TRIGGERS = ["bb_lowerband", "bb_upperband"]
POSSIBLE_VALUES = {"above": ">", "below": "<"}


def build_hyperopt_buyelements(buy_indicators: Dict[str, str]):
    """
    Build the arguments with the placefillers for the buygenerator
    """
    buy_guards = ""
    buy_triggers = ""
    buy_space = ""

    for indicator in buy_indicators:
        # Error handling
        if not indicator in POSSIBLE_GUARDS and not indicator in POSSIBLE_TRIGGERS:
            raise OperationalException(
                f"`{indicator}` is not part of the available indicators. The current options are {POSSIBLE_GUARDS + POSSIBLE_TRIGGERS}.")
        elif not buy_indicators[indicator] in POSSIBLE_VALUES:
            raise OperationalException(
                f"`{buy_indicators[indicator]}` is not part of the available indicator options. The current options are {POSSIBLE_VALUES}.")
        # If the indicator is a guard
        elif indicator in POSSIBLE_GUARDS:
            # get the symbol corrosponding to the value
            aim = POSSIBLE_VALUES[buy_indicators[indicator]]

            # add the guard to its argument
            buy_guards += f"if '{indicator}-enabled' in params and params['{indicator}-enabled']: conditions.append(dataframe['{indicator}'] {aim} params['{indicator}-value'])"

            # add the space to its argument
            buy_space += f"Integer(10, 90, name='{indicator}-value'), Categorical([True, False], name='{indicator}-enabled'),"
        # If the indicator is a trigger
        elif indicator in POSSIBLE_TRIGGERS:
            # get the symbol corrosponding to the value
            aim = POSSIBLE_VALUES[buy_indicators[indicator]]

            # add the trigger to its argument
            buy_triggers += f"if params['trigger'] == '{indicator}': conditions.append(dataframe['{indicator}'] {aim} dataframe['close'])"

    # Final line of indicator space makes all triggers
    
    buy_space += "Categorical(["

    # adding all triggers to the list
    for indicator in buy_indicators:
        if indicator in POSSIBLE_TRIGGERS:
            buy_space += f"'{indicator}', "

    # Deleting the last ", "
    buy_space = buy_space[:-2]
    buy_space += "], name='trigger')"

    return {"buy_guards": buy_guards, "buy_triggers": buy_triggers, "buy_space": buy_space}


def build_hyperopt_sellelements(sell_indicators: Dict[str, str]):
    """
    Build the arguments with the placefillers for the sellgenerator
    """
    sell_guards = ""
    sell_triggers = ""
    sell_space = ""

    for indicator in sell_indicators:
        # Error handling
        if not indicator in POSSIBLE_GUARDS and not indicator in POSSIBLE_TRIGGERS:
            raise OperationalException(
                f"`{indicator}` is not part of the available indicators. The current options are {POSSIBLE_GUARDS + POSSIBLE_TRIGGERS}.")
        elif not sell_indicators[indicator] in POSSIBLE_VALUES:
            raise OperationalException(
                f"`{sell_indicators[indicator]}` is not part of the available indicator options. The current options are {POSSIBLE_VALUES}.")
        # If indicator is a guard
        elif indicator in POSSIBLE_GUARDS:
            # get the symbol corrosponding to the value
            aim = POSSIBLE_VALUES[sell_indicators[indicator]]

            # add the guard to its argument
            sell_guards += f"if '{indicator}-enabled' in params and params['sell-{indicator}-enabled']: conditions.append(dataframe['{indicator}'] {aim} params['sell-{indicator}-value'])"

            # add the space to its argument
            sell_space += f"Integer(10, 90, name='sell-{indicator}-value'), Categorical([True, False], name='sell-{indicator}-enabled'),"
        # If the indicator is a trigger
        elif indicator in POSSIBLE_TRIGGERS:
            # get the symbol corrosponding to the value
            aim = POSSIBLE_VALUES[sell_indicators[indicator]]

            # add the trigger to its argument
            sell_triggers += f"if params['sell-trigger'] == 'sell-{indicator}': conditions.append(dataframe['{indicator}'] {aim} dataframe['close'])"

    # Final line of indicator space makes all triggers

    sell_space += "Categorical(["

    # Adding all triggers to the list
    for indicator in sell_indicators:
        if indicator in POSSIBLE_TRIGGERS:
            sell_space += f"'sell-{indicator}', "

    # Deleting the last ", "
    sell_space = sell_space[:-2]
    sell_space += "], name='trigger')"

    return {"sell_guards": sell_guards, "sell_triggers": sell_triggers, "sell_space": sell_space}


def deploy_custom_hyperopt(hyperopt_name: str, hyperopt_path: Path, buy_indicators: Dict[str, str], sell_indicators: Dict[str, str]) -> None:
    """
    Deploys a custom hyperopt template to hyperopt_path
    """

    # Build the arguments for the buy and sell generators
    buy_args = build_hyperopt_buyelements(buy_indicators)
    sell_args = build_hyperopt_sellelements(sell_indicators)

    # Build the final template
    strategy_text = render_template(templatefile='base_hyperopt.py.j2',
                                    arguments={"hyperopt": hyperopt_name,
                                               "buy_guards": buy_args["buy_guards"],
                                               "buy_triggers": buy_args["buy_triggers"],
                                               "buy_space": buy_args["buy_space"],
                                               "sell_guards": sell_args["sell_guards"],
                                               "sell_triggers": sell_args["sell_triggers"],
                                               "sell_space": sell_args["sell_space"],
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

        buy_indicators = ast.literal_eval(args['buy_indicators'])
        sell_indicators = ast.literal_eval(args['sell_indicators'])

        deploy_custom_hyperopt(args['hyperopt'], new_path,
                               buy_indicators, sell_indicators)
