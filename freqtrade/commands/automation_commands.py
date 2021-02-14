import ast
import logging
from pathlib import Path
from typing import Any, Dict

from freqtrade.constants import (USERPATH_HYPEROPTS,
                                 USERPATH_STRATEGIES,
                                 POSSIBLE_GUARDS,
                                 POSSIBLE_TRIGGERS,
                                 POSSIBLE_AIMS)
from freqtrade.exceptions import OperationalException
from freqtrade.state import RunMode
from freqtrade.configuration import setup_utils_configuration
from freqtrade.misc import render_template

logger = logging.getLogger(__name__)


# ---------------------------------------------------extract-strategy------------------------------------------------------
def extract_dicts(strategypath: Path):
    # store the file in a list for reference
    stored_file = []
    with open(strategypath) as file:
        for line in file:
            stored_file.append(line)

    # find the start and end of buy trend
    for position, line in enumerate(stored_file):
        if "populate_buy_trend(" in line:
            start_buy_number = position
        elif "populate_sell_trend(" in line:
            end_buy_number = position

    # list the numbers between the start and end of buy trend
    buy_lines = []
    for i in range(start_buy_number, end_buy_number):
        buy_lines.append(i)

    # populate the indicators dictionaries with indicators attached to the line they are on
    buyindicators = {}
    sellindicators = {}
    for position, line in enumerate(stored_file):
        # check the lines in buy trend for indicator and add them 
        if position in buy_lines and "(dataframe['" in line:
            # use split twice to remove the context around the indicator
            back_of_line = line.split("(dataframe['", 1)[1]
            buyindicator = back_of_line.split("'] ", 1)[0]

            buyindicators[buyindicator] = position

        # check the lines in sell trend for indicator and add them 
        elif position > end_buy_number and "(dataframe['" in line:
            # use split twice to remove the context around the indicator
            back_of_line = line.split("(dataframe['", 1)[1]
            sellindicator = back_of_line.split("'] ", 1)[0]

            sellindicators[sellindicator] = position
    
    # build the final buy dictionary
    buy_dict = {}
    for indicator in buyindicators:
        # find the corrosponding aim
        for position, line in enumerate(stored_file):
            if position == buyindicators[indicator]:
                # use split twice to remove the context around the indicator
                back_of_line = line.split(f"(dataframe['{indicator}'] ", 1)[1]
                aim = back_of_line.split()[0]
        buy_dict[indicator] = aim

    # build the final sell dictionary
    sell_dict = {}
    for indicator in sellindicators:
        # find the corrosponding aim
        for position, line in enumerate(stored_file):
            if position == sellindicators[indicator]:
                # use split twice to remove the context around the indicator
                back_of_line = line.split(f"(dataframe['{indicator}'] ", 1)[1]
                aim = back_of_line.split()[0]
        sell_dict[indicator] = aim

    # put the final dicts into a tuple
    final_dicts = (buy_dict, sell_dict)

    return final_dicts


def start_extract_strategy(args: Dict[str, Any]) -> None:
    """
    Check if the right subcommands where passed and start extracting the strategy data
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    # check if all required options are filled in
    if not 'strategy' in args or not args['strategy']:
        raise OperationalException("`extract-strategy` requires --strategy to be set.")
    else:
        # if the name is not specified use (strategy)_extract
        if not 'extract_name' in args or not args['extract_name']:
            args['extract_name'] = args['strategy'] + "_extract"

        new_path = config['user_data_dir'] / USERPATH_STRATEGIES / (args['extract_name'] + '.txt')
        if new_path.exists():
            raise OperationalException(f"`{new_path}` already exists. "
                                       "Please choose another name.")
        # the path of the chosen strategy
        strategy_path = config['user_data_dir'] / USERPATH_STRATEGIES / (args['strategy'] + '.py')

        # extract the buy and sell indicators as dicts
        extracted_dicts = str(extract_dicts(strategy_path))

        # save the dicts in a file
        logger.info(f"Writing custom hyperopt to `{new_path}`.")
        new_path.write_text(extracted_dicts)


# --------------------------------------------------custom-hyperopt------------------------------------------------------
'''
    TODO 
    -make the code below more dynamic with a large list of indicators and aims
    -buy_space integer values variation based on aim(later deep learning)
    -add --mode , see notes
    -when making the strategy reading tool, make sure that the populate indicators gets copied to here
    -Custom stoploss and roi
    -cli option to read extracted strategies files (--extraction)
'''


def custom_hyperopt_buyelements(buy_indicators: Dict[str, str]):
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
        elif not buy_indicators[indicator] in POSSIBLE_AIMS:
            raise OperationalException(
                f"`{buy_indicators[indicator]}` is not part of the available indicator options. The current options are {POSSIBLE_AIMS}.")
        # If the indicator is a guard
        elif indicator in POSSIBLE_GUARDS:
            # get the symbol corrosponding to the value
            aim = POSSIBLE_AIMS[buy_indicators[indicator]]

            # add the guard to its argument
            buy_guards += f"if '{indicator}-enabled' in params and params['{indicator}-enabled']: conditions.append(dataframe['{indicator}'] {aim} params['{indicator}-value'])"

            # add the space to its argument
            buy_space += f"Integer(10, 90, name='{indicator}-value'), Categorical([True, False], name='{indicator}-enabled'),"
        # If the indicator is a trigger
        elif indicator in POSSIBLE_TRIGGERS:
            # get the symbol corrosponding to the value
            aim = POSSIBLE_AIMS[buy_indicators[indicator]]

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


def custom_hyperopt_sellelements(sell_indicators: Dict[str, str]):
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
        elif not sell_indicators[indicator] in POSSIBLE_AIMS:
            raise OperationalException(
                f"`{sell_indicators[indicator]}` is not part of the available indicator options. The current options are {POSSIBLE_AIMS}.")
        # If indicator is a guard
        elif indicator in POSSIBLE_GUARDS:
            # get the symbol corrosponding to the value
            aim = POSSIBLE_AIMS[sell_indicators[indicator]]

            # add the guard to its argument
            sell_guards += f"if '{indicator}-enabled' in params and params['sell-{indicator}-enabled']: conditions.append(dataframe['{indicator}'] {aim} params['sell-{indicator}-value'])"

            # add the space to its argument
            sell_space += f"Integer(10, 90, name='sell-{indicator}-value'), Categorical([True, False], name='sell-{indicator}-enabled'),"
        # If the indicator is a trigger
        elif indicator in POSSIBLE_TRIGGERS:
            # get the symbol corrosponding to the value
            aim = POSSIBLE_AIMS[sell_indicators[indicator]]

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
    buy_args = custom_hyperopt_buyelements(buy_indicators)
    sell_args = custom_hyperopt_sellelements(sell_indicators)

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


def start_custom_hyperopt(args: Dict[str, Any]) -> None:
    """
    Check if the right subcommands where passed and start building the hyperopt
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if not 'hyperopt' in args or not args['hyperopt']:
        raise OperationalException("`custom-hyperopt` requires --hyperopt to be set.")
    elif not 'buy_indicators' in args or not args['buy_indicators']:
        raise OperationalException("`custom-hyperopt` requires --buy-indicators to be set.")
    elif not 'sell_indicators' in args or not args['sell_indicators']:
        raise OperationalException("`custom-hyperopt` requires --sell-indicators to be set.")
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

# --------------------------------------------------build-hyperopt------------------------------------------------------