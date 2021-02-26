import ast
import logging
from pathlib import Path
from typing import Any, Dict, List

from freqtrade.constants import (USERPATH_HYPEROPTS,
                                 USERPATH_STRATEGIES)
from freqtrade.exceptions import OperationalException
from freqtrade.state import RunMode
from freqtrade.configuration import setup_utils_configuration
from freqtrade.misc import render_template

logger = logging.getLogger(__name__)


# ---------------------------------------------------extract-strategy------------------------------------------------------

def get_indicator_info(file: List, indicators: Dict) -> None:
    """
    Get all necessary information to build a custom hyperopt space using
    the file and a dictionary filled with the indicators and their corropsonding line numbers.
    """
    info_list = []
    for indicator in indicators:
        indicator_info = []

        # find the corrosponding aim
        for position, line in enumerate(file):
            if position == indicators[indicator]:
                # use split twice to remove the context around the indicator
                back_of_line = line.split(f"(dataframe['{indicator}'] ", 1)[1]
                aim = back_of_line.split()[0]

                # add the indicator and aim to the info
                indicator_info.append(indicator)
                indicator_info.append(aim)

                # check if first character after aim is a d in which case the indicator is a trigger
                if back_of_line.split()[1][0] == "d":
                    indicator_info.append("trigger")

                    # add the second indicator of the guard to the info list
                    back_of_line = back_of_line.split("dataframe['")[1]
                    second_indicator = back_of_line.split("'])")[0]
                    indicator_info.append(second_indicator)

                # elif indicator[0:3] == "CDL":
                    # indicator_info.append("guard")

                # else it is a regular guard
                else:
                    indicator_info.append("guard")

                    value = back_of_line.split()[1]
                    value = value[:-1]
                    value = float(value)

                    indicator_info.append(value)
        info_list.append(indicator_info)

    return info_list


def extract_lists(strategypath: Path) -> None:
    """
    Get the indicators, their aims and the stoploss and format them into lists
    """

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

    # build the final lists
    buy_info_list = get_indicator_info(stored_file, buyindicators)
    sell_info_list = get_indicator_info(stored_file, sellindicators)
    
    # put the final lists into a tuple
    final_lists = (buy_info_list, sell_info_list)

    return final_lists


def start_extract_strategy(args: Dict) -> None:
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
        extracted_lists = str(extract_lists(strategy_path))

        # save the dicts in a file
        logger.info(f"Writing custom hyperopt to `{new_path}`.")
        new_path.write_text(extracted_lists)


# --------------------------------------------------custom-hyperopt------------------------------------------------------

def custom_hyperopt_buyelements(buy_indicators: List):
    """
    Build the arguments with the placefillers for the buygenerator
    """
    buy_guards = ""
    buy_triggers = ""
    buy_space = ""

    for indicator_info in buy_indicators:
        indicator = indicator_info[0]
        aim = indicator_info[1]
        usage = indicator_info[2]

        # If the indicator is a guard
        if usage == "guard":
            value = indicator_info[3]
            
            if value >= -1.0 and value <= 1.0:
                lower_bound = value - 0.3
                upper_bound = value + 0.3
            else:
                lower_bound = value - 30.0
                upper_bound = value + 30.0

            # add the guard to its argument
            buy_guards += f"if params.get('{indicator}-enabled'):\n    conditions.append(dataframe['{indicator}'] {aim} params['{indicator}-value'])\n"

            # add the space to its argument
            buy_space += f"Integer({lower_bound}, {upper_bound}, name='{indicator}-value'),\nCategorical([True, False], name='{indicator}-enabled'),\n"

        # If the indicator is a trigger
        elif usage == "trigger":
            secondindicator = indicator_info[3]
            # add the trigger to its argument
            buy_triggers += f"if params['trigger'] == '{indicator}':\n    conditions.append(dataframe['{indicator}'] {aim} dataframe['{secondindicator}'])\n"

    # Final line of indicator space makes all triggers

    buy_space += "Categorical(["

    # adding all triggers to the list
    for indicator_info in buy_indicators:
        indicator = indicator_info[0]
        usage = indicator_info[2]

        if usage == "trigger":
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

    for indicator_info in sell_indicators:
        indicator = indicator_info[0]
        aim = indicator_info[1]
        usage = indicator_info[2]

        # If the indicator is a guard
        if usage == "guard":
            value = indicator_info[3]
            
            if value >= -1 and value <= 1:
                lower_bound = value - 0.3
                upper_bound = value + 0.3
            else:
                lower_bound = value - 30
                upper_bound = value + 30

            # add the guard to its argument
            sell_guards += f"if params.get('sell-{indicator}-enabled'):\n    conditions.append(dataframe['{indicator}'] {aim} params['sell-{indicator}-value'])\n"

            # add the space to its argument
            sell_space += f"Integer({lower_bound}, {upper_bound}, name='sell-{indicator}-value'),\nCategorical([True, False], name='sell-{indicator}-enabled'),\n"

        # If the indicator is a trigger
        elif usage == "trigger":
            secondindicator = indicator_info[3]

            # add the trigger to its argument
            sell_triggers += f"if params['sell-trigger'] == 'sell-{indicator}':\n    conditions.append(dataframe['{indicator}'] {aim} dataframe['{secondindicator}'])\n"

    # Final line of indicator space makes all triggers

    sell_space += "Categorical(["

    # adding all triggers to the list
    for indicator_info in sell_indicators:
        indicator = indicator_info[0]
        usage = indicator_info[2]

        if usage == "trigger":
            sell_space += f"'sell-{indicator}', "

    # Deleting the last ", "
    sell_space = sell_space[:-2]
    sell_space += "], name='sell-trigger')"

    return {"sell_guards": sell_guards, "sell_triggers": sell_triggers, "sell_space": sell_space}


def deploy_custom_hyperopt(hyperopt_name: str, hyperopt_path: Path, buy_indicators: Dict[str, str], sell_indicators: Dict[str, str]) -> None:
    """
    Deploys a custom hyperopt template to hyperopt_path
    """

    # Build the arguments for the buy and sell generators
    buy_args = custom_hyperopt_buyelements(buy_indicators)
    sell_args = custom_hyperopt_sellelements(sell_indicators)

    # Build the final template
    strategy_text = render_template(templatefile='base_custom_hyperopt.py.j2',
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

    # check what the name of the hyperopt should be

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

def start_build_hyperopt(args: Dict[str, Any]) -> None:
    """
    Check if the right subcommands where passed and start building the hyperopt
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    # strategy and hyperopt need to be defined
    if not 'strategy' in args or not args['strategy']:
        raise OperationalException("`build-hyperopt` requires --strategy to be set.")
    if not 'hyperopt' in args or not args['hyperopt']:
        args['hyperopt'] = args['strategy'] + "opt"
    else:
        if args['hyperopt'] == 'DefaultHyperopt':
            raise OperationalException("DefaultHyperopt is not allowed as name.")

        # the path of the chosen strategy
        strategy_path = config['user_data_dir'] / USERPATH_STRATEGIES / (args['strategy'] + '.py')

        # the path where the hyperopt should be written
        new_path = config['user_data_dir'] / USERPATH_HYPEROPTS / (args['hyperopt'] + '.py')
        if new_path.exists():
            raise OperationalException(f"`{new_path}` already exists. "
                                       "Please choose another Hyperopt Name.")

        # extract the buy and sell indicators as dicts
        extracted_lists = extract_lists(strategy_path)

        buy_indicators = extracted_lists[0]
        sell_indicators = extracted_lists[1]

        # use the dicts to write the hyperopt
        deploy_custom_hyperopt(args['hyperopt'], new_path,
                               buy_indicators, sell_indicators)
