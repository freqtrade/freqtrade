
import io
import logging
from collections import OrderedDict
from pathlib import Path
from pprint import pformat
from typing import Dict, List

import rapidjson
import tabulate
from colorama import Fore, Style
from joblib import load
from pandas import isna, json_normalize

from freqtrade.exceptions import OperationalException
from freqtrade.misc import round_dict


logger = logging.getLogger(__name__)


class HyperoptTools():

    @staticmethod
    def _read_results(results_file: Path) -> List:
        """
        Read hyperopt results from file
        """
        logger.info("Reading epochs from '%s'", results_file)
        data = load(results_file)
        return data

    @staticmethod
    def load_previous_results(results_file: Path) -> List:
        """
        Load data for epochs from the file if we have one
        """
        epochs: List = []
        if results_file.is_file() and results_file.stat().st_size > 0:
            epochs = HyperoptTools._read_results(results_file)
            # Detection of some old format, without 'is_best' field saved
            if epochs[0].get('is_best') is None:
                raise OperationalException(
                    "The file with HyperoptTools results is incompatible with this version "
                    "of Freqtrade and cannot be loaded.")
            logger.info(f"Loaded {len(epochs)} previous evaluations from disk.")
        return epochs

    @staticmethod
    def print_epoch_details(results, total_epochs: int, print_json: bool,
                            no_header: bool = False, header_str: str = None) -> None:
        """
        Display details of the hyperopt result
        """
        params = results.get('params_details', {})

        # Default header string
        if header_str is None:
            header_str = "Best result"

        if not no_header:
            explanation_str = HyperoptTools._format_explanation_string(results, total_epochs)
            print(f"\n{header_str}:\n\n{explanation_str}\n")

        if print_json:
            result_dict: Dict = {}
            for s in ['buy', 'sell', 'roi', 'stoploss', 'trailing']:
                HyperoptTools._params_update_for_json(result_dict, params, s)
            print(rapidjson.dumps(result_dict, default=str, number_mode=rapidjson.NM_NATIVE))

        else:
            HyperoptTools._params_pretty_print(params, 'buy', "Buy hyperspace params:")
            HyperoptTools._params_pretty_print(params, 'sell', "Sell hyperspace params:")
            HyperoptTools._params_pretty_print(params, 'roi', "ROI table:")
            HyperoptTools._params_pretty_print(params, 'stoploss', "Stoploss:")
            HyperoptTools._params_pretty_print(params, 'trailing', "Trailing stop:")

    @staticmethod
    def _params_update_for_json(result_dict, params, space: str) -> None:
        if space in params:
            space_params = HyperoptTools._space_params(params, space)
            if space in ['buy', 'sell']:
                result_dict.setdefault('params', {}).update(space_params)
            elif space == 'roi':
                # TODO: get rid of OrderedDict when support for python 3.6 will be
                # dropped (dicts keep the order as the language feature)

                # Convert keys in min_roi dict to strings because
                # rapidjson cannot dump dicts with integer keys...
                # OrderedDict is used to keep the numeric order of the items
                # in the dict.
                result_dict['minimal_roi'] = OrderedDict(
                    (str(k), v) for k, v in space_params.items()
                )
            else:  # 'stoploss', 'trailing'
                result_dict.update(space_params)

    @staticmethod
    def _params_pretty_print(params, space: str, header: str) -> None:
        if space in params:
            space_params = HyperoptTools._space_params(params, space, 5)
            params_result = f"\n# {header}\n"
            if space == 'stoploss':
                params_result += f"stoploss = {space_params.get('stoploss')}"
            elif space == 'roi':
                # TODO: get rid of OrderedDict when support for python 3.6 will be
                # dropped (dicts keep the order as the language feature)
                minimal_roi_result = rapidjson.dumps(
                    OrderedDict(
                        (str(k), v) for k, v in space_params.items()
                    ),
                    default=str, indent=4, number_mode=rapidjson.NM_NATIVE)
                params_result += f"minimal_roi = {minimal_roi_result}"
            elif space == 'trailing':

                for k, v in space_params.items():
                    params_result += f'{k} = {v}\n'

            else:
                params_result += f"{space}_params = {pformat(space_params, indent=4)}"
                params_result = params_result.replace("}", "\n}").replace("{", "{\n ")

            params_result = params_result.replace("\n", "\n    ")
            print(params_result)

    @staticmethod
    def _space_params(params, space: str, r: int = None) -> Dict:
        d = params[space]
        # Round floats to `r` digits after the decimal point if requested
        return round_dict(d, r) if r else d

    @staticmethod
    def is_best_loss(results, current_best_loss: float) -> bool:
        return results['loss'] < current_best_loss

    @staticmethod
    def _format_explanation_string(results, total_epochs) -> str:
        return (("*" if results['is_initial_point'] else " ") +
                f"{results['current_epoch']:5d}/{total_epochs}: " +
                f"{results['results_explanation']} " +
                f"Objective: {results['loss']:.5f}")

    @staticmethod
    def get_result_table(config: dict, results: list, total_epochs: int, highlight_best: bool,
                         print_colorized: bool, remove_header: int) -> str:
        """
        Log result table
        """
        if not results:
            return ''

        tabulate.PRESERVE_WHITESPACE = True

        trials = json_normalize(results, max_level=1)
        trials['Best'] = ''
        if 'results_metrics.winsdrawslosses' not in trials.columns:
            # Ensure compatibility with older versions of hyperopt results
            trials['results_metrics.winsdrawslosses'] = 'N/A'

        trials = trials[['Best', 'current_epoch', 'results_metrics.trade_count',
                         'results_metrics.winsdrawslosses',
                         'results_metrics.avg_profit', 'results_metrics.total_profit',
                         'results_metrics.profit', 'results_metrics.duration',
                         'loss', 'is_initial_point', 'is_best']]
        trials.columns = ['Best', 'Epoch', 'Trades', ' Win Draw Loss', 'Avg profit',
                          'Total profit', 'Profit', 'Avg duration', 'Objective',
                          'is_initial_point', 'is_best']
        trials['is_profit'] = False
        trials.loc[trials['is_initial_point'], 'Best'] = '*     '
        trials.loc[trials['is_best'], 'Best'] = 'Best'
        trials.loc[trials['is_initial_point'] & trials['is_best'], 'Best'] = '* Best'
        trials.loc[trials['Total profit'] > 0, 'is_profit'] = True
        trials['Trades'] = trials['Trades'].astype(str)

        trials['Epoch'] = trials['Epoch'].apply(
            lambda x: '{}/{}'.format(str(x).rjust(len(str(total_epochs)), ' '), total_epochs)
        )
        trials['Avg profit'] = trials['Avg profit'].apply(
            lambda x: '{:,.2f}%'.format(x).rjust(7, ' ') if not isna(x) else "--".rjust(7, ' ')
        )
        trials['Avg duration'] = trials['Avg duration'].apply(
            lambda x: '{:,.1f} m'.format(x).rjust(7, ' ') if not isna(x) else "--".rjust(7, ' ')
        )
        trials['Objective'] = trials['Objective'].apply(
            lambda x: '{:,.5f}'.format(x).rjust(8, ' ') if x != 100000 else "N/A".rjust(8, ' ')
        )

        trials['Profit'] = trials.apply(
            lambda x: '{:,.8f} {} {}'.format(
                x['Total profit'], config['stake_currency'],
                '({:,.2f}%)'.format(x['Profit']).rjust(10, ' ')
            ).rjust(25+len(config['stake_currency']))
            if x['Total profit'] != 0.0 else '--'.rjust(25+len(config['stake_currency'])),
            axis=1
        )
        trials = trials.drop(columns=['Total profit'])

        if print_colorized:
            for i in range(len(trials)):
                if trials.loc[i]['is_profit']:
                    for j in range(len(trials.loc[i])-3):
                        trials.iat[i, j] = "{}{}{}".format(Fore.GREEN,
                                                           str(trials.loc[i][j]), Fore.RESET)
                if trials.loc[i]['is_best'] and highlight_best:
                    for j in range(len(trials.loc[i])-3):
                        trials.iat[i, j] = "{}{}{}".format(Style.BRIGHT,
                                                           str(trials.loc[i][j]), Style.RESET_ALL)

        trials = trials.drop(columns=['is_initial_point', 'is_best', 'is_profit'])
        if remove_header > 0:
            table = tabulate.tabulate(
                trials.to_dict(orient='list'), tablefmt='orgtbl',
                headers='keys', stralign="right"
            )

            table = table.split("\n", remove_header)[remove_header]
        elif remove_header < 0:
            table = tabulate.tabulate(
                trials.to_dict(orient='list'), tablefmt='psql',
                headers='keys', stralign="right"
            )
            table = "\n".join(table.split("\n")[0:remove_header])
        else:
            table = tabulate.tabulate(
                trials.to_dict(orient='list'), tablefmt='psql',
                headers='keys', stralign="right"
            )
        return table

    @staticmethod
    def export_csv_file(config: dict, results: list, total_epochs: int, highlight_best: bool,
                        csv_file: str) -> None:
        """
        Log result to csv-file
        """
        if not results:
            return

        # Verification for overwrite
        if Path(csv_file).is_file():
            logger.error(f"CSV file already exists: {csv_file}")
            return

        try:
            io.open(csv_file, 'w+').close()
        except IOError:
            logger.error(f"Failed to create CSV file: {csv_file}")
            return

        trials = json_normalize(results, max_level=1)
        trials['Best'] = ''
        trials['Stake currency'] = config['stake_currency']

        base_metrics = ['Best', 'current_epoch', 'results_metrics.trade_count',
                        'results_metrics.avg_profit', 'results_metrics.median_profit',
                        'results_metrics.total_profit',
                        'Stake currency', 'results_metrics.profit', 'results_metrics.duration',
                        'loss', 'is_initial_point', 'is_best']
        param_metrics = [("params_dict."+param) for param in results[0]['params_dict'].keys()]
        trials = trials[base_metrics + param_metrics]

        base_columns = ['Best', 'Epoch', 'Trades', 'Avg profit', 'Median profit', 'Total profit',
                        'Stake currency', 'Profit', 'Avg duration', 'Objective',
                        'is_initial_point', 'is_best']
        param_columns = list(results[0]['params_dict'].keys())
        trials.columns = base_columns + param_columns

        trials['is_profit'] = False
        trials.loc[trials['is_initial_point'], 'Best'] = '*'
        trials.loc[trials['is_best'], 'Best'] = 'Best'
        trials.loc[trials['is_initial_point'] & trials['is_best'], 'Best'] = '* Best'
        trials.loc[trials['Total profit'] > 0, 'is_profit'] = True
        trials['Epoch'] = trials['Epoch'].astype(str)
        trials['Trades'] = trials['Trades'].astype(str)

        trials['Total profit'] = trials['Total profit'].apply(
            lambda x: '{:,.8f}'.format(x) if x != 0.0 else ""
        )
        trials['Profit'] = trials['Profit'].apply(
            lambda x: '{:,.2f}'.format(x) if not isna(x) else ""
        )
        trials['Avg profit'] = trials['Avg profit'].apply(
            lambda x: '{:,.2f}%'.format(x) if not isna(x) else ""
        )
        trials['Avg duration'] = trials['Avg duration'].apply(
            lambda x: '{:,.1f} m'.format(x) if not isna(x) else ""
        )
        trials['Objective'] = trials['Objective'].apply(
            lambda x: '{:,.5f}'.format(x) if x != 100000 else ""
        )

        trials = trials.drop(columns=['is_initial_point', 'is_best', 'is_profit'])
        trials.to_csv(csv_file, index=False, header=True, mode='w', encoding='UTF-8')
        logger.info(f"CSV file created: {csv_file}")
