
import io
import logging
from pathlib import Path
from typing import Any, Dict, List

import rapidjson
import tabulate
from colorama import Fore, Style
from pandas import isna, json_normalize

from freqtrade.exceptions import OperationalException
from freqtrade.misc import round_coin_value, round_dict


logger = logging.getLogger(__name__)


class HyperoptTools():

    @staticmethod
    def has_space(config: Dict[str, Any], space: str) -> bool:
        """
        Tell if the space value is contained in the configuration
        """
        # The 'trailing' space is not included in the 'default' set of spaces
        if space == 'trailing':
            return any(s in config['spaces'] for s in [space, 'all'])
        else:
            return any(s in config['spaces'] for s in [space, 'all', 'default'])

    @staticmethod
    def _read_results_pickle(results_file: Path) -> List:
        """
        Read hyperopt results from pickle file
        LEGACY method - new files are written as json and cannot be read with this method.
        """
        from joblib import load

        logger.info(f"Reading pickled epochs from '{results_file}'")
        data = load(results_file)
        return data

    @staticmethod
    def _read_results(results_file: Path) -> List:
        """
        Read hyperopt results from file
        """
        import rapidjson
        logger.info(f"Reading epochs from '{results_file}'")
        with results_file.open('r') as f:
            data = [rapidjson.loads(line) for line in f]
        return data

    @staticmethod
    def load_previous_results(results_file: Path) -> List:
        """
        Load data for epochs from the file if we have one
        """
        epochs: List = []
        if results_file.is_file() and results_file.stat().st_size > 0:
            if results_file.suffix == '.pickle':
                epochs = HyperoptTools._read_results_pickle(results_file)
            else:
                epochs = HyperoptTools._read_results(results_file)
            # Detection of some old format, without 'is_best' field saved
            if epochs[0].get('is_best') is None:
                raise OperationalException(
                    "The file with HyperoptTools results is incompatible with this version "
                    "of Freqtrade and cannot be loaded.")
            logger.info(f"Loaded {len(epochs)} previous evaluations from disk.")
        return epochs

    @staticmethod
    def show_epoch_details(results, total_epochs: int, print_json: bool,
                           no_header: bool = False, header_str: str = None) -> None:
        """
        Display details of the hyperopt result
        """
        params = results.get('params_details', {})
        non_optimized = results.get('params_not_optimized', {})

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
            HyperoptTools._params_pretty_print(params, 'buy', "Buy hyperspace params:",
                                               non_optimized)
            HyperoptTools._params_pretty_print(params, 'sell', "Sell hyperspace params:",
                                               non_optimized)
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
                # Convert keys in min_roi dict to strings because
                # rapidjson cannot dump dicts with integer keys...
                result_dict['minimal_roi'] = {str(k): v for k, v in space_params.items()}
            else:  # 'stoploss', 'trailing'
                result_dict.update(space_params)

    @staticmethod
    def _params_pretty_print(params, space: str, header: str, non_optimized={}) -> None:
        if space in params or space in non_optimized:
            space_params = HyperoptTools._space_params(params, space, 5)
            result = f"\n# {header}\n"
            if space == 'stoploss':
                result += f"stoploss = {space_params.get('stoploss')}"
            elif space == 'roi':
                minimal_roi_result = rapidjson.dumps({
                        str(k): v for k, v in space_params.items()
                }, default=str, indent=4, number_mode=rapidjson.NM_NATIVE)
                result += f"minimal_roi = {minimal_roi_result}"
            elif space == 'trailing':

                for k, v in space_params.items():
                    result += f'{k} = {v}\n'

            else:
                no_params = HyperoptTools._space_params(non_optimized, space, 5)

                result += f"{space}_params = {HyperoptTools._pprint(space_params, no_params)}"

            result = result.replace("\n", "\n    ")
            print(result)

    @staticmethod
    def _space_params(params, space: str, r: int = None) -> Dict:
        d = params.get(space)
        if d:
            # Round floats to `r` digits after the decimal point if requested
            return round_dict(d, r) if r else d
        return {}

    @staticmethod
    def _pprint(params, non_optimized, indent: int = 4):
        """
        Pretty-print hyperopt results (based on 2 dicts - with add. comment)
        """
        p = params.copy()
        p.update(non_optimized)
        result = '{\n'

        for k, param in p.items():
            result += " " * indent + f'"{k}": '
            result += f'"{param}",' if isinstance(param, str) else f'{param},'
            if k in non_optimized:
                result += "  # value loaded from strategy"
            result += "\n"
        result += '}'
        return result

    @staticmethod
    def is_best_loss(results, current_best_loss: float) -> bool:
        return bool(results['loss'] < current_best_loss)

    @staticmethod
    def format_results_explanation_string(results_metrics: Dict, stake_currency: str) -> str:
        """
        Return the formatted results explanation in a string
        """
        return (f"{results_metrics['total_trades']:6d} trades. "
                f"{results_metrics['wins']}/{results_metrics['draws']}"
                f"/{results_metrics['losses']} Wins/Draws/Losses. "
                f"Avg profit {results_metrics['profit_mean'] * 100: 6.2f}%. "
                f"Median profit {results_metrics['profit_median'] * 100: 6.2f}%. "
                f"Total profit {results_metrics['profit_total_abs']: 11.8f} {stake_currency} "
                f"({results_metrics['profit_total'] * 100: 7.2f}%). "
                f"Avg duration {results_metrics['holding_avg']} min."
                )

    @staticmethod
    def _format_explanation_string(results, total_epochs) -> str:
        return (("*" if results['is_initial_point'] else " ") +
                f"{results['current_epoch']:5d}/{total_epochs}: " +
                f"{results['results_explanation']} " +
                f"Objective: {results['loss']:.5f}")

    @staticmethod
    def prepare_trials_columns(trials, legacy_mode: bool, has_drawdown: bool) -> str:

        trials['Best'] = ''

        if 'results_metrics.winsdrawslosses' not in trials.columns:
            # Ensure compatibility with older versions of hyperopt results
            trials['results_metrics.winsdrawslosses'] = 'N/A'

        if not has_drawdown:
            # Ensure compatibility with older versions of hyperopt results
            trials['results_metrics.max_drawdown_abs'] = None
            trials['results_metrics.max_drawdown'] = None

        if not legacy_mode:
            # New mode, using backtest result for metrics
            trials['results_metrics.winsdrawslosses'] = trials.apply(
                lambda x: f"{x['results_metrics.wins']} {x['results_metrics.draws']:>4} "
                          f"{x['results_metrics.losses']:>4}", axis=1)
            trials = trials[['Best', 'current_epoch', 'results_metrics.total_trades',
                             'results_metrics.winsdrawslosses',
                             'results_metrics.profit_mean', 'results_metrics.profit_total_abs',
                             'results_metrics.profit_total', 'results_metrics.holding_avg',
                             'results_metrics.max_drawdown', 'results_metrics.max_drawdown_abs',
                             'loss', 'is_initial_point', 'is_best']]

        else:
            # Legacy mode
            trials = trials[['Best', 'current_epoch', 'results_metrics.trade_count',
                             'results_metrics.winsdrawslosses', 'results_metrics.avg_profit',
                             'results_metrics.total_profit', 'results_metrics.profit',
                             'results_metrics.duration', 'results_metrics.max_drawdown',
                             'results_metrics.max_drawdown_abs', 'loss', 'is_initial_point',
                             'is_best']]

        trials.columns = ['Best', 'Epoch', 'Trades', ' Win Draw Loss', 'Avg profit',
                          'Total profit', 'Profit', 'Avg duration', 'Max Drawdown',
                          'max_drawdown_abs', 'Objective', 'is_initial_point', 'is_best']

        return trials

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

        legacy_mode = 'results_metrics.total_trades' not in trials
        has_drawdown = 'results_metrics.max_drawdown_abs' in trials.columns

        trials = HyperoptTools.prepare_trials_columns(trials, legacy_mode, has_drawdown)

        trials['is_profit'] = False
        trials.loc[trials['is_initial_point'], 'Best'] = '*     '
        trials.loc[trials['is_best'], 'Best'] = 'Best'
        trials.loc[trials['is_initial_point'] & trials['is_best'], 'Best'] = '* Best'
        trials.loc[trials['Total profit'] > 0, 'is_profit'] = True
        trials['Trades'] = trials['Trades'].astype(str)
        perc_multi = 1 if legacy_mode else 100
        trials['Epoch'] = trials['Epoch'].apply(
            lambda x: '{}/{}'.format(str(x).rjust(len(str(total_epochs)), ' '), total_epochs)
        )
        trials['Avg profit'] = trials['Avg profit'].apply(
            lambda x: f'{x * perc_multi:,.2f}%'.rjust(7, ' ') if not isna(x) else "--".rjust(7, ' ')
        )
        trials['Avg duration'] = trials['Avg duration'].apply(
            lambda x: f'{x:,.1f} m'.rjust(7, ' ') if isinstance(x, float) else f"{x}"
                      if not isna(x) else "--".rjust(7, ' ')
        )
        trials['Objective'] = trials['Objective'].apply(
            lambda x: f'{x:,.5f}'.rjust(8, ' ') if x != 100000 else "N/A".rjust(8, ' ')
        )

        stake_currency = config['stake_currency']

        if has_drawdown:
            trials['Max Drawdown'] = trials.apply(
                lambda x: '{} {}'.format(
                    round_coin_value(x['max_drawdown_abs'], stake_currency),
                    '({:,.2f}%)'.format(x['Max Drawdown'] * perc_multi).rjust(10, ' ')
                ).rjust(25 + len(stake_currency))
                if x['Max Drawdown'] != 0.0 else '--'.rjust(25 + len(stake_currency)),
                axis=1
            )
        else:
            trials = trials.drop(columns=['Max Drawdown'])

        trials = trials.drop(columns=['max_drawdown_abs'])

        trials['Profit'] = trials.apply(
            lambda x: '{} {}'.format(
                round_coin_value(x['Total profit'], stake_currency),
                '({:,.2f}%)'.format(x['Profit'] * perc_multi).rjust(10, ' ')
            ).rjust(25+len(stake_currency))
            if x['Total profit'] != 0.0 else '--'.rjust(25+len(stake_currency)),
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

        if 'results_metrics.total_trades' in trials:
            base_metrics = ['Best', 'current_epoch', 'results_metrics.total_trades',
                            'results_metrics.profit_mean', 'results_metrics.profit_median',
                            'results_metrics.profit_total',
                            'Stake currency',
                            'results_metrics.profit_total_abs', 'results_metrics.holding_avg',
                            'loss', 'is_initial_point', 'is_best']
            perc_multi = 100
        else:
            perc_multi = 1
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
        trials['Median profit'] = trials['Median profit'] * perc_multi

        trials['Total profit'] = trials['Total profit'].apply(
            lambda x: f'{x:,.8f}' if x != 0.0 else ""
        )
        trials['Profit'] = trials['Profit'].apply(
            lambda x: f'{x:,.2f}' if not isna(x) else ""
        )
        trials['Avg profit'] = trials['Avg profit'].apply(
            lambda x: f'{x * perc_multi:,.2f}%' if not isna(x) else ""
        )
        if perc_multi == 1:
            trials['Avg duration'] = trials['Avg duration'].apply(
                lambda x: f'{x:,.1f} m' if isinstance(
                    x, float) else f"{x.total_seconds() // 60:,.1f} m" if not isna(x) else ""
            )
        trials['Objective'] = trials['Objective'].apply(
            lambda x: f'{x:,.5f}' if x != 100000 else ""
        )

        trials = trials.drop(columns=['is_initial_point', 'is_best', 'is_profit'])
        trials.to_csv(csv_file, index=False, header=True, mode='w', encoding='UTF-8')
        logger.info(f"CSV file created: {csv_file}")
