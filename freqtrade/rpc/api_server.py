import logging
import threading
from copy import deepcopy
from datetime import date, datetime
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any, Callable, Dict

from arrow import Arrow
from flask import Flask, jsonify, request
from flask.json import JSONEncoder
from flask_cors import CORS
from flask_jwt_extended import (JWTManager, create_access_token, create_refresh_token,
                                get_jwt_identity, jwt_refresh_token_required,
                                verify_jwt_in_request_optional)
from werkzeug.security import safe_str_cmp
from werkzeug.serving import make_server

from freqtrade.__init__ import __version__
from freqtrade.constants import DATETIME_PRINT_FORMAT, USERPATH_STRATEGIES
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import Trade
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from freqtrade.rpc.rpc import RPC, RPCException


logger = logging.getLogger(__name__)

BASE_URI = "/api/v1"


class FTJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, Arrow):
                return obj.for_json()
            elif isinstance(obj, datetime):
                return obj.strftime(DATETIME_PRINT_FORMAT)
            elif isinstance(obj, date):
                return obj.strftime("%Y-%m-%d")
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return list(iterable)
        return JSONEncoder.default(self, obj)


# Type should really be Callable[[ApiServer, Any], Any], but that will create a circular dependency
def require_login(func: Callable[[Any, Any], Any]):

    def func_wrapper(obj, *args, **kwargs):
        verify_jwt_in_request_optional()
        auth = request.authorization
        if get_jwt_identity() or auth and obj.check_auth(auth.username, auth.password):
            return func(obj, *args, **kwargs)
        else:
            return jsonify({"error": "Unauthorized"}), 401

    return func_wrapper


# Type should really be Callable[[ApiServer], Any], but that will create a circular dependency
def rpc_catch_errors(func: Callable[..., Any]):

    def func_wrapper(obj, *args, **kwargs):

        try:
            return func(obj, *args, **kwargs)
        except RPCException as e:
            logger.exception("API Error calling %s: %s", func.__name__, e)
            return obj.rest_error(f"Error querying {func.__name__}: {e}")

    return func_wrapper


def shutdown_session(exception=None):
    # Remove scoped session
    Trade.session.remove()


class ApiServer(RPC):
    """
    This class runs api server and provides rpc.rpc functionality to it

    This class starts a non-blocking thread the api server runs within
    """

    def check_auth(self, username, password):
        return (safe_str_cmp(username, self._config['api_server'].get('username')) and
                safe_str_cmp(password, self._config['api_server'].get('password')))

    def __init__(self, freqtrade) -> None:
        """
        Init the api server, and init the super class RPC
        :param freqtrade: Instance of a freqtrade bot
        :return: None
        """
        super().__init__(freqtrade)

        self._config = freqtrade.config
        self.app = Flask(__name__)
        self._cors = CORS(self.app,
                          resources={r"/api/*": {
                              "supports_credentials": True,
                              "origins": self._config['api_server'].get('CORS_origins', [])}}
                          )

        # Setup the Flask-JWT-Extended extension
        self.app.config['JWT_SECRET_KEY'] = self._config['api_server'].get(
            'jwt_secret_key', 'super-secret')

        self.jwt = JWTManager(self.app)
        self.app.json_encoder = FTJSONEncoder

        self.app.teardown_appcontext(shutdown_session)

        # Register application handling
        self.register_rest_rpc_urls()

        if self._config.get('fiat_display_currency', None):
            self._fiat_converter = CryptoToFiatConverter()

        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def cleanup(self) -> None:
        logger.info("Stopping API Server")
        self.srv.shutdown()

    def run(self):
        """
        Method that runs flask app in its own thread forever.
        Section to handle configuration and running of the Rest server
        also to check and warn if not bound to a loopback, warn on security risk.
        """
        rest_ip = self._config['api_server']['listen_ip_address']
        rest_port = self._config['api_server']['listen_port']

        logger.info(f'Starting HTTP Server at {rest_ip}:{rest_port}')
        if not IPv4Address(rest_ip).is_loopback:
            logger.warning("SECURITY WARNING - Local Rest Server listening to external connections")
            logger.warning("SECURITY WARNING - This is insecure please set to your loopback,"
                           "e.g 127.0.0.1 in config.json")

        if not self._config['api_server'].get('password'):
            logger.warning("SECURITY WARNING - No password for local REST Server defined. "
                           "Please make sure that this is intentional!")

        # Run the Server
        logger.info('Starting Local Rest Server.')
        try:
            self.srv = make_server(rest_ip, rest_port, self.app)
            self.srv.serve_forever()
        except Exception:
            logger.exception("Api server failed to start.")
        logger.info('Local Rest Server started.')

    def send_msg(self, msg: Dict[str, str]) -> None:
        """
        We don't push to endpoints at the moment.
        Take a look at webhooks for that functionality.
        """
        pass

    def rest_error(self, error_msg, error_code=502):
        return jsonify({"error": error_msg}), error_code

    def register_rest_rpc_urls(self):
        """
        Registers flask app URLs that are calls to functionality in rpc.rpc.

        First two arguments passed are /URL and 'Label'
        Label can be used as a shortcut when refactoring
        :return:
        """
        self.app.register_error_handler(404, self.page_not_found)

        # Actions to control the bot
        self.app.add_url_rule(f'{BASE_URI}/token/login', 'login',
                              view_func=self._token_login, methods=['POST'])
        self.app.add_url_rule(f'{BASE_URI}/token/refresh', 'token_refresh',
                              view_func=self._token_refresh, methods=['POST'])
        self.app.add_url_rule(f'{BASE_URI}/start', 'start',
                              view_func=self._start, methods=['POST'])
        self.app.add_url_rule(f'{BASE_URI}/stop', 'stop', view_func=self._stop, methods=['POST'])
        self.app.add_url_rule(f'{BASE_URI}/stopbuy', 'stopbuy',
                              view_func=self._stopbuy, methods=['POST'])
        self.app.add_url_rule(f'{BASE_URI}/reload_config', 'reload_config',
                              view_func=self._reload_config, methods=['POST'])
        # Info commands
        self.app.add_url_rule(f'{BASE_URI}/balance', 'balance',
                              view_func=self._balance, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/count', 'count', view_func=self._count, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/locks', 'locks', view_func=self._locks, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/daily', 'daily', view_func=self._daily, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/edge', 'edge', view_func=self._edge, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/logs', 'log', view_func=self._get_logs, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/profit', 'profit',
                              view_func=self._profit, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/performance', 'performance',
                              view_func=self._performance, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/status', 'status',
                              view_func=self._status, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/version', 'version',
                              view_func=self._version, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/show_config', 'show_config',
                              view_func=self._show_config, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/ping', 'ping',
                              view_func=self._ping, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/trades', 'trades',
                              view_func=self._trades, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/trades/<int:tradeid>', 'trades_delete',
                              view_func=self._trades_delete, methods=['DELETE'])

        self.app.add_url_rule(f'{BASE_URI}/pair_candles', 'pair_candles',
                              view_func=self._analysed_candles, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/pair_history', 'pair_history',
                              view_func=self._analysed_history, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/plot_config', 'plot_config',
                              view_func=self._plot_config, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/strategies', 'strategies',
                              view_func=self._list_strategies, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/strategy/<string:strategy>', 'strategy',
                              view_func=self._get_strategy, methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/available_pairs', 'pairs',
                              view_func=self._list_available_pairs, methods=['GET'])

        # Combined actions and infos
        self.app.add_url_rule(f'{BASE_URI}/blacklist', 'blacklist', view_func=self._blacklist,
                              methods=['GET', 'POST'])
        self.app.add_url_rule(f'{BASE_URI}/whitelist', 'whitelist', view_func=self._whitelist,
                              methods=['GET'])
        self.app.add_url_rule(f'{BASE_URI}/forcebuy', 'forcebuy',
                              view_func=self._forcebuy, methods=['POST'])
        self.app.add_url_rule(f'{BASE_URI}/forcesell', 'forcesell', view_func=self._forcesell,
                              methods=['POST'])

    @require_login
    def page_not_found(self, error):
        """
        Return "404 not found", 404.
        """
        return jsonify({
            'status': 'error',
            'reason': f"There's no API call for {request.base_url}.",
            'code': 404
        }), 404

    @require_login
    @rpc_catch_errors
    def _token_login(self):
        """
        Handler for /token/login
        Returns a JWT token
        """
        auth = request.authorization
        if auth and self.check_auth(auth.username, auth.password):
            keystuff = {'u': auth.username}
            ret = {
                'access_token': create_access_token(identity=keystuff),
                'refresh_token': create_refresh_token(identity=keystuff),
            }
            return jsonify(ret)

        return jsonify({"error": "Unauthorized"}), 401

    @jwt_refresh_token_required
    @rpc_catch_errors
    def _token_refresh(self):
        """
        Handler for /token/refresh
        Returns a JWT token based on a JWT refresh token
        """
        current_user = get_jwt_identity()
        new_token = create_access_token(identity=current_user, fresh=False)

        ret = {'access_token': new_token}
        return jsonify(ret)

    @require_login
    @rpc_catch_errors
    def _start(self):
        """
        Handler for /start.
        Starts TradeThread in bot if stopped.
        """
        msg = self._rpc_start()
        return jsonify(msg)

    @require_login
    @rpc_catch_errors
    def _stop(self):
        """
        Handler for /stop.
        Stops TradeThread in bot if running
        """
        msg = self._rpc_stop()
        return jsonify(msg)

    @require_login
    @rpc_catch_errors
    def _stopbuy(self):
        """
        Handler for /stopbuy.
        Sets max_open_trades to 0 and gracefully sells all open trades
        """
        msg = self._rpc_stopbuy()
        return jsonify(msg)

    @rpc_catch_errors
    def _ping(self):
        """
        simple ping version
        """
        return jsonify({"status": "pong"})

    @require_login
    @rpc_catch_errors
    def _version(self):
        """
        Prints the bot's version
        """
        return jsonify({"version": __version__})

    @require_login
    @rpc_catch_errors
    def _show_config(self):
        """
        Prints the bot's version
        """
        return jsonify(RPC._rpc_show_config(self._config, self._freqtrade.state))

    @require_login
    @rpc_catch_errors
    def _reload_config(self):
        """
        Handler for /reload_config.
        Triggers a config file reload
        """
        msg = self._rpc_reload_config()
        return jsonify(msg)

    @require_login
    @rpc_catch_errors
    def _count(self):
        """
        Handler for /count.
        Returns the number of trades running
        """
        msg = self._rpc_count()
        return jsonify(msg)

    @require_login
    @rpc_catch_errors
    def _locks(self):
        """
        Handler for /locks.
        Returns the currently active locks.
        """
        return jsonify(self._rpc_locks())

    @require_login
    @rpc_catch_errors
    def _daily(self):
        """
        Returns the last X days trading stats summary.

        :return: stats
        """
        timescale = request.args.get('timescale', 7)
        timescale = int(timescale)

        stats = self._rpc_daily_profit(timescale,
                                       self._config['stake_currency'],
                                       self._config.get('fiat_display_currency', '')
                                       )

        return jsonify(stats)

    @require_login
    @rpc_catch_errors
    def _get_logs(self):
        """
        Returns latest logs
         get:
          param:
            limit: Only get a certain number of records
        """
        limit = int(request.args.get('limit', 0)) or None
        return jsonify(self._rpc_get_logs(limit))

    @require_login
    @rpc_catch_errors
    def _edge(self):
        """
        Returns information related to Edge.
        :return: edge stats
        """
        stats = self._rpc_edge()

        return jsonify(stats)

    @require_login
    @rpc_catch_errors
    def _profit(self):
        """
        Handler for /profit.

        Returns a cumulative profit statistics
        :return: stats
        """

        stats = self._rpc_trade_statistics(self._config['stake_currency'],
                                           self._config.get('fiat_display_currency')
                                           )

        return jsonify(stats)

    @require_login
    @rpc_catch_errors
    def _performance(self):
        """
        Handler for /performance.

        Returns a cumulative performance statistics
        :return: stats
        """
        stats = self._rpc_performance()

        return jsonify(stats)

    @require_login
    @rpc_catch_errors
    def _status(self):
        """
        Handler for /status.

        Returns the current status of the trades in json format
        """
        try:
            results = self._rpc_trade_status()
            return jsonify(results)
        except RPCException:
            return jsonify([])

    @require_login
    @rpc_catch_errors
    def _balance(self):
        """
        Handler for /balance.

        Returns the current status of the trades in json format
        """
        results = self._rpc_balance(self._config['stake_currency'],
                                    self._config.get('fiat_display_currency', ''))
        return jsonify(results)

    @require_login
    @rpc_catch_errors
    def _trades(self):
        """
        Handler for /trades.

        Returns the X last trades in json format
        """
        limit = int(request.args.get('limit', 0))
        results = self._rpc_trade_history(limit)
        return jsonify(results)

    @require_login
    @rpc_catch_errors
    def _trades_delete(self, tradeid: int):
        """
        Handler for DELETE /trades/<tradeid> endpoint.
        Removes the trade from the database (tries to cancel open orders first!)
        get:
          param:
            tradeid: Numeric trade-id assigned to the trade.
        """
        result = self._rpc_delete(tradeid)
        return jsonify(result)

    @require_login
    @rpc_catch_errors
    def _whitelist(self):
        """
        Handler for /whitelist.
        """
        results = self._rpc_whitelist()
        return jsonify(results)

    @require_login
    @rpc_catch_errors
    def _blacklist(self):
        """
        Handler for /blacklist.
        """
        add = request.json.get("blacklist", None) if request.method == 'POST' else None
        results = self._rpc_blacklist(add)
        return jsonify(results)

    @require_login
    @rpc_catch_errors
    def _forcebuy(self):
        """
        Handler for /forcebuy.
        """
        asset = request.json.get("pair")
        price = request.json.get("price", None)
        price = float(price) if price is not None else price

        trade = self._rpc_forcebuy(asset, price)
        if trade:
            return jsonify(trade.to_json())
        else:
            return jsonify({"status": f"Error buying pair {asset}."})

    @require_login
    @rpc_catch_errors
    def _forcesell(self):
        """
        Handler for /forcesell.
        """
        tradeid = request.json.get("tradeid")
        results = self._rpc_forcesell(tradeid)
        return jsonify(results)

    @require_login
    @rpc_catch_errors
    def _analysed_candles(self):
        """
        Handler for /pair_candles.
        Returns the dataframe the bot is using during live/dry operations.
        Takes the following get arguments:
        get:
          parameters:
            - pair: Pair
            - timeframe: Timeframe to get data for (should be aligned to strategy.timeframe)
            - limit: Limit return length to the latest X candles
        """
        pair = request.args.get("pair")
        timeframe = request.args.get("timeframe")
        limit = request.args.get("limit", type=int)
        if not pair or not timeframe:
            return self.rest_error("Mandatory parameter missing.", 400)

        results = self._rpc_analysed_dataframe(pair, timeframe, limit)
        return jsonify(results)

    @require_login
    @rpc_catch_errors
    def _analysed_history(self):
        """
        Handler for /pair_history.
        Returns the dataframe of a given timerange
        Takes the following get arguments:
        get:
          parameters:
            - pair: Pair
            - timeframe: Timeframe to get data for (should be aligned to strategy.timeframe)
            - strategy: Strategy to use - Must exist in configured strategy-path!
            - timerange: timerange in the format YYYYMMDD-YYYYMMDD (YYYYMMDD- or (-YYYYMMDD))
                         are als possible. If omitted uses all available data.
        """
        pair = request.args.get("pair")
        timeframe = request.args.get("timeframe")
        timerange = request.args.get("timerange")
        strategy = request.args.get("strategy")

        if not pair or not timeframe or not timerange or not strategy:
            return self.rest_error("Mandatory parameter missing.", 400)

        config = deepcopy(self._config)
        config.update({
            'strategy': strategy,
        })
        results = RPC._rpc_analysed_history_full(config, pair, timeframe, timerange)
        return jsonify(results)

    @require_login
    @rpc_catch_errors
    def _plot_config(self):
        """
        Handler for /plot_config.
        """
        return jsonify(self._rpc_plot_config())

    @require_login
    @rpc_catch_errors
    def _list_strategies(self):
        directory = Path(self._config.get(
            'strategy_path', self._config['user_data_dir'] / USERPATH_STRATEGIES))
        from freqtrade.resolvers.strategy_resolver import StrategyResolver
        strategy_objs = StrategyResolver.search_all_objects(directory, False)
        strategy_objs = sorted(strategy_objs, key=lambda x: x['name'])

        return jsonify({'strategies': [x['name'] for x in strategy_objs]})

    @require_login
    @rpc_catch_errors
    def _get_strategy(self, strategy: str):
        """
        Get a single strategy
        get:
          parameters:
            - strategy: Only get this strategy
        """
        config = deepcopy(self._config)
        from freqtrade.resolvers.strategy_resolver import StrategyResolver
        try:
            strategy_obj = StrategyResolver._load_strategy(strategy, config,
                                                           extra_dir=config.get('strategy_path'))
        except OperationalException:
            return self.rest_error("Strategy not found.", 404)

        return jsonify({
            'strategy': strategy_obj.get_strategy_name(),
            'code': strategy_obj.__source__,
         })

    @require_login
    @rpc_catch_errors
    def _list_available_pairs(self):
        """
        Handler for /available_pairs.
        Returns an object, with pairs, available pair length and pair_interval combinations
        Takes the following get arguments:
        get:
          parameters:
            - stake_currency: Filter on this stake currency
            - timeframe: Timeframe to get data for Filter elements to this timeframe
        """
        timeframe = request.args.get("timeframe")
        stake_currency = request.args.get("stake_currency")

        from freqtrade.data.history import get_datahandler
        dh = get_datahandler(self._config['datadir'], self._config.get('dataformat_ohlcv', None))

        pair_interval = dh.ohlcv_get_available_data(self._config['datadir'])

        if timeframe:
            pair_interval = [pair for pair in pair_interval if pair[1] == timeframe]
        if stake_currency:
            pair_interval = [pair for pair in pair_interval if pair[0].endswith(stake_currency)]
        pair_interval = sorted(pair_interval, key=lambda x: x[0])

        pairs = list({x[0] for x in pair_interval})

        result = {
            'length': len(pairs),
            'pairs': pairs,
            'pair_interval': pair_interval,
        }
        return jsonify(result)
