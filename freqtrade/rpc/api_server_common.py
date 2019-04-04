import logging
import flask
from flask import request, jsonify

logger = logging.getLogger(__name__)


class MyApiApp(flask.Flask):
    def __init__(self, import_name):
        """
        Contains common rest routes and resource that do not need
        to access to rpc.rpc functionality
        """
        super(MyApiApp, self).__init__(import_name)

        """
        Registers flask app URLs that are not calls to functionality in rpc.rpc.
        :return:
        """
        self.before_request(self.my_preprocessing)
        self.register_error_handler(404, self.page_not_found)
        self.add_url_rule('/', 'hello', view_func=self.hello, methods=['GET'])
        self.add_url_rule('/stop_api', 'stop_api', view_func=self.stop_api, methods=['GET'])

    def my_preprocessing(self):
        # Do stuff to flask.request
        pass

    def page_not_found(self, error):
        # Return "404 not found", 404.
        return jsonify({'status': 'error',
                        'reason': '''There's no API call for %s''' % request.base_url,
                        'code': 404}), 404

    def hello(self):
        """
        None critical but helpful default index page.

        That lists URLs added to the flask server.
        This may be deprecated at any time.
        :return: index.html
        """
        rest_cmds = 'Commands implemented: <br>' \
                    '<a href=/daily?timescale=7>Show 7 days of stats</a>' \
                    '<br>' \
                    '<a href=/stop>Stop the Trade thread</a>' \
                    '<br>' \
                    '<a href=/start>Start the Traded thread</a>' \
                    '<br>' \
                    '<a href=/paypal> 404 page does not exist</a>' \
                    '<br>' \
                    '<br>' \
                    '<a href=/stop_api>Shut down the api server -  be sure</a>'
        return rest_cmds

    def stop_api(self):
        """ For calling shutdown_api_server over via api server HTTP"""
        self.shutdown_api_server()
        return 'Api Server shutting down... '

    def shutdown_api_server(self):
        """
        Stop the running flask application

        Records the shutdown in logger.info
        :return:
        """
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running the Flask Werkzeug Server')
        if func is not None:
            logger.info('Stopping the Local Rest Server')
            func()
            return
