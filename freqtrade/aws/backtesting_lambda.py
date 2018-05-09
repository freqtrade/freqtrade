from freqtrade.optimize.backtesting import Backtesting

def backtest(event, context):
    """
        this method is running on the AWS server
        and back tests this application for us
        and stores the back testing results in a local database

        this event can be given as:

        :param event:
            {
                'strategy' : 'url handle where we can find the strategy'
                'pair' : ' pair to backtest, BTC_ETH as example'
                'timeframe' : 'how long should we backtest for, 0-100 as example for the last 100 ticks'
            }
        :param context:
            standard AWS context, so pleaes ignore for now!
        :return:
            no return
    """


    backtesting = Backtesting()
    backtesting.start()

    pass

