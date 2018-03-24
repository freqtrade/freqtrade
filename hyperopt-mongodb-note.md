Basically, to sum things up:

The locations of your hyperopt directories may vary. Some reside in .local.

Make these changes per this PR in these files:

https://github.com/hyperopt/hyperopt/pull/287/files

/usr/local/lib/python3.6/dist-packages/hyperopt/fmin.py

removetrial['state'] == base.JOB_STATE_RUNNING

addtrial['state'] = base.JOB_STATE_RUNNING

Make these changes per this PR into the hyperopt files:

https://github.com/hyperopt/hyperopt/pull/288/files

remove:import six.moves.cPickle as pickle

add: import dill as pickle in the following files:

/usr/local/lib/python3.6/dist-packages/hyperopt/fmin.py

/usr/local/lib/python3.6/dist-packages/hyperopt/main.py

/usr/local/lib/python3.6/dist-packages/hyperopt/mongoexp.py

/usr/local/lib/python3.6/dist-packages/hyperopt/utils.py

Install dill:

pip3.6 install dill

replace freqtrade/optimize/hyperopt.py with the hyperopt-mongodb.py in the freqtrade root.

Management tool for mongodb:

https://robomongo.org/download


Then, run the scipts. 5000 epochs can take a few hours even on a high-end machine. You can use the mongodb took above to check the progress on the local machine.


Lastly, read the test_strategy.py file located in user_data/strategy/test_strategy.py

Once you've done that, run scripts. You can restart the worker and the freqtrader hyperopt process but I do not advise it. Monitor the output with mongodb.


