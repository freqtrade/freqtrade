Basically, to sum things up:

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

Get this raw file and replace your hyperopy.py with this one in freqtrade/optmize/:

https://raw.githubusercontent.com/MoonGem/freqtrade/develop/freqtrade/optimize/hyperopt.py

then run the mongodb scripts, wait until the worker shows that it does not appear to be working then ctrl+c the worker, however, this is as far as I could get. I think this is the correct process.
However, the queue does eventually empty after awhile and you shouldn't need to ctrlc the worker.
Kinda buggy I guess still.

You may also need to install the latest hyperopt from their github to avoid subscription error:

pip3.6 install git+https://github.com/hyperopt/hyperopt.git


Lastly, copy hyperopt-mongodb.py over freqtrade/optimise/hyperopt.py then run scripts.
