#!/usr/bin/env python3
import math, sys, os, time, pp, math, re
from io import StringIO
# tuple of all parallel python servers to connect with
ppservers = ()
#ppservers = ("10.0.0.1",)







def backtesting(ind):
    er1 = str(ind)
    ou1 = str(ind * 1024)
    import threading, traceback
    from io import StringIO
    from freqtrade.main import main, set_loggers
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    ind1 = sys.stdout = StringIO()
    ind2 = sys.stderr = StringIO()
    dat = threading.Thread(target=main(['backtesting']))
    dat.start()
    dat.join()
    er1 = ind2.getvalue()
    ou1 = ind1.getvalue()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    return er1, ou1

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print("Starting pp with", job_server.get_ncpus(), "workers")


start_time = time.time()

# Since jobs are not equal in the execution time, division of the problem 
# into a 100 of small subproblems leads to a better load balancing
parts = 128
rounds = 128
jobs = []
current = 0
for index in range(rounds):
    for index in range(parts):
        jobs.append(job_server.submit(backtesting, (index,)))
    job_server.wait()
    for job in jobs:
        res = job()
        string = str(res)
        params = re.search(r'~~~~(.*)~~~~', string).group(1)
        mfi = re.search(r'MFI Value(.*)XXX', string)
        fastd = re.search(r'FASTD Value(.*)XXX', string)
        adx = re.search(r'ADX Value(.*)XXX', string)
        rsi = re.search(r'RSI Value(.*)XXX', string)
        tot = re.search(r'TOTAL(.*)', string).group(1)
        total = re.search(r'[-+]?([0-9]*\.[0-9]+|[0-9]+)', tot).group(1)
        if total and (float(total) > float(current)):
            current = total
            print('total better profit paremeters:  ')
            print(total)
            if params:
                print(params)
                print('~~~~~~')
                print('Only enable the above settings, not all settings below are used!')
                print('~~~~~~')
            if mfi:
                print('~~~MFI~~~')
                print(mfi.group(1))
            if fastd:
                print('~~~FASTD~~~')
                print(fastd.group(1))
            if adx:
                print('~~~ADX~~~')
                print(adx.group(1))
            if rsi:
                print('~~~RSI~~~')
                print(rsi.group(1))






    jobs = []
print("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()
