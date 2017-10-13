#!/usr/bin/python3.6

from urllib.request import urlopen

currencies = ["ok","neo","dash","etc","eth","snt"]

for cur in currencies:
	url = 'https://bittrex.com/Api/v2.0/pub/market/GetTicks?marketName=BTC-'+cur+'&tickInterval=fiveMin'
	x = urlopen(url)
	data = x.read()
	str2 = str(data,'utf-8')
	with open("btc-"+cur+".json", "w") as fichier:
		fichier.write(str2)
