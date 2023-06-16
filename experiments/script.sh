#!/bin/zsh

date
python main.py -c config/small_100.json     -o data/data_small_100.json
python main.py -c config/small_1000.json    -o data/data_small_1000.json
python main.py -c config/small_3000.json    -o data/data_small_3000.json
python main.py -c config/small_10000.json   -o data/data_small_10000.json
python main.py -c config/medium_10000.json  -o data/data_medium_10000.json
python main.py -c config/medium_30000.json  -o data/data_medium_30000.json
python main.py -c config/medium_100000.json -o data/data_medium_100000.json
python main.py -c config/large_10000.json   -o data/data_large_10000.json
python main.py -c config/large_30000.json   -o data/data_large_30000.json
python main.py -c config/large_100000.json  -o data/data_large_100000.json
date