#!/bin/bash

#experiment A

nohup python3 AAAI_mamamia_experiments.py shadowmodel A mst 0.10 > A_0.10.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A mst 0.32 > A_0.32.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A mst 1.00 > A_1.00.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A mst 3.16 > A_3.16.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A mst 10.00 > A_10.00.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A mst 31.62 > A_31.62.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A mst 100.00 > A_100.00.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A mst 316.23 > A_316.23.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A mst 1000.00 > A_1000.00.txt &

nohup python3 AAAI_mamamia_experiments.py shadowmodel A priv 0.10 > A_priv_0.10.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A priv 0.32 > A_priv_0.32.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A priv 1.00 > A_priv_1.00.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A priv 3.16 > A_priv_3.16.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A priv 10.00 > A_priv_10.00.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A priv 31.62 > A_priv_31.62.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A priv 100.00 > A_priv_100.00.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A priv 316.23 > A_priv_316.23.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A priv 1000.00 > A_priv_1000.00.txt &

nohup python3 AAAI_mamamia_experiments.py shadowmodel A gsd 0.10 > A_gsd_0.10.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A gsd 0.32 > A_gsd_0.32.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A gsd 1.00 > A_gsd_1.00.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A gsd 3.16 > A_gsd_3.16.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A gsd 10.00 > A_gsd_10.00.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A gsd 31.62 > A_gsd_31.62.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A gsd 100.00 > A_gsd_100.00.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A gsd 316.23 > A_gsd_316.23.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel A gsd 1000.00 > A_gsd_1000.00.txt &

#nohup python3 AAAI_mamamia_experiments.py shadowmodel A rap 0.10 > A_rap_0.10.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel A rap 0.32 > A_rap_0.32.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel A rap 1.00 > A_rap_1.00.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel A rap 3.16 > A_rap_3.16.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel A rap 10.00 > A_rap_10.00.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel A rap 31.62 > A_rap_31.62.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel A rap 100.00 > A_rap_100.00.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel A rap 316.23 > A_rap_316.23.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel A rap 1000.00 > A_rap_1000.00.txt &


#experiment B

nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 100 > A_100.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 316 > A_316.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 1000 > A_1000.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 3162 > A_3162.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 10000 > A_10000.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 31623 > A_31623.txt &

nohup python3 AAAI_mamamia_experiments.py shadowmodel B priv 100 > A_priv_100.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B priv 316 > A_priv_316.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B priv 1000 > A_priv_1000.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B priv 3162 > A_priv_3162.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel B priv 10000 > A_priv_10000.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B priv 31623 > A_priv_31623.txt &

nohup python3 AAAI_mamamia_experiments.py shadowmodel B gsd 100 > A_gsd_100.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B gsd 316 > A_gsd_316.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B gsd 1000 > A_gsd_1000.txt &
nohup python3 AAAI_mamamia_experiments.py shadowmodel B gsd 3162 > A_gsd_3162.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel B gsd 10000 > A_gsd_10000.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel B gsd 31623 > A_gsd_31623.txt &

#nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 100 > A_100.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 316 > A_316.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 1000 > A_1000.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 3162 > A_3162.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 10000 > A_10000.txt &
#nohup python3 AAAI_mamamia_experiments.py shadowmodel B mst 31623 > A_31623.txt &


#experiment D

