#!/bin/bash

#experiment A

nohup python3 mamamia_experiments.py shadowmodel A mst 0.10 > outfiles/A_0.10.txt &
nohup python3 mamamia_experiments.py shadowmodel A mst 0.32 > outfiles/A_0.32.txt &
nohup python3 mamamia_experiments.py shadowmodel A mst 1.00 > outfiles/A_1.00.txt &
nohup python3 mamamia_experiments.py shadowmodel A mst 3.16 > outfiles/A_3.16.txt &
nohup python3 mamamia_experiments.py shadowmodel A mst 10.00 > outfiles/A_10.00.txt &
nohup python3 mamamia_experiments.py shadowmodel A mst 31.62 > outfiles/A_31.62.txt &
nohup python3 mamamia_experiments.py shadowmodel A mst 100.00 > outfiles/A_100.00.txt &
nohup python3 mamamia_experiments.py shadowmodel A mst 316.23 > outfiles/A_316.23.txt &
nohup python3 mamamia_experiments.py shadowmodel A mst 1000.00 > outfiles/A_1000.00.txt &

nohup python3 mamamia_experiments.py shadowmodel A priv 0.10 > outfiles/A_priv_0.10.txt &
nohup python3 mamamia_experiments.py shadowmodel A priv 0.32 > outfiles/A_priv_0.32.txt &
nohup python3 mamamia_experiments.py shadowmodel A priv 1.00 > outfiles/A_priv_1.00.txt &
nohup python3 mamamia_experiments.py shadowmodel A priv 3.16 > outfiles/A_priv_3.16.txt &
nohup python3 mamamia_experiments.py shadowmodel A priv 10.00 > outfiles/A_priv_10.00.txt &
nohup python3 mamamia_experiments.py shadowmodel A priv 31.62 > outfiles/A_priv_31.62.txt &
nohup python3 mamamia_experiments.py shadowmodel A priv 100.00 > outfiles/A_priv_100.00.txt &
nohup python3 mamamia_experiments.py shadowmodel A priv 316.23 > outfiles/A_priv_316.23.txt &
nohup python3 mamamia_experiments.py shadowmodel A priv 1000.00 > outfiles/A_priv_1000.00.txt &

nohup python3 mamamia_experiments.py shadowmodel A gsd 0.10 > outfiles/A_gsd_0.10.txt &
nohup python3 mamamia_experiments.py shadowmodel A gsd 0.32 > outfiles/A_gsd_0.32.txt &
nohup python3 mamamia_experiments.py shadowmodel A gsd 1.00 > outfiles/A_gsd_1.00.txt &
nohup python3 mamamia_experiments.py shadowmodel A gsd 3.16 > outfiles/A_gsd_3.16.txt &
nohup python3 mamamia_experiments.py shadowmodel A gsd 10.00 > outfiles/A_gsd_10.00.txt &
nohup python3 mamamia_experiments.py shadowmodel A gsd 31.62 > outfiles/A_gsd_31.62.txt &
nohup python3 mamamia_experiments.py shadowmodel A gsd 100.00 > outfiles/A_gsd_100.00.txt &
nohup python3 mamamia_experiments.py shadowmodel A gsd 316.23 > outfiles/A_gsd_316.23.txt &
nohup python3 mamamia_experiments.py shadowmodel A gsd 1000.00 > outfiles/A_gsd_1000.00.txt &

#nohup python3 mamamia_experiments.py shadowmodel A rap 0.10 > outfiles/A_rap_0.10.txt &
#nohup python3 mamamia_experiments.py shadowmodel A rap 0.32 > outfiles/A_rap_0.32.txt &
#nohup python3 mamamia_experiments.py shadowmodel A rap 1.00 > outfiles/A_rap_1.00.txt &
#nohup python3 mamamia_experiments.py shadowmodel A rap 3.16 > outfiles/A_rap_3.16.txt &
#nohup python3 mamamia_experiments.py shadowmodel A rap 10.00 > outfiles/A_rap_10.00.txt &
#nohup python3 mamamia_experiments.py shadowmodel A rap 31.62 > outfiles/A_rap_31.62.txt &
#nohup python3 mamamia_experiments.py shadowmodel A rap 100.00 > outfiles/A_rap_100.00.txt &
#nohup python3 mamamia_experiments.py shadowmodel A rap 316.23 > outfiles/A_rap_316.23.txt &
#nohup python3 mamamia_experiments.py shadowmodel A rap 1000.00 > outfiles/A_rap_1000.00.txt &


#experiment B

nohup python3 mamamia_experiments.py shadowmodel B mst 100 > outfiles/B_100.txt &
nohup python3 mamamia_experiments.py shadowmodel B mst 316 > outfiles/B_316.txt &
nohup python3 mamamia_experiments.py shadowmodel B mst 1000 > outfiles/B_1000.txt &
nohup python3 mamamia_experiments.py shadowmodel B mst 3162 > outfiles/B_3162.txt &
#nohup python3 mamamia_experiments.py shadowmodel B mst 10000 > outfiles/B_10000.txt &
nohup python3 mamamia_experiments.py shadowmodel B mst 31623 > outfiles/B_31623.txt &

nohup python3 mamamia_experiments.py shadowmodel B priv 100 > outfiles/B_priv_100.txt &
nohup python3 mamamia_experiments.py shadowmodel B priv 316 > outfiles/B_priv_316.txt &
nohup python3 mamamia_experiments.py shadowmodel B priv 1000 > outfiles/B_priv_1000.txt &
nohup python3 mamamia_experiments.py shadowmodel B priv 3162 > outfiles/B_priv_3162.txt &
#nohup python3 mamamia_experiments.py shadowmodel B priv 10000 > outfiles/B_priv_10000.txt &
nohup python3 mamamia_experiments.py shadowmodel B priv 31623 > outfiles/B_priv_31623.txt &

nohup python3 mamamia_experiments.py shadowmodel B gsd 100 > outfiles/B_gsd_100.txt &
nohup python3 mamamia_experiments.py shadowmodel B gsd 316 > outfiles/B_gsd_316.txt &
nohup python3 mamamia_experiments.py shadowmodel B gsd 1000 > outfiles/B_gsd_1000.txt &
nohup python3 mamamia_experiments.py shadowmodel B gsd 3162 > outfiles/B_gsd_3162.txt &
#nohup python3 mamamia_experiments.py shadowmodel B gsd 10000 > outfiles/B_gsd_10000.txt &
#nohup python3 mamamia_experiments.py shadowmodel B gsd 31623 > outfiles/B_gsd_31623.txt &

#nohup python3 mamamia_experiments.py shadowmodel B rap 100 > outfiles/B_rap_100.txt &
#nohup python3 mamamia_experiments.py shadowmodel B rap 316 > outfiles/B_rap_316.txt &
#nohup python3 mamamia_experiments.py shadowmodel B rap 1000 > outfiles/B_rap_1000.txt &
#nohup python3 mamamia_experiments.py shadowmodel B rap 3162 > outfiles/B_rap_3162.txt &
#nohup python3 mamamia_experiments.py shadowmodel B rap 10000 > outfiles/B_rap_10000.txt &
#nohup python3 mamamia_experiments.py shadowmodel B rap 31623 > outfiles/B_rap_31623.txt &


#experiment D

nohup python3 mamamia_experiments.py shadowmodel D mst 0.10 > outfiles/D_0.10.txt &
nohup python3 mamamia_experiments.py shadowmodel D mst 0.32 > outfiles/D_0.32.txt &
nohup python3 mamamia_experiments.py shadowmodel D mst 1.00 > outfiles/D_1.00.txt &
nohup python3 mamamia_experiments.py shadowmodel D mst 3.16 > outfiles/D_3.16.txt &
nohup python3 mamamia_experiments.py shadowmodel D mst 10.00 True > outfiles/D_10.00.txt & # snake FPs already determined in experiment B
nohup python3 mamamia_experiments.py shadowmodel D mst 31.62 > outfiles/D_31.62.txt &
nohup python3 mamamia_experiments.py shadowmodel D mst 100.00 > outfiles/D_100.00.txt &
nohup python3 mamamia_experiments.py shadowmodel D mst 316.23 > outfiles/D_316.23.txt &
nohup python3 mamamia_experiments.py shadowmodel D mst 1000.00 > outfiles/D_1000.00.txt &

nohup python3 mamamia_experiments.py shadowmodel D priv 0.10 > outfiles/D_priv_0.10.txt &
nohup python3 mamamia_experiments.py shadowmodel D priv 0.32 > outfiles/D_priv_0.32.txt &
nohup python3 mamamia_experiments.py shadowmodel D priv 1.00 > outfiles/D_priv_1.00.txt &
nohup python3 mamamia_experiments.py shadowmodel D priv 3.16 > outfiles/D_priv_3.16.txt &
nohup python3 mamamia_experiments.py shadowmodel D priv 10.00 True > outfiles/D_priv_10.00.txt & # snake FPs already determined in experiment B
nohup python3 mamamia_experiments.py shadowmodel D priv 31.62 > outfiles/D_priv_31.62.txt &
nohup python3 mamamia_experiments.py shadowmodel D priv 100.00 > outfiles/D_priv_100.00.txt &
nohup python3 mamamia_experiments.py shadowmodel D priv 316.23 > outfiles/D_priv_316.23.txt &
nohup python3 mamamia_experiments.py shadowmodel D priv 1000.00 > outfiles/D_priv_1000.00.txt &

nohup python3 mamamia_experiments.py shadowmodel D gsd 0.10 > outfiles/D_gsd_0.10.txt &
nohup python3 mamamia_experiments.py shadowmodel D gsd 0.32 > outfiles/D_gsd_0.32.txt &
nohup python3 mamamia_experiments.py shadowmodel D gsd 1.00 > outfiles/D_gsd_1.00.txt &
nohup python3 mamamia_experiments.py shadowmodel D gsd 3.16 > outfiles/D_gsd_3.16.txt &
nohup python3 mamamia_experiments.py shadowmodel D gsd 10.00 True > outfiles/D_gsd_10.00.txt & # snake FPs already determined in experiment B
nohup python3 mamamia_experiments.py shadowmodel D gsd 31.62 > outfiles/D_gsd_31.62.txt &
nohup python3 mamamia_experiments.py shadowmodel D gsd 100.00 > outfiles/D_gsd_100.00.txt &
nohup python3 mamamia_experiments.py shadowmodel D gsd 316.23 > outfiles/D_gsd_316.23.txt &
nohup python3 mamamia_experiments.py shadowmodel D gsd 1000.00 > outfiles/D_gsd_1000.00.txt &


#nohup python3 mamamia_experiments.py shadowmodel D rap 0.10 > outfiles/D_rap_0.10.txt &
#nohup python3 mamamia_experiments.py shadowmodel D rap 0.32 > outfiles/D_rap_0.32.txt &
#nohup python3 mamamia_experiments.py shadowmodel D rap 1.00 > outfiles/D_rap_1.00.txt &
#nohup python3 mamamia_experiments.py shadowmodel D rap 3.16 > outfiles/D_rap_3.16.txt &
#nohup python3 mamamia_experiments.py shadowmodel D rap 10.00 > outfiles/D_rap_10.00.txt &
#nohup python3 mamamia_experiments.py shadowmodel D rap 31.62 > outfiles/D_rap_31.62.txt &
#nohup python3 mamamia_experiments.py shadowmodel D rap 100.00 > outfiles/D_rap_100.00.txt &
#nohup python3 mamamia_experiments.py shadowmodel D rap 316.23 > outfiles/D_rap_316.23.txt &
#nohup python3 mamamia_experiments.py shadowmodel D rap 1000.00 > outfiles/D_rap_1000.00.txt &

