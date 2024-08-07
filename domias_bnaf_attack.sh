#!/bin/bash


#sdg = sys.argv[2]
#epsilon = sys.argv[3]
#n_size = sys.argv[4]
#data = sys.argv[5]



#experiment A

nohup python3 domias_bnaf.py mst 0.10 10000 snake > outfiles/bnaf_0.10.txt &
nohup python3 domias_bnaf.py mst 0.32 10000 snake > outfiles/bnaf_0.32.txt &
nohup python3 domias_bnaf.py mst 1.00 10000 snake > outfiles/bnaf_1.00.txt &
nohup python3 domias_bnaf.py mst 3.16 10000 snake > outfiles/bnaf_3.16.txt &
nohup python3 domias_bnaf.py mst 10.00 10000 snake > outfiles/bnaf_10.00.txt &
nohup python3 domias_bnaf.py mst 31.62 10000 snake > outfiles/bnaf_31.62.txt &
nohup python3 domias_bnaf.py mst 100.00 10000 snake > outfiles/bnaf_100.00.txt &
nohup python3 domias_bnaf.py mst 316.23 10000 snake > outfiles/bnaf_316.23.txt &
nohup python3 domias_bnaf.py mst 1000.00 10000 snake > outfiles/bnaf_1000.00.txt &

nohup python3 domias_bnaf.py priv 0.10 10000 snake > outfiles/bnaf_priv_0.10.txt &
nohup python3 domias_bnaf.py priv 0.32 10000 snake > outfiles/bnaf_priv_0.32.txt &
nohup python3 domias_bnaf.py priv 1.00 10000 snake > outfiles/bnaf_priv_1.00.txt &
nohup python3 domias_bnaf.py priv 3.16 10000 snake > outfiles/bnaf_priv_3.16.txt &
nohup python3 domias_bnaf.py priv 10.00 10000 snake > outfiles/bnaf_priv_10.00.txt &
nohup python3 domias_bnaf.py priv 31.62 10000 snake > outfiles/bnaf_priv_31.62.txt &
nohup python3 domias_bnaf.py priv 100.00 10000 snake > outfiles/bnaf_priv_100.00.txt &
nohup python3 domias_bnaf.py priv 316.23 10000 snake > outfiles/bnaf_priv_316.23.txt &
nohup python3 domias_bnaf.py priv 1000.00 10000 snake > outfiles/bnaf_priv_1000.00.txt &

nohup python3 domias_bnaf.py gsd 0.10 10000 snake > outfiles/bnaf_gsd_0.10.txt &
nohup python3 domias_bnaf.py gsd 0.32 10000 snake > outfiles/bnaf_gsd_0.32.txt &
nohup python3 domias_bnaf.py gsd 1.00 10000 snake > outfiles/bnaf_gsd_1.00.txt &
nohup python3 domias_bnaf.py gsd 3.16 10000 snake > outfiles/bnaf_gsd_3.16.txt &
nohup python3 domias_bnaf.py gsd 10.00 10000 snake > outfiles/bnaf_gsd_10.00.txt &
nohup python3 domias_bnaf.py gsd 31.62 10000 snake > outfiles/bnaf_gsd_31.62.txt &
nohup python3 domias_bnaf.py gsd 100.00 10000 snake > outfiles/bnaf_gsd_100.00.txt &
nohup python3 domias_bnaf.py gsd 316.23 10000 snake > outfiles/bnaf_gsd_316.23.txt &
nohup python3 domias_bnaf.py gsd 1000.00 10000 snake > outfiles/bnaf_gsd_1000.00.txt &



#experiment B

nohup python3 domias_bnaf.py mst 10.00 100 snake > outfiles/bnaf_100.txt &
nohup python3 domias_bnaf.py mst 10.00 316 snake > outfiles/bnaf_316.txt &
nohup python3 domias_bnaf.py mst 10.00 1000 snake > outfiles/bnaf_1000.txt &
nohup python3 domias_bnaf.py mst 10.00 3162 snake > outfiles/bnaf_3162.txt &
#nohup python3 domias_bnaf.py mst 10.00 10000 snake > outfiles/bnaf_10000.txt &
nohup python3 domias_bnaf.py mst 10.00 31623 snake > outfiles/bnaf_31623.txt &

nohup python3 domias_bnaf.py priv 10.00 100 snake > outfiles/bnaf_priv_100.txt &
nohup python3 domias_bnaf.py priv 10.00 316 snake > outfiles/bnaf_priv_316.txt &
nohup python3 domias_bnaf.py priv 10.00 1000 snake > outfiles/bnaf_priv_1000.txt &
nohup python3 domias_bnaf.py priv 10.00 3162 snake > outfiles/bnaf_priv_3162.txt &
#nohup python3 domias_bnaf.py priv 10.00 10000 snake > outfiles/bnaf_priv_10000.txt &
nohup python3 domias_bnaf.py priv 10.00 31623 snake > outfiles/bnaf_priv_31623.txt &

nohup python3 domias_bnaf.py gsd 10.00 100 snake > outfiles/bnaf_gsd_100.txt &
nohup python3 domias_bnaf.py gsd 10.00 316 snake > outfiles/bnaf_gsd_316.txt &
nohup python3 domias_bnaf.py gsd 10.00 1000 snake > outfiles/bnaf_gsd_1000.txt &
nohup python3 domias_bnaf.py gsd 10.00 3162 snake > outfiles/bnaf_gsd_3162.txt &
#nohup python3 domias_bnaf.py gsd 10.00 10000 snake > outfiles/bnaf_gsd_10000.txt &
#nohup python3 domias_bnaf.py gsd 10.00 31623 snake > outfiles/bnaf_gsd_31623.txt &



#experiment D

nohup python3 domias_bnaf.py mst 0.10 1000 snake > outfiles/bnaf_0.10.txt &
nohup python3 domias_bnaf.py mst 0.32 1000 snake > outfiles/bnaf_0.32.txt &
nohup python3 domias_bnaf.py mst 1.00 1000 snake > outfiles/bnaf_1.00.txt &
nohup python3 domias_bnaf.py mst 3.16 1000 snake > outfiles/bnaf_3.16.txt &
#nohup python3 domias_bnaf.py mst 10.00 1000 snake > outfiles/bnaf_10.00.txt & # already attacked in experiment B
nohup python3 domias_bnaf.py mst 31.62 1000 snake > outfiles/bnaf_31.62.txt &
nohup python3 domias_bnaf.py mst 100.00 1000 snake > outfiles/bnaf_100.00.txt &
nohup python3 domias_bnaf.py mst 316.23 1000 snake > outfiles/bnaf_316.23.txt &
nohup python3 domias_bnaf.py mst 1000.00 1000 snake > outfiles/bnaf_1000.00.txt &

nohup python3 domias_bnaf.py priv 0.10 1000 snake > outfiles/bnaf_priv_0.10.txt &
nohup python3 domias_bnaf.py priv 0.32 1000 snake > outfiles/bnaf_priv_0.32.txt &
nohup python3 domias_bnaf.py priv 1.00 1000 snake > outfiles/bnaf_priv_1.00.txt &
nohup python3 domias_bnaf.py priv 3.16 1000 snake > outfiles/bnaf_priv_3.16.txt &
#nohup python3 domias_bnaf.py priv 10.00 1000 snake > outfiles/bnaf_priv_10.00.txt & # already attacked in experiment B
nohup python3 domias_bnaf.py priv 31.62 1000 snake > outfiles/bnaf_priv_31.62.txt &
nohup python3 domias_bnaf.py priv 100.00 1000 snake > outfiles/bnaf_priv_100.00.txt &
nohup python3 domias_bnaf.py priv 316.23 1000 snake > outfiles/bnaf_priv_316.23.txt &
nohup python3 domias_bnaf.py priv 1000.00 1000 snake > outfiles/bnaf_priv_1000.00.txt &

nohup python3 domias_bnaf.py gsd 0.10 1000 snake > outfiles/bnaf_gsd_0.10.txt &
nohup python3 domias_bnaf.py gsd 0.32 1000 snake > outfiles/bnaf_gsd_0.32.txt &
nohup python3 domias_bnaf.py gsd 1.00 1000 snake > outfiles/bnaf_gsd_1.00.txt &
nohup python3 domias_bnaf.py gsd 3.16 1000 snake > outfiles/bnaf_gsd_3.16.txt &
#nohup python3 domias_bnaf.py gsd 10.00 1000 snake > outfiles/bnaf_gsd_10.00.txt & # already attacked in experiment B
nohup python3 domias_bnaf.py gsd 31.62 1000 snake > outfiles/bnaf_gsd_31.62.txt &
nohup python3 domias_bnaf.py gsd 100.00 1000 snake > outfiles/bnaf_gsd_100.00.txt &
nohup python3 domias_bnaf.py gsd 316.23 1000 snake > outfiles/bnaf_gsd_316.23.txt &
nohup python3 domias_bnaf.py gsd 1000.00 1000 snake > outfiles/bnaf_gsd_1000.00.txt &


# cali data


nohup python3 domias_bnaf.py mst 0.10 1000 cali > outfiles/bnaf_0.10.txt &
nohup python3 domias_bnaf.py mst 0.32 1000 cali > outfiles/bnaf_0.32.txt &
nohup python3 domias_bnaf.py mst 1.00 1000 cali > outfiles/bnaf_1.00.txt &
nohup python3 domias_bnaf.py mst 3.16 1000 cali > outfiles/bnaf_3.16.txt &
nohup python3 domias_bnaf.py mst 10.00 1000 cali > outfiles/bnaf_10.00.txt &
nohup python3 domias_bnaf.py mst 31.62 1000 cali > outfiles/bnaf_31.62.txt &
nohup python3 domias_bnaf.py mst 100.00 1000 cali > outfiles/bnaf_100.00.txt &
nohup python3 domias_bnaf.py mst 316.23 1000 cali > outfiles/bnaf_316.23.txt &
nohup python3 domias_bnaf.py mst 1000.00 1000 cali > outfiles/bnaf_1000.00.txt &

nohup python3 domias_bnaf.py priv 0.10 1000 cali > outfiles/bnaf_priv_0.10.txt &
nohup python3 domias_bnaf.py priv 0.32 1000 cali > outfiles/bnaf_priv_0.32.txt &
nohup python3 domias_bnaf.py priv 1.00 1000 cali > outfiles/bnaf_priv_1.00.txt &
nohup python3 domias_bnaf.py priv 3.16 1000 cali > outfiles/bnaf_priv_3.16.txt &
nohup python3 domias_bnaf.py priv 10.00 1000 cali > outfiles/bnaf_priv_10.00.txt &
nohup python3 domias_bnaf.py priv 31.62 1000 cali > outfiles/bnaf_priv_31.62.txt &
nohup python3 domias_bnaf.py priv 100.00 1000 cali > outfiles/bnaf_priv_100.00.txt &
nohup python3 domias_bnaf.py priv 316.23 1000 cali > outfiles/bnaf_priv_316.23.txt &
nohup python3 domias_bnaf.py priv 1000.00 1000 cali > outfiles/bnaf_priv_1000.00.txt &

nohup python3 domias_bnaf.py gsd 0.10 1000 cali > outfiles/bnaf_gsd_0.10.txt &
nohup python3 domias_bnaf.py gsd 0.32 1000 cali > outfiles/bnaf_gsd_0.32.txt &
nohup python3 domias_bnaf.py gsd 1.00 1000 cali > outfiles/bnaf_gsd_1.00.txt &
nohup python3 domias_bnaf.py gsd 3.16 1000 cali > outfiles/bnaf_gsd_3.16.txt &
nohup python3 domias_bnaf.py gsd 10.00 1000 cali > outfiles/bnaf_gsd_10.00.txt &
nohup python3 domias_bnaf.py gsd 31.62 1000 cali > outfiles/bnaf_gsd_31.62.txt &
nohup python3 domias_bnaf.py gsd 100.00 1000 cali > outfiles/bnaf_gsd_100.00.txt &
nohup python3 domias_bnaf.py gsd 316.23 1000 cali > outfiles/bnaf_gsd_316.23.txt &
nohup python3 domias_bnaf.py gsd 1000.00 1000 cali > outfiles/bnaf_gsd_1000.00.txt &




#
#
## additional domias experiments
#
## no overlapping aux
#nohup python3 domias_bnaf.py mst 0.10 1000 snake False False > outfiles/bnaf_0.10.txt &
#nohup python3 domias_bnaf.py mst 0.32 1000 snake False False > outfiles/bnaf_0.32.txt &
#nohup python3 domias_bnaf.py mst 1.00 1000 snake False False > outfiles/bnaf_1.00.txt &
#nohup python3 domias_bnaf.py mst 3.16 1000 snake False False > outfiles/bnaf_3.16.txt &
#nohup python3 domias_bnaf.py mst 10.00 1000 snake False False > outfiles/bnaf_10.00.txt &
#nohup python3 domias_bnaf.py mst 31.62 1000 snake False False > outfiles/bnaf_31.62.txt &
#nohup python3 domias_bnaf.py mst 100.00 1000 snake False False > outfiles/bnaf_100.00.txt &
#nohup python3 domias_bnaf.py mst 316.23 1000 snake False False > outfiles/bnaf_316.23.txt &
#nohup python3 domias_bnaf.py mst 1000.00 1000 snake False False > outfiles/bnaf_1000.00.txt &
#
#nohup python3 domias_bnaf.py priv 0.10 1000 snake False False > outfiles/bnaf_priv_0.10.txt &
#nohup python3 domias_bnaf.py priv 0.32 1000 snake False False > outfiles/bnaf_priv_0.32.txt &
#nohup python3 domias_bnaf.py priv 1.00 1000 snake False False > outfiles/bnaf_priv_1.00.txt &
#nohup python3 domias_bnaf.py priv 3.16 1000 snake False False > outfiles/bnaf_priv_3.16.txt &
#nohup python3 domias_bnaf.py priv 10.00 1000 snake False False > outfiles/bnaf_priv_10.00.txt &
#nohup python3 domias_bnaf.py priv 31.62 1000 snake False False > outfiles/bnaf_priv_31.62.txt &
#nohup python3 domias_bnaf.py priv 100.00 1000 snake False False > outfiles/bnaf_priv_100.00.txt &
#nohup python3 domias_bnaf.py priv 316.23 1000 snake False False > outfiles/bnaf_priv_316.23.txt &
#nohup python3 domias_bnaf.py priv 1000.00 1000 snake False False > outfiles/bnaf_priv_1000.00.txt &
#
#nohup python3 domias_bnaf.py gsd 0.10 1000 snake False False > outfiles/bnaf_gsd_0.10.txt &
#nohup python3 domias_bnaf.py gsd 0.32 1000 snake False False > outfiles/bnaf_gsd_0.32.txt &
#nohup python3 domias_bnaf.py gsd 1.00 1000 snake False False > outfiles/bnaf_gsd_1.00.txt &
#nohup python3 domias_bnaf.py gsd 3.16 1000 snake False False > outfiles/bnaf_gsd_3.16.txt &
#nohup python3 domias_bnaf.py gsd 10.00 1000 snake False False > outfiles/bnaf_gsd_10.00.txt &
#nohup python3 domias_bnaf.py gsd 31.62 1000 snake False False > outfiles/bnaf_gsd_31.62.txt &
#nohup python3 domias_bnaf.py gsd 100.00 1000 snake False False > outfiles/bnaf_gsd_100.00.txt &
#nohup python3 domias_bnaf.py gsd 316.23 1000 snake False False > outfiles/bnaf_gsd_316.23.txt &
#nohup python3 domias_bnaf.py gsd 1000.00 1000 snake False False > outfiles/bnaf_gsd_1000.00.txt &
#
#
## set MI
#nohup python3 domias_bnaf.py mst 0.10 1000 snake True True > outfiles/bnaf_0.10.txt &
#nohup python3 domias_bnaf.py mst 0.32 1000 snake True True > outfiles/bnaf_0.32.txt &
#nohup python3 domias_bnaf.py mst 1.00 1000 snake True True > outfiles/bnaf_1.00.txt &
#nohup python3 domias_bnaf.py mst 3.16 1000 snake True True > outfiles/bnaf_3.16.txt &
#nohup python3 domias_bnaf.py mst 10.00 1000 snake True True > outfiles/bnaf_10.00.txt & # snake FPs already determined in experiment B
#nohup python3 domias_bnaf.py mst 31.62 1000 snake True True > outfiles/bnaf_31.62.txt &
#nohup python3 domias_bnaf.py mst 100.00 1000 snake True True > outfiles/bnaf_100.00.txt &
#nohup python3 domias_bnaf.py mst 316.23 1000 snake True True > outfiles/bnaf_316.23.txt &
#nohup python3 domias_bnaf.py mst 1000.00 1000 snake True True > outfiles/bnaf_1000.00.txt &
#
#nohup python3 domias_bnaf.py priv 0.10 1000 snake True True > outfiles/bnaf_priv_0.10.txt &
#nohup python3 domias_bnaf.py priv 0.32 1000 snake True True > outfiles/bnaf_priv_0.32.txt &
#nohup python3 domias_bnaf.py priv 1.00 1000 snake True True > outfiles/bnaf_priv_1.00.txt &
#nohup python3 domias_bnaf.py priv 3.16 1000 snake True True > outfiles/bnaf_priv_3.16.txt &
#nohup python3 domias_bnaf.py priv 10.00 1000 snake True True > outfiles/bnaf_priv_10.00.txt & # snake FPs already determined in experiment B
#nohup python3 domias_bnaf.py priv 31.62 1000 snake True True > outfiles/bnaf_priv_31.62.txt &
#nohup python3 domias_bnaf.py priv 100.00 1000 snake True True > outfiles/bnaf_priv_100.00.txt &
#nohup python3 domias_bnaf.py priv 316.23 1000 snake True True > outfiles/bnaf_priv_316.23.txt &
#nohup python3 domias_bnaf.py priv 1000.00 1000 snake True True > outfiles/bnaf_priv_1000.00.txt &
#
#nohup python3 domias_bnaf.py gsd 0.10 1000 snake True True > outfiles/bnaf_gsd_0.10.txt &
#nohup python3 domias_bnaf.py gsd 0.32 1000 snake True True > outfiles/bnaf_gsd_0.32.txt &
#nohup python3 domias_bnaf.py gsd 1.00 1000 snake True True > outfiles/bnaf_gsd_1.00.txt &
#nohup python3 domias_bnaf.py gsd 3.16 1000 snake True True > outfiles/bnaf_gsd_3.16.txt &
#nohup python3 domias_bnaf.py gsd 10.00 1000 snake True True > outfiles/bnaf_gsd_10.00.txt & # snake FPs already determined in experiment B
#nohup python3 domias_bnaf.py gsd 31.62 1000 snake True True > outfiles/bnaf_gsd_31.62.txt &
#nohup python3 domias_bnaf.py gsd 100.00 1000 snake True True > outfiles/bnaf_gsd_100.00.txt &
#nohup python3 domias_bnaf.py gsd 316.23 1000 snake True True > outfiles/bnaf_gsd_316.23.txt &
#nohup python3 domias_bnaf.py gsd 1000.00 1000 snake True True > outfiles/bnaf_gsd_1000.00.txt &




