
count=$1
for i in $(seq $count); do
    nohup python3 gen_tapas_shadowsets.py gen $2 gsd . > outfiles/gen_b_$2_$i.txt &
done


