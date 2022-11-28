lmp="${HOME}/lammps/build/lmp -k on g 1 -sf kk -pk kokkos newton on neigh full"

for L in $(seq 1 9) $(seq 10 5 29) $(seq 30 10 89) $(seq 90 20 150)
do
    #L=$(echo "2^$i" | bc -l)
    N=$((8 * $L * $L * $L))
    sw=$($lmp -var L $L -in in.sw 2> /dev/null | awk '/Performance/ {print $6;}')
    flare=$($lmp -var L $L -in in.bench 2> /dev/null | awk '/Performance/ {print $6;}')
    normsw=$(echo "$sw*$N" | bc -l)
    normflare=$(echo "$flare*$N" | bc -l)
    echo $L $sw $flare $N $normsw $normflare
done
