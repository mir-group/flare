#!/bin/bash
set -e

# reference file
~/lammps/build/lmp -in in.si > /dev/null
mv si.dump si_fasit.dump

# loop through all lines, check that there's an atom
# of the same type in the same place with the same forces etc.
# in the reference file
# (MPI reorders atoms, diff doesn't work)
function checkdump {
    awk '{
    if (NF==15){
        pattern = $2;
        for(i = 3; i <= NF; i++){
            pattern = (pattern " " $i);
        }
        grepcmd = "grep -q \"" pattern"\" si_fasit.dump";
        ret = system(grepcmd);
        #print pattern;
        if (ret > 0) {
            exit 1;
        }
    }
    }' si.dump
}

for neigh in full
do
    for mpi in 1 2
    do
        echo "MPI=$mpi, CUDA=no, neigh=$neigh"
        mpirun -np $mpi ~/lammps/ompbuild/lmp -sf kk -pk kokkos newton on neigh $neigh -k on t 2 -in in.si > /dev/null
        checkdump && echo correct || echo wrong
        echo "MPI=$mpi, CUDA=yes, neigh=$neigh"
        CUDA_LAUNCH_BLOCKING=1 mpirun -np $mpi ~/lammps/cudabuild/lmp -k on g 1 -sf kk -pk kokkos newton on neigh $neigh -in in.si > /dev/null
        checkdump && echo correct || echo wrong
    done
done
