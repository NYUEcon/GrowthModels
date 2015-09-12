#!/bin/bash
#PBS -l nodes=8:ppn=20
#PBS -l walltime=9:59:59
#PBS -N PleaseConverge
#PBS -M spencer.lyon@nyu.edu
#PBS -m abe

module purge

# this moves us to the directory where qsub was submitted
# should be $WORK/Research/GrowthModels/Models/TwoAgent
cd $PBS_O_WORKDIR

cat $PBS_NODEFILE > my_machines

# run the code! -- use machinefile to start one julia on each process
$HOME/src/julia/bin/julia --machinefile my_machines bigwig.jl
