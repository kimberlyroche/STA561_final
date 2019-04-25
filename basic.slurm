#!/bin/bash
#SBATCH -J PR2_knn
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#

module add Anaconda3/4.3.0-gcb01

source activate sta561_final

cd /home/ker48/STA561_final

#python pr1.py
#python pr2.py cosine
#python pr2.py euclid
python pr2.py knn
#python pr5.py
