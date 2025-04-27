#!/bin/bash

mkdir -p outerr

module purge
module load OpenMPI/4.1.5-GCC-12.3.0

export MPICC=(which mpicc)

PATH_TO_CODE="<PATH_TO_BONSAI_DATA_REPRESENTATION_FOLDER>"
SCRIPT_PATH="bonsai_bash.sh"

ID="bonsai_example"
NCORES=5
TIME="00:30:00"
QOS=30min
MEM_PER_CORE=4

mem=$((NCORES*MEM_PER_CORE))G

file="yaml_paths.txt"
line_count=$(wc -l < "$file")

for (( i=1; i<=$line_count; i++ ))
do
  YAML_PATH=$(head -n ${i} ./yaml_paths.txt | tail -1)

  echo Starting Bonsai based on configuration file: ${YAML_PATH}

  preprocess_id=$(sbatch --export ALL  --parsable --mem=${mem} --cpus-per-task=1 --ntasks=${NCORES} --nodes=1 --time=00:30:00 \
  --output=./outerr/pre_dataset${i}%x_%j.out \
  --error=./outerr/pre_dataset${i}%x_%j.err --qos=30min \
  --job-name="${ID}"_${i} \
  ${SCRIPT_PATH} -n ${NCORES} -s preprocess -y ${YAML_PATH})

  #preprocess_id is a jobID; pass to next call as dependency
  #Start main tree reconstruction after preprocess_id=ok

 core_calc_id=$(sbatch --export ALL  --parsable --mem=${mem} --cpus-per-task=1 --ntasks=${NCORES} --nodes=1 --time=${TIME} \
  --dependency=afterok:$preprocess_id \
  --output=./outerr/core_dataset${i}%x_%j.out \
  --error=./outerr/core_dataset${i}%x_%j.err --qos=${QOS} \
  --job-name="${ID}"_${i} \
  --mail-type=END,FAIL \
  --mail-user=<YOUR_EMAIL> \
  ${SCRIPT_PATH} -n ${NCORES} -s core_calc -y ${YAML_PATH})

 sbatch --export ALL  --mem=${mem} --cpus-per-task=1 --ntasks=1 --time=00:30:00 \
  --dependency=afterok:$core_calc_id \
  --output=./outerr/metadata_dataset${i}%x_%j.out \
  --error=./outerr/metadata_dataset${i}%x_%j.err --qos=30min \
  --job-name="${ID}"_${i} \
  ${SCRIPT_PATH} -n ${NCORES} -s metadata -y ${YAML_PATH}

done
