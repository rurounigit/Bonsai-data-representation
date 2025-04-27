# *Bonsai*-data-representation
Algorithm to create an accurate and interpretable data-representation by reconstructing the maximum likelihood cell-tree from single-cell data, and visualizing this using the *Bonsai-scout* app.

## Input
The *Bonsai* tree-reconstruction was originally designed for the analysis of single-cell RNA-sequencing data, but can be used on any high-dimensional dataset that contains objects with with high-dimensional feature vectors. *Bonsai* needs to be provided with a vector of the most likely feature-values for each object, and takes into account standard-deviations on these feature estimates if provided, see the *Bonsai*-publication for details.

When using *Bonsai* for scRNAseq-data, we highly recommend using *Sanity* for processing of the raw counts [Sanity's GitHub-page](https://github.com/jmbreda/Sanity), or to upload your data to our [single-cell pipeline](https://bonsai.unibas.ch). If running *Bonsai* after *Sanity*, ***Bonsai* just requires the files in the output-directory created by *Sanity***, and can be run with the argument `--input_is_sanity_output True`. If running *Bonsai* based on other data, read the Section on [Running *Bonsai* on other data-types](https://github.com/dhdegroot/Bonsai-data-representation#running-bonsai-on-other-data-types).

## Installing *Bonsai*
1. Clone the GitHub repository:
```
git clone https://github.com/dhdegroot/Bonsai-data-representation.git
```

2. Create a conda environment:
```
conda create --name bonsai python=3.9 -y
conda activate bonsai
```

3. Install mpi4py
For bigger datasets (~2000 cells and larger), we recommend running *Bonsai* using parallel computation. In that case, one needs to install *mpi4py* and therefore an underlying installation of *OpenMPI*. 
For running *Bonsai* on a computing cluster, see the below information on [Installing *mpi4py* on an HPC Cluster](https://github.com/dhdegroot/Bonsai-data-representation/blob/main/README.md#installing-openmpi-on-an-hpc-cluster). Running parallel computation on your personal computer will only lead to modest speed-up, but documentation for installing mpi4py can be found at [mpi4py GitHub](https://github.com/mpi4py/mpi4py/blob/master/INSTALL.rst).
Important note: Even with a failed installation of *mpi4py*, *Bonsai* will run, but it will only make use of 1 computing core.

4. Navigate to the cloned GitHub repository:
```
cd <PathToLocalBonsaiRepository>
```

5. Install the required *Python* packages by
```
pip install -r requirements.txt
```

### Installing *OpenMPI* on an HPC Cluster
We here follow the steps as described on [Installing OpenMPI on HPC Clusters](https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py):

* Load the MPI version you want to use, we recommend using a relatively new version of OpenMPI. In order to get a list of all the available Open MPI versions on the cluster, run:
```
module avail openmpi
```
* Replace the version numbers in the line below if this version is not available, and then run the command:
```
module purge
module load OpenMPI/4.1.5-GCC-12.3.0
```
(When you eventually will start a *Bonsai* run, you will need to do this `module load` command again (for example in a *Slurm*-script). You will there also need to replace <x.y.z> by the correct version number.
* Set the loaded version of MPI to be used with mpi4py:
```
export MPICC=$(which mpicc)
```
Finally, we install mpi4py using `pip`:
```
pip install mpi4py==4.0.0 --no-cache-dir
```
It could be that you received the "Could not build wheels for mpi4py"-error. In that case, you may follow the steps below to complete installation by swapping which version of `ld` `pip` will use:
```
cd /home/<NetID>/.conda/envs/fast-mpi4py/compiler_compat
rm -f ld
ln -s /usr/bin/ld ld
```
Now you can rerun the pip install command above and it should succeed. Afterwards, you should point "ld" in your conda environment back to what it was:
```
cd /home/<NetID>/.conda/envs/fast-mpi4py/compiler_compat
rm -f ld
ln -s ../bin/x86_64-conda-linux-gnu-ld ld
```
Regardless of which version of Python you're using, you can check that mpi4py was properly installed by running:
```
python -c "import mpi4py"
```
If this command does not give an error-message, *mpi4py* was successfully installed.

## Running *Bonsai*
In this section, we will go over the steps to recontruct a Bonsai data-representation. These steps are also followed in the example below [(Example 1: Your first *Bonsai* run)](https://github.com/dhdegroot/Bonsai-data-representation?tab=readme-ov-file#example-1-your-first-bonsai-run), so as a good test if *Bonsai* and its dependencies were properly installed, you can follow that example.

The *Bonsai*-algorithm is started by calling the `bonsai_main.py`-script from the command line. As an argument, 
it needs a path to a YAML-file with run configurations, which you need to modify for your specific run first. 
### Preparing the configuration-file
The easiest way to create the configurations file is to use the provided script that can be found in 
`Bonsai-data-representation/bonsai/create_config_file.py`. One can for example run it by specifying only the path to where
you want to store this YAML-file. From the `Bonsai-data-representation` directory, run:
```
python3 bonsai/create_config_file.py --new_yaml_path <YOUR_DESIRED_CONFIG_FOLDER>/new_yaml.yaml
```
This will create a file at the specified path in which one can change the desired configurations with any text editor. 
In the YAML-file, one can find extensive explanations of the different run configurations.

Alternatively, one can give the desired configurations as arguments to the `create_config_file`-script. 
From the main directory "Bonsai-data-representation" one can for example run (note that all paths should be either absolute or 
relative to the "Bonsai-data-representation"-folder):
```
python3 bonsai/create_config_file.py \
  --new_yaml_path examples/new_yaml.yaml \
  --dataset simulated_binary_6_gens_samplingNoise \
  --verbose True \
  --data_folder examples/example_data/simulated_binary_6_gens_samplingNoise/ \
  --results_folder examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/ \
  --input_is_sanity_output True \
  --zscore_cutoff 1.0 \
  --UB_ellipsoid_size 1.0 \
  --skip_greedy_merging False \
  --skip_redo_starry False \
  --skip_opt_times False \
  --skip_nnn_reordering False \
  --nnn_n_randommoves 1000 \
  --nnn_n_randomtrees 2 \
  --pickup_intermediate False \
  --use_knn 10 \
  --filenames_data delta.txt,d_delta.txt
```
*See the Section [Possible run configurations](https://github.com/dhdegroot/Bonsai-data-representation/blob/main/README.md#possible-run-configurations) for information on the different configurations that you can change. **For first usage, one only needs to customize the configurations `dataset`, `data_folder`, `filenames_data`, `results_folder`, `input_is_sanity_output`. The other configurations are only required for advanced usage.***

### Running Bonsai on a single core
Once you have prepared the YAML-file, you only need to start the *Bonsai* run. 
It now takes only two arguments. The argument `--config_filepath` is strictly required and should contain the path to the configuration-file (either relative to the "Bonsai-data-representation"-directory, or absolute. The `--step`-argument is optional, and allows us to cut the calculation up into bricks that have different computational resource requirements, see [Example 2](https://github.com/dhdegroot/Bonsai-data-representation/blob/main/README.md#example-2-running-bonsai-on-an-hpc-cluster) below, for an example. Setting `--step all` will just run the full algorithm at once. From the "Bonsai-data-representation"-directory, one thus runs:
```
python3 bonsai/bonsai_main.py \
  --config_filepath <YOUR_DESIRED_CONFIG_FOLDER>/new_yaml.yaml \ 
  --step all
```

### Running Bonsai using parallel computation
For a parallel run, one uses the same arguments, except that we call the script through *OpenMPI* like:
```
mpiexec -n <NUMBER_OF_CORES> python -m mpi4py bonsai_main.py --config_filepath <YOUR_DESIRED_CONFIG_FOLDER>/new_yaml.yaml --step all
```

### The *Bonsai* results
The results will be stored in the results-folder as indicated in the .yaml-file. It will contain the following type of results:
* **a subdirectory starting with `final_bonsai_`.** This directory will contain the final state of the tree stored in two ways. 
  - First, there is a `.nwk`-file which is a universally accepted format for storing trees. 
  - Second, the files `edgeInfo.txt`, `vertInfo.txt`, `metadata.json` together contain a more human-readable form of storing the tree. These files are necessary for visualizing the tree using the `bonsai_scout`-app.
    - `edgeInfo.txt` contains three columns, and every row defines an edge in the tree. The first two columns indicate the `vertInd` (vertex-index) of the vertices connected by the edge, the third column indicates the length of the edge
    - `vertInfo.txt` contains for each vertex its `vertInd` and its `vertName`. The `vertName` is important for coupling the original single-cell data to the tree. The column `nodeInd` is a historical artefact, and its usage will be removed in the near future.
    - `metadata.json` contains some metadata on the dataset, importantly, it also contains a path to in what folder the original gene expression data can be found.
  - Third, the files `posterior_ltqs_vertByGene.npy` and `posterior_ltqsVars_vertByGene.npy` contain the inferred gene-expression values for all nodes, i.e. also the internal nodes, when we have marginalized over all other node positions. These values thus really indicate our best guess of the gene expression levels in that node. These data could be reconstructed from the original data as well, but storing it here in a binary-format saves time when we want to visualize the tree with the data.
* several subdirectories starting with `intermediate_bonsai_`. In these folders, the state of the reconstructed tree is stored as it was after the different steps in the *Bonsai* reconstruction algorithm. These directories are very useful when one wants to run only part of the *Bonsai*-algorithm again from one of the intermediate steps, for example when the run failed due to running out of time or memory. Since the data-files `..._vertByGene.npy` are removed from these subdirectories, the memory footprint of these folders is small. Still, one may choose to remove these directories when the full *Bonsai* run has completed.
* a copy of the configuration YAML-file with an added timestamp. In this way, one can always see with what parameters the results were created, and when this run was started.

## Running Bonsai on other data-types
### Minimally required files
*Bonsai* minimally requires two data-files
* `means.txt`
* `stds.txt`

### TO BE CONTINUED

## *Bonsai-scout*: Visualizing the *Bonsai* results
The reconstructed tree can be visualized in the Bonsai-scout-app that was developed for this. 

### Installing the required packages
First, make sure that you have the required packages installed. Since, we need a few packages more than for the *Bonsai*-reconstruction and we wanted to keep the *Bonsai*-dependencies as lean as possible, we created a new conda environment for *Bonsai-scout*. (If you created this environment before, you can easily switch with `conda activate bonsai_scout`).

* Make a new conda environment:
```
conda create --name bonsai_scout python=3.9 -y
conda activate bonsai_scout
```
* Install the required packages:
```
pip install -r requirements_bonsai_scout.txt
```

### Preprocessing for the visualization
Before running the app, we must first do a preprocessing run, using (from "Bonsai-data-representation")
```
python3 bonsai_scout/bonsai_scout_preprocess.py \
--results_folder <BONSAI_RESULTS_FOLDER> \
--annotation_path <PATH_TO_CELL_ANNOTATION_FILE> \
--take_all_genes False \
--config_filepath ''
```
You need to provide the results-folder where the *Bonsai*-results were stored. This directory normally also contains a copy of the configurations file that was used for *Bonsai*. If this is not the case, or not the correct YAML-file, then you can use the `--config_filepath` argument to override this. The argument `--take_all_genes` can be set to True if you want to be able to overlay the expression of all genes on the tree. By default, only the genes are available on which *Bonsai* reconstructed the tree (those with high enough signal-to-noise). Setting `--take_all_genes True` will increase memory usage and slow down the preprocessing.

In the folder under `--annotation_path`, one can provide several files in `.csv`- or `.tsv`-format that contain annotation for all the cells. These files should contain one column with the cell-IDs, and the next columns should all correspond to one annotation-type. These can be celltype-categories, but can also be numerical values. These annotations can be overlaid in the plot. An example-file is 
```
cellID,experimenter,measurement
55K12032T_S5,T,0.7
4230K12033A_S30,A,0.84
4028K12031A_S28,A,0.29
11K12011T_S1,T,0.493
```
One can also provide full feature matrices. These additional features can be visualized on the trees, just like gene expression data, or can be used to call marker features for a specific subset on the tree. **Make sure to start the filename of this annotation-file with `mat_`, for example `mat_UMI_counts`.** The file should be in the same format as a normal annotation file, but the preprocessing will assume that only numerical values are given.

The preprocessing step will create two files and store them in the *Bonsai* results-folder:
- `bonsai_vis_data.hdf` which contains all the necessary data. This file will not be changed while running the *Shiny*-app.
- `bonsai_vis_settings.json` which contains the default settings for the visualization. When you interact with the app, the data in this file will be updated. In the app, one can then store these settings, after which a `bonsai_vis_settings_<TIMESTAMP>.json` will be created. This file can then be used for a later run of the app, to retrieve the same settings. Since this `.json`-file is human-readable and -editable, one can also edit some settings manually, for example for picking a customized colormap.

### Running the shiny app
Now you can view your results in *Bonsai*'s Shiny app by running (from the "Bonsai-data-representation"-directory)
```
python3 bonsai_scout/run_shiny_bonsai.py \
--results_folder <BONSAI_RESULTS_FOLDER> \
--settings_filename bonsai_vis_settings.json
```
There should now be a print message pointing you to the correct link: 
```Your app will shortly be running at: http://0.0.0.0:8243. Use your browser (not Safari) to view it.```
and
```INFO:     Uvicorn running on http://0.0.0.0:8243 (Press CTRL+C to quit)```

**Warning: if you are running these commands through an SSH-connection, your link will not be correct. In that case you should do something like `http://<ADDRESS_OF_MY_SSH_NODE>:8243/`**, where 8243 matches the random port that is reported to you in the print-messages. **Warning 2:** The Shiny app, seems to sometimes have trouble running through the ssh-connection, resulting in a weird error-message that we should still debug. To be on the safe side, download the files `bonsai_vis_data.hdf`, `bonsai_vis_settings.json` to your personal computer and run the app from there.

## Example 1: Your first *Bonsai* run
To test if *Bonsai* is set-up correctly, let's run it on a very small dataset of 64 cells. The output from *Sanity* for this dataset is stored in `Bonsai-data-representation/examples/example_data/simulated_binary_6_gens_samplingNoise`. 
Let's create a YAML-file with the correct run configurations first. To keep things organized, we can store the YAML-file in the `1_simple_example`-folder:
```
python3 bonsai/create_config_file.py \
--new_yaml_path examples/new_yaml.yaml \
--dataset simulated_binary_6_gens_samplingNoise \
--verbose True \
--data_folder examples/example_data/simulated_binary_6_gens_samplingNoise/ \
--results_folder examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/ \
--filenames_data delta.txt,d_delta.txt \
--input_is_sanity_output True \
--zscore_cutoff 1.0 \
--UB_ellipsoid_size 1.0 --use_knn 10 \
--skip_greedy_merging False --skip_redo_starry False --skip_opt_times False --skip_nnn_reordering False \
--nnn_n_randommoves 100 --nnn_n_randomtrees 2 \
--pickup_intermediate False
```
You can open the YAML-file and see how the parameters are set. Also, please read the extensive comments to familiarize yourself with what the different options do.

We can now move to running *Bonsai*:
```
python3 bonsai/bonsai_main.py \
 --config_filepath examples/new_yaml.yaml \
 --step all
```
This can take a few minutes. At the moment there is still a lot of output printed, that gives an idea of the progress of *Bonsai. We will clean up this output in the future.

Results will be stored in the indicated results_folder: `examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/`. If the *Bonsai*-run fully completed, there will be a directory called `final_bonsai_zscore1.0`. You can view the results there.

We will now visualize the tree with some cell-annotation. For this, we already provide an annotation-file at `examples/example_data/simulated_binary_6_gens_samplingNoise/celltype_annotation.csv`. Since this simulated dataset is a binary tree, we annotated the cells by their clade at different heights of the tree. 

Now run the preprocessing of the visualization:
```
python3 bonsai_scout/bonsai_scout_preprocess.py \
  --results_folder examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/ \
  --annotation_path examples/example_data/simulated_binary_6_gens_samplingNoise/celltype_annotation.csv \
  --take_all_genes False \
  --config_filepath ''
```
Finally, run the following command to start the Shiny-app:
```
python3 bonsai_scout/run_bonsai_scout_app.py \
--results_folder examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/ \
--settings_filename bonsai_vis_settings.json
```
There should now be a print message pointing you to the correct link: 
```Your app will shortly be running at: http://0.0.0.0:8243. Use your browser (not Safari) to view it.```

## Example 2: Running *Bonsai* on an HPC-cluster
We will here give an example script on how to run *Bonsai* on an HPC cluster. The example is based on a Slurm-environment, but the basic principles can probably be generalized to other HPC-architectures.

We have divided *Bonsai* into three main steps: `preprocess`, `core_calc`, and `metadata`. The steps `preprocess` and `metadata` read in the data on all genes, while `core_calc` only deals with the genes that have a signal-to-noise ratio higher than the parameter `zscore_cutoff` which we recommend to set to 1. Therefore, the `core_calc`-step typically needs less allocated memory than the other steps. In contrast, `core_calc` is the most time-consuming. Both `preprocess` and `core_calc` benefit strongly from parallelization over multiple computing cores. 

An example *slurm*-script is given in `examples/2_bonsai_on_slurm/example_slurm_script.sh`, and we also added it below in this read-me. Note that there are some constants that one may need to adapt depending on the size of your dataset, i.e. `N_CORES`, `TIME`, `QOS`, `MEM_PER_CORE`. Also, make sure to adapt `PATH_TO_CODE` to have the absolute path to the `Bonsai-data-representation`-directory.

The `for`-loop that currently runs over only one value, can be used to start a *Bonsai* run on several datasets (with run configurations defined in their YAML-files) with only one script. The paths to the YAML-files have to be stored on different lines in a file called `yaml_paths.txt` which should be in the same folder as the slurm-script. The slurm-script will call a bash-script that executes the final *Python* call and that should also be in the same folder. You can find our example bash-script below.

One can now start the Bonsai-runs on all the datasets comprised in `yaml_paths.txt` by calling `sh example_run_script.sh`.

**example_slurm_script.sh**
```
#!/bin/bash

mkdir -p outerr

module purge
module load OpenMPI/4.1.5-GCC-12.3.0

export MPICC=(which mpicc)

PATH_TO_CODE="<PATH_TO_BONSAI_DEVELOPMENT>/Bonsai-data-representation"
SCRIPT_PATH="bonsai_bash.sh"

ID="bonsai_batch"
NCORES=5
TIME="06:00:00"
QOS=6hours
MEM_PER_CORE=5

mem=$((NCORES*MEM_PER_CORE))G

for i in {1..1}
do
  YAML_PATH=$(head -n ${i} ./yaml_paths.txt | tail -1)
  echo Starting Bonsai based on configuration file: ${YAML_PATH}

  preprocess_id=$(sbatch --export ALL  --parsable --mem=${mem} --cpus-per-task=1 --ntasks=${NCORES} --nodes=1 --time=00:30:00 \
  --output=./outerr/dataset${i}%x_%j.out \
  --error=./outerr/dataset${i}%x_%j.err --qos=30min \
  --job-name="${ID}"_bonsai_pre_${i} \
  ${SCRIPT_PATH} -n ${NCORES} -s preprocess -y ${YAML_PATH})

  #preprocess_id is a jobID; pass to next call as dependency
  #Start main tree reconstruction after preprocess_id=ok

 core_calc_id=$(sbatch --export ALL  --parsable --mem=${mem} --cpus-per-task=1 --ntasks=${NCORES} --nodes=1 --time=${TIME} \
  --dependency=afterok:$preprocess_id \
  --output=./outerr/dataset${i}%x_%j.out \
  --error=./outerr/dataset${i}%x_%j.err --qos=${QOS} \
  --job-name="${ID}"_bonsai_core_${i} \
  --mail-type=END,FAIL \
  --mail-user=your.mailaddress@univ.ch \
  ${SCRIPT_PATH} -n ${NCORES} -s core_calc -y ${YAML_PATH})

 sbatch --export ALL  --mem=${mem} --cpus-per-task=1 --ntasks=1 --time=00:30:00 \
  --dependency=afterok:$core_calc_id \
  --output=./outerr/dataset${i}%x_%j.out \
  --error=./outerr/dataset${i}%x_%j.err --qos=30min \
  --job-name="${ID}"_bonsai_metadata_${i} \
  --mail-type=END,FAIL \
  --mail-user=your.mailaddress@univ.ch \
  ${SCRIPT_PATH} -n ${NCORES} -s metadata -y ${YAML_PATH}
done
```

**bonsai_bash.sh**
```
#!/bin/bash

while getopts n:s:y: flag
do
    case "${flag}" in
        n) NCORES=${OPTARG};;
        s) STEP=${OPTARG};;
        y) YAML_PATH=${OPTARG};;
    esac
done

PATH_TO_CODE="/scicore/home/nimwegen/degroo0000/Bonsai-data-representation"

srun python -m mpi4py ${PATH_TO_CODE}/bonsai/bonsai_main.py --config_filepath ${YAML_PATH} --step ${STEP}
```

## Possible run configurations

### TO BE CONTINUED
For now, you can read the extensive comments in your YAML-config file or in the create_config_file.py-script. This is all you need, and more.
