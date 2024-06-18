# ET-Flow: Equivariant Flow Matching for Molecular Conformer Generation


## Setup Environment
Run the following commands to setup the environment:
```bash
conda env create -n etflow -f env.yml
conda activate etflow
# to install the etflow package
python3 -m pip install -e .
```

## Preprocessing Data
Download the data by running the following commands:
```bash
wget https://dataverse.harvard.edu/api/access/datafile/4327252 -O <output_folder_path/rdkit_folder.tar>
tar -zxvf <output_folder_path/rdkit_folder.tar>
```

Preprocess the data by running the following command. Pass in the path to the data from the previous step and the folder to save the preprocessed files:
```bash
python scripts/preprocess.py -p < path_to_saved_file > -d <folder_path_to_save_outputs>
```

## Training 
We provide our configs for training on the GEOM-DRUGS and the GEOM-QM9 datasets. Run the following commands once datasets are preprocessed and the environment is set up:

```bash
python etflow/train.py -c configs/geom_drugs.yaml
```
