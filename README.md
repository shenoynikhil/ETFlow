<div align="center">

# ET-Flow: Equivariant Flow Matching for Molecule Conformer Generation
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
[![Conference](http://img.shields.io/badge/NeurIPS-2024-4b44ce.svg)](https://neurips.cc/virtual/2024/poster/94522)
[![Checkpoints]( https://img.shields.io/badge/Checkpoints-6AA84F)](https://zenodo.org/records/14226681)

<img src="./img/etflow.png" width="600">
</div>

Implementation of [Equivariant Flow Matching for Molecule Conformer Generation](https://arxiv.org/abs/2410.22388) by M Hassan, N Shenoy, J Lee, H Stark, S Thaler and D Beaini. The paper was accepted at [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/94522).

ET-Flow is a state-of-the-art generative model for generating small molecule conformations using equivariant transformers and flow matching.

### Install ET-Flow
We are now available on PyPI. Easily install the package using the following command:

```bash
pip install etflow
```

*Note*: If there are issues with `pytorch_cluster`/`pytorch_geometric` and `pytorch`, it might be easier to install pytorch first and then the `etflow` package via pip.

### Generating Conformations for Custom Smiles
**Option 1**: Load the model config and checkpoint with automatic download and caching. See ([tutorial.ipynb](tutorial.ipynb)) or use the following snippet to load the model and generate conformations for custom smiles input.

```python
from etflow import BaseFlow
model = BaseFlow.from_default(model="drugs-o3")

# prediction 3 conformations for one molecule given by smiles
smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
output = model.predict([smiles], num_samples=3, as_mol=True)
mol = output[smiles] # rdkit mol object

# if we want just positions as numpy array
output = model.predict([smiles], num_samples=3)
output[smiles] # np.ndarray with shape (num_samples, num_atoms, 3)

# for prediction on more than 1 smiles
smiles_1 = ...
smiles_2 = ...
output = model.predict([smiles_1, smiles_2], num_samples=3, as_mol=True)
```

We currently support the following configurations and checkpoint:
- `drugs-o3`
- `qm9-o3`
- `drugs-so3`

**Option 2**: Load the model config, download checkpoints from the following zenodo [link](https://zenodo.org/records/14226681) and load it manually into the model config. We have a sample notebook ([generate_confs.ipynb](generate_confs.ipynb)) to generate conformations for custom smiles input. One needs to pass the config and corresponding checkpoint path in order as additional inputs.

Note: Scaffold Splits and Checkpoints are stored at the following zenodo [link](https://zenodo.org/records/16551316).

### Setup Dev Environment
Run the following commands to setup the environment:
```bash
conda env create -n etflow -f env.yml
conda activate etflow
# to install the etflow package
python3 -m pip install -e .
```


### Preprocessing Data

[!IMPORTANT] I have changed some parts of the data preprocessing scripts to make it more efficient. However, these changes mean that the configs might not lead to the same results as the one reported in the paper. I am working on reproducing the results with the new preprocessed data. Thanks for your patience.

To pre-process the data, perform the following steps,
1. Download the raw GEOM and unzip the raw data using the following commands,

```bash
DATA_DIR=</path_to_data>
wget https://dataverse.harvard.edu/api/access/datafile/4327252 -O $DATA_DIR/rdkit_folder.tar
tar -xvf $DATA_DIR/rdkit_folder.tar -C $DATA_DIR
```

For the splits and test mols, download the files from the [torsional diffusion](https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7?usp=drive_link) and extract them to the respective folders inside `$DATA_DIR`. Ideally it should look like the following (after extracting the zip files),

```bash
$DATA_DIR/
├── QM9/
└── DRUGS/
└── XL/
```

Make sure to set the environment variable `DATA_DIR` to the path of the data directory with `export DATA_DIR=</path_to_data>`.

2. Process the data for `ET-Flow` training. All preprocessed data will be created inside a `processed` folder inside this directory.

```bash
python scripts/prepare_data.py -p $DATA_DIR/rdkit_folder
```

This should create a `processed` folder inside `$DATA_DIR` with the preprocessed data.

### Training
We provide our configs for training on the GEOM-DRUGS and the GEOM-QM9 datasets in various configurations. Run the following commands once datasets are preprocessed and the environment is set up:

```bash
python scripts/train.py -c configs/drugs-base.yaml
```

The following two configs from the `configs/` directory can be used for replicating paper results:
- `drugs-base.yaml`: ET-Flow trained on GEOM-DRUGS dataset
- `qm9-base.yaml`: ET-Flow trained on GEOM-QM9 dataset

### Evaluation
Evaluation happens in 2 steps as follows,

1. Generating Conformations
To run the evaluation on either GEOM or QM9 given a config and a checkpoint, run the following command,
```bash
# here n: number of inference steps for flow matching
python scripts/eval.py --config=<config-path> --checkpoint=<checkpoint-path>
```

To run the evaluation on GEOM-XL (a test-set containing much larger molecules), run the following command,
```bash
python scripts/eval_xl.py --config=<config-path> --checkpoint=<checkpoint-path>
```

2. Evaluating Conformations with RMSD Metrics
The above sample generation script should created a `generated_files.pkl` at the following path, `logs/samples/<config-path>/<data-time>/flow_nsteps_{value-passed-above}/generated_files.pkl`. With the given path, we can get the various RMSD metrics using,

```bash
python scripts/eval_cov_mat.py --path=<path-to-generated-files.pkl> --num_workers=10
```

### Acknowledgements
Our codebase is built using the following open-source contributions,
- [torchmd-net](https://github.com/torchmd/torchmd-net)
- [e3-diffusion-for-molecules](https://github.com/ehoogeboom/e3_diffusion_for_molecules)
- [pytorch lightning](https://lightning.ai/pytorch-lightning)

### Contact
For further questions, feel free to raise an issue.

### Citation
```
@misc{hassan2024etflow,
      title={ET-Flow: Equivariant Flow-Matching for Molecular Conformer Generation},
      author={Majdi Hassan and Nikhil Shenoy and Jungyoon Lee and Hannes Stark and Stephan Thaler and Dominique Beaini},
      year={2024},
      eprint={2410.22388},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2410.22388},
}
```
