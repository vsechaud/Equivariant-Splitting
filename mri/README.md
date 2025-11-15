# Equivariant Splitting for Accelerated MRI

Equivariant imaging learns to solve the challenging, ill-posed problem of accelerated MRI directly from noisy k-space measurements.

**Setting up the environment**

For better reproducibility, we recommend using `conda` to set up the environment using our provided `environment.yml` file.

```sh
conda env create -f environment.yml
```

**Creating the dataset**

To prepare the dataset used in `train.py` for the experiments, follow the instructions below.

1. Download [FastMRI](https://fastmri.med.nyu.edu) single-coil knee acquisitions `knee_singlecoil_train` (72.7 GB)
2. Place the downloaded data in `FastMRI/knee/singlecoil_train`
3. Generate the slice dataset using `python create_dataset.py`

If everything is set up correctly, it should create a directory `FastMRI-Slices` containing a file named `fastmri_knee_singlecoil.pt`.

**Training a model**

```sh
python train.py <config>
```

**Configurations**

The parameter `<config>` corresponds to one of the configuration names below.

| Loss        | Equivariant  | Acceleration | Configuration name        |
|-------------|--------------|--------------|---------------------------|
| Supervised  | ✅           | x8           | MRIx8_EQ_Supervised       |
| Supervised  | ❌           | x8           | MRIx8_NEQ_Supervised      |
| ES (Ours)   | ✅           | x8           | MRIx8_EQ_ES               |
| ES          | ❌           | x8           | MRIx8_NEQ_ES              |
| EI          | ✅           | x8           | MRIx8_EQ_EI               |
| SURE        | ✅           | x8           | MRIx8_EQ_SURE             |
| Supervised  | ✅           | x6           | MRIx6_EQ_Supervised       |
| Supervised  | ❌           | x6           | MRIx6_NEQ_Supervised      |
| ES (Ours)   | ✅           | x6           | MRIx6_EQ_ES               |
| ES          | ❌           | x6           | MRIx6_NEQ_ES              |
| EI          | ✅           | x6           | MRIx6_EQ_EI               |
| SURE        | ✅           | x6           | MRIx6_EQ_SURE             |

### Acknowledgments

[![SSIBench](https://img.shields.io/badge/GitHub-SSIBench-blue.svg)](https://github.com/Andrewwango/ssibench)
[![DeepInverse](https://img.shields.io/github/stars/deepinv/deepinv?label=DeepInverse)](https://deepinv.github.io/deepinv)


This work makes use of the efficient training losses and MRI operator in DeepInverse and the convenient MRI training and comparison features in SSIBench.
