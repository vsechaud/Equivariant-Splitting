# Equivariant Splitting for Sparse-View Computed Tomography

Equivariant splitting learns to solve the challenging, ill-posed problem of sparse-view computed tomography directly from noisy sinograms.

**Setting up the environment**

For better reproducibility, we recommend using `conda` to set up the environment using our provided `environment.yml` file.

```sh
conda env create -f environment.yml
```

> [!NOTE]
> The environment installs [our fork of DeepInverse](https://github.com/jscanvic/deepinv/commit/a57330d8e9812e8144a158c4d0eed709358505cd), not an official release.

**Creating the dataset**

To prepare the dataset used in `train.py` for the experiments, follow the instructions below.

1. Download the [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) CT scans using the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
2. Place the downloaded data in `LIDC_IDRI` so that it contains the `LIDC-IDRI` directory and the `metadata.csv` file
3. Generate the sinogram dataset using `python create_dataset.py`

If everything is set up correctly, it should create a directory `LIDC_IDRI-Tomography` containing a file named `dinv_dataset0.h5`.

**Training a model**

```sh
python train.py <config>
```

**Configurations**

The parameter `<config>` corresponds to one of the configuration names below.

| Loss        | Equivariant  | Configuration name        |
|-------------|--------------|---------------------------|
| Supervised  | ✅            | CT_EQ_Supervised          |
| Supervised  | ❌            | CT_NEQ_Supervised         |
| ES (Ours)   | ✅            | CT_EQ_ES                  |
| ES          | ❌            | CT_NEQ_ES                 |
| EI          | ✅            | CT_EQ_EI                  |
| EI          | ❌            | CT_NEQ_EI                 |

### Acknowledgments

[![DeepInverse](https://img.shields.io/github/stars/deepinv/deepinv?label=DeepInverse)](https://deepinv.github.io/deepinv)

This work makes use of the efficient training losses, tomography operator and LIDC-IDRI dataset in DeepInverse.

### Citation

Please consider citing this work if you find it useful in your research:

```
@inproceedings{sechaud2026equivariant,
    title={Equivariant Splitting: Self-supervised learning from incomplete data},
    author={Sechaud, Victor and Scanvic, J{\'e}r{\'e}my and Barth{\'e}lemy, Quentin and Abry, Patrice and Tachella, Juli{\'a}n},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=upMIVpe467}
}
```
