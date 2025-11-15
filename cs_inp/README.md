# Compressive sensing and inpainting
This folder contains scripts to run experiments for CS and inpainting settings.

## Usage

The `run_train.py` script calls the `train` function defined in `train.py`.  

`run_train.py` expects a path to a YAML configuration file as an argument. Example configuration files can be found in the `configs/` folder.

### Example

```bash
python run_train.py --config_path configs/config_cs_split_equi.yaml
