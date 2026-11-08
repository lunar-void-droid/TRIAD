This repository contains the executable implementation of TRIAD, a LibCity-style model for multi-step traffic flow forecasting from road-sensor time series.

The code is organized as a compact traffic forecasting pipeline. The model is implemented in `libcity/model/traffic_flow_prediction/TRIAD.py`, while `run_model.py` provides the command-line entry point for training, checkpoint loading, and test evaluation. The repository currently includes experiment configurations for PEMSD4 and PEMSD8.

```
TRIAD/
|-- run_model.py
|-- triad_pemsd4_config.json
|-- triad_pemsd8_config.json
|-- requirements.txt
|-- libcity/
|   |-- config/
|   |-- data/
|   |-- evaluator/
|   |-- executor/
|   |-- model/traffic_flow_prediction/TRIAD.py
|   |-- pipeline/
|   `-- utils/
`-- README.md
```

This repository provides the TRIAD model, the LibCity-style data loading and training pipeline, dataset configuration templates, and the commands needed to reproduce the main training/evaluation workflow.

Raw PEMSD datasets, generated dataset caches, model checkpoints, and evaluation outputs are not included. The data files must be prepared separately in the LibCity format described below.

The implementation uses a compact subset of the LibCity interfaces for traffic state prediction. For the complete urban spatial-temporal prediction framework and its original dataset conventions, please refer to LibCity:

- Homepage: [https://libcity.ai/](https://libcity.ai/)
- GitHub: [https://github.com/LibCity/Bigscity-LibCity](https://github.com/LibCity/Bigscity-LibCity)
- Documentation: [https://bigscity-libcity-docs.readthedocs.io/](https://bigscity-libcity-docs.readthedocs.io/)

The dependency file targets Python 3.8/3.9, PyTorch 1.12.1, and CUDA 11.3. The exact PyTorch wheel should be selected according to the local operating system and CUDA installation.

```bash
conda create -n triad python=3.8 -y
conda activate triad
pip install -r requirements.txt
```

For a CPU-only run, set `"gpu": false` in the selected configuration file. The default experiment settings use CUDA when it is available.

Raw PEMSD datasets are not committed to this repository. Place each dataset under `raw_data/<DATASET>/` in LibCity format:

```
raw_data/
  PEMSD4/
    PEMSD4.geo
    PEMSD4.rel
    PEMSD4.dyna
    config.json
  PEMSD8/
    PEMSD8.geo
    PEMSD8.rel
    PEMSD8.dyna
    config.json
```

The `.geo` file describes the sensor nodes and must contain `geo_id`. The `.rel` file describes node relations and normally contains `origin_id`, `destination_id`, and a weight or distance column. The `.dyna` file contains dynamic observations with `time`, `entity_id`, and one or more value columns. An `.ext` file may be added for external features; the default configuration also derives time-of-day and day-of-week features from timestamps.

Each dataset directory must contain a `config.json`. A minimal example is:

```json
{
  "geo_file": "PEMSD8",
  "rel_file": "PEMSD8",
  "data_files": "PEMSD8",
  "ext_file": "PEMSD8",
  "data_col": "traffic_speed",
  "weight_col": "distance",
  "time_intervals": 300
}
```

Replace `data_col` and `weight_col` with the actual column names in the CSV files. If no external feature file is available, set `"load_external": false` in the experiment configuration.

Run the PEMSD8 experiment from the repository root with:

```bash
python run_model.py \
  --task traffic_state_pred \
  --model TRIAD \
  --dataset PEMSD8 \
  --config_file triad_pemsd8_config \
  --train true \
  --saved_model true
```

The same command in Windows PowerShell can be written on one line:

```powershell
python run_model.py --task traffic_state_pred --model TRIAD --dataset PEMSD8 --config_file triad_pemsd8_config --train true --saved_model true
```

The default configuration uses the previous 12 time steps to predict the next 12 steps. For 5-minute traffic measurements, this corresponds to one hour of history and one hour of forecasting. The data is split into training, validation, and test sets, standardized with training-set statistics, and evaluated with masked MAE, masked RMSE, and masked MAPE.

To evaluate a saved model, use the same `exp_id` that was assigned during training and disable training:

```bash
python run_model.py \
  --task traffic_state_pred \
  --model TRIAD \
  --dataset PEMSD8 \
  --config_file triad_pemsd8_config \
  --exp_id <TRAINING_EXP_ID> \
  --train false \
  --saved_model false
```

Generated checkpoints and evaluation files are written to:

```
libcity/cache/<exp_id>/model_cache/
libcity/cache/<exp_id>/evaluate_cache/
libcity/log/
```

The model can be configured through the JSON files or by overriding supported arguments from the command line. For example:

```bash
python run_model.py \
  --dataset PEMSD8 \
  --config_file triad_pemsd8_config \
  --gpu false \
  --batch_size 16 \
  --max_epoch 10 \
  --learning_rate 0.0005
```

If you use the LibCity framework together with TRIAD, please cite LibCity:
@inproceedings{libcity,
  author = {Wang, Jingyuan and Jiang, Jiawei and Jiang, Wenjun and Li, Chao and Zhao, Wayne Xin},
  title = {LibCity: An Open Library for Traffic Prediction},
  year = {2021},
  publisher = {Association for Computing Machinery},
  booktitle = {Proceedings of the 29th International Conference on Advances in Geographic Information Systems},
  pages = {145--148},
  doi = {10.1145/3474717.3483923}
}
