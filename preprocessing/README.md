Preprocessing, formatting and feature engineering of EV time series
===

The data pipeline is structured as

* `process_raw_data.py` to process the raw data time series.
* `gen_seq_data.py` or `gen_cross_data.py` is used to generate labeled observations for learning using a sliding window.
    * The `seq` script is used to generate sequential data for the RNN models.
    * The `cross` script is used to generate cross-sectional data for the XGBoost models.
* `feature_engineering.py` is used as support for encoding of attributes and normalization. 
* `alvis` bash scripts are only for cloud computing cluster execution.