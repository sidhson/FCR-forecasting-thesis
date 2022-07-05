M.Sc. thesis: Data-driven forecasting of electric vehicle charging for frequency regulation
===

Code from master thesis at Lund University within Automatic Control in collaboration with Emulate Energy AB. The data has been removed to respect the privacy of the electric vehicle users. Code relating to proprietary software has also been omitted.

Report: [Data-driven forecasting of electric vehicle charging for frequency regulation](https://lup.lub.lu.se/student-papers/search/publication/9092798)

Authors: Fredrik Sidh (fr8415si-s) and Gustaf Sundell (gu0147su-s)

Abstract: Electric vehicle (EV) charging may be used in aggregation as virtual batteries to provide a frequency regulating service to the power grid. The service is sold on the Frequency Containment Reserve (FCR) markets, and is traded one and two days ahead. Forecasts of charging patterns are essential to reliably provide this ancillary service. The thesis aims to build a generalized model for forecasting EV charging behavior of 47 EVs in Sweden. The charging behavior is characterized by the state of charge and whether the EV is plugged in to the home charging station or not. Recurrent Neural Networks (RNNs) and XGBoost are applied to produce forecasts that fit the two FCR market settings. Performance of the models is evaluated and compared to a naive baseline in terms of RMSE, MAE, accuracy and F1-score. The naive baseline assumes the same charging behavior as the previous week. The results show both classes of models to consistently beat the naive baseline on both markets, and XGBoost proved to be the best forecaster.