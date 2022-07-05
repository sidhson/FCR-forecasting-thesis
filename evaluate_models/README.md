Model evaluation
===

Evaluation scripts for evaluating models on validation and test data. Comparison through statistics and graphs. 

- `model_scoring.py` : assisting functions to compute statistics per forecasted time step as well as distributions over time steps.  
- `model_evaluation.py` : create an evaluation-object for a model to compute predictions on validation or test data. Predictions may be stored and loaded at a later stage. Generation of metrics plots. 
- `compare_models.py` : compare a set of models and generate figures with relative performance. 
- `evaluate_per_ev.py` : evalute model perfromance with respect to each EV.
