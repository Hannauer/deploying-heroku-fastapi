# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Used a random forest forest classifiers as the model. n_estimators = 50 and random_state = 42 where used as hyperparameters. All the other parameters were the default ones.

## Intended Use
The model uses census data to predict the category of a person salary

## Training Data
The database base for the model can be found at https://archive.ics.uci.edu/ml/datasets/census+income. 80% of the data is used for 
model training.

## Evaluation Data
The database base for the model can be found at https://archive.ics.uci.edu/ml/datasets/census+income. 20% of the data was used for model evaluation.

## Metrics
Metrics used: accuracy score, f1 beta score, precision and recall. (Include performance here)
## Ethical Considerations
The metrics were also calculated in data slices so it's possible to identicate groups were the model has some ethical bias.

## Caveats and Recommendations
The data seens to biased on gender, futher investigation in needed.