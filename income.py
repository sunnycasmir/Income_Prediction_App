#import libries
import numpy as np
import pandas as pd
from pycaret.classification import compare_models, create_model, setup
from pycaret.classification import create_api, tune_model, evaluate_model, create_app
from pycaret.datasets import get_data
from pycaret.classification import finalize_model, automl
from pycaret.classification import evaluate_model
from pycaret.classification import plot_model
from pycaret.classification import save_model

all_datasets = get_data('index')
df = get_data('income')

#checking the categorical variables
cat_cols = df.select_dtypes(include = 'object').columns.tolist()

#put all married in one class
def clean_marital_status(status):
    married_list = ['Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent']
    if status in married_list:
        return 'Married'
    else:
        return 'Not_married'
    
# Assuming df is your pandas DataFrame containing the 'marital-status' column
df['marital-status'] = df['marital-status'].apply(lambda x: clean_marital_status(x))

df.rename(columns = {'income >50K': 'income'}, inplace=True)

#Training process
s = setup(data = df, target = 'income', fix_imbalance=True, log_experiment=True,
       log_plots=True, remove_outliers=True,experiment_name='my_income_exp')

#Compare Models
top_3_models = compare_models(n_select=3, sort='F1')

#Tune Model
tuned_models = [tune_model(model, fold=5) for model in top_3_models]

#Finalize Model
best_model = automl(optimize='F1')

best_model

#evealuate
evaluate_model(best_model)

#plot model
plot_model(best_model, plot = 'learning')

#save model
save_model(best_model, 'best_model')



