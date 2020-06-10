import pickle
import os
import numpy as np
import pandas as pd
import json
import subprocess

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump
from typing import Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns
import azureml.core
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig
from azureml.core.experiment import Experiment
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azure.storage.blob import BlockBlobService
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import Model

run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

print("Loading training data...")
# https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
#datastore = ws.get_default_datastore()
#datastore_paths = [(datastore, 'diabetes/diabetes.csv')]
#traindata = Dataset.Tabular.from_delimited_files(path=datastore_paths)
#diabetes = traindata.to_pandas_dataframe()
#print("Columns:", diabetes.columns) 
#print("Diabetes data set dimensions : {}".format(diabetes.shape))

datastore = ws.get_default_datastore()
print(datastore.datastore_type, datastore.account_name, datastore.container_name)
file_path = request.json['file_path']
print(file_path)
file_name = request.json['file_name']
ds.upload(src_dir=file_path, target_path= None, overwrite=True, show_progress=True)

stock_ds = Dataset.Tabular.from_delimited_files(path=datastore.path(file_name))
stock_ds = stock_ds.register(workspace = ws,
                             name = file_name,
                             description = 'Introact Owner Data')

compute_target = AmlCompute(ws, cluster_name)
print('Found existing AML compute context.')    
dataset_name = file_name

# Get a dataset by name
df = Dataset.get_by_name(workspace=ws, name=dataset_name)

X = df.drop_columns(columns=[target_var])
y = df.keep_columns(columns=[target_var], validate=True)
print(y)
#y = diabetes.pop('Y')
#X_train, X_test, y_train, y_test = train_test_split(diabetes, y, test_size=0.2, random_state=0)
#data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}
conda_run_config = RunConfiguration(framework="python")        
conda_run_config.environment.docker.enabled = True
conda_run_config.environment.docker.base_image = azureml.core.runconfig.DEFAULT_CPU_IMAGE        
cd = CondaDependencies.create(pip_packages=['azureml-sdk[automl]'], 
                              conda_packages=['numpy', 'py-xgboost<=0.80'])
conda_run_config.environment.python.conda_dependencies = cd       
print('run config is ready')

ExperimentName = request.json['ExperimentName']       
tasks = request.json['tasks']
iterations = request.json['iterations']
n_cross_validations = request.json['n_cross_validations']
iteration_timeout_minutes = request.json['iteration_timeout_minutes']
primary_metric = request.json['primary_metric']
max_concurrent_iterations = request.json['max_concurrent_iterations']

automl_settings = {
                "name": ExperimentName,
                "iteration_timeout_minutes": iteration_timeout_minutes,
                "iterations": iterations,
                "n_cross_validations": n_cross_validations,
                "primary_metric": primary_metric,
                "preprocess": True,
                "max_concurrent_iterations": max_concurrent_iterations
                #"verbosity": logging.INFO
            }
automl_config = AutoMLConfig(task=tasks,
                                         #debug_log='automl_errors.log',
                                         #path=os.getcwd(),
                                         compute_target=compute_target,
                                         run_configuration=conda_run_config,
                                         X=X,
                                         y=y,
                                         **automl_settings,
                                        )

experiment=Experiment(ws, ExperimentName)
remote_run = experiment.submit(automl_config, show_output=True)
best_run, fitted_model = remote_run.get_output()
best_run_toJson = best_run.get_metrics()

best_model_name = best_run.name
model = remote_run.register_model(model_name=best_model, description = 'AutoML Model')
print(model.name, model.id, model.version, sep = '\t')
model_path = os.path.join(cwd, best_model, best_model_name)

# Randomly pic alpha
#alphas = np.arange(0.0, 1.0, 0.05)
#alpha = alphas[np.random.choice(alphas.shape[0], 1, replace=False)][0]
#print("alpha:", alpha)
#run.log("alpha", alpha)
#reg = Ridge(alpha=alpha)
#reg.fit(data["train"]["X"], data["train"]["y"])
#run.log_list("coefficients", reg.coef_)

#print("Evaluate the model...")
#preds = reg.predict(data["test"]["X"])
#mse = mean_squared_error(preds, data["test"]["y"])
#print("Mean Squared Error:", mse)
#run.log("mse", mse)

# Save model as part of the run history
print("Exporting the model as pickle file...")
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)

model_filename = "model.pkl"
model_path = os.path.join(outputs_folder, model_filename)
dump(reg, model_path)

# upload the model file explicitly into artifacts
print("Uploading the model into run artifacts...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

run.complete()