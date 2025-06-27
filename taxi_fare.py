import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras

import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
training_df = taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS','FARE', 'COMPANY', "PAYMENT_TYPE", 'TIP_RATE']]

print("Read dataset completed successfully")
print(f"Total number of row {format(len(training_df.index))}")

max_fare = training_df['FARE'].max()

print(training_df.corr(numeric_only=True))

sns.pairplot(training_df, x_vars = ["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars = ["FARE", "TRIP_MILES", "TRIP_SECONDS"])

#@title Define plotting functions

def make_plots(df, feature_names, label_name, model_output, sample_size=200):

  random_sample = df.sample(n=sample_size).copy()
  random_sample.reset_index()
  weights, bias, epochs, rmse = model_output

  is_2d_plot = len(feature_names) == 1
  model_plot_type = "scatter" if is_2d_plot else "surface"
  fig = make_subplots(rows=1, cols=2,
                      subplot_titles=("Loss Curve", "Model Plot"),
                      specs=[[{"type": "scatter"}, {"type": model_plot_type}]])

  plot_data(random_sample, feature_names, label_name, fig)
  plot_model(random_sample, feature_names, weights, bias, fig)
  plot_loss_curve(epochs, rmse, fig)

  fig.show()
  return

def plot_loss_curve(epochs, rmse, fig):
  curve = px.line(x=epochs, y=rmse)
  curve.update_traces(line_color='#ff0000', line_width=3)

  fig.append_trace(curve.data[0], row=1, col=1)
  fig.update_xaxes(title_text="Epoch", row=1, col=1)
  fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])

  return

def plot_data(df, features, label, fig):
  if len(features) == 1:
    scatter = px.scatter(df, x=features[0], y=label)
  else:
    scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)

  fig.append_trace(scatter.data[0], row=1, col=2)
  if len(features) == 1:
    fig.update_xaxes(title_text=features[0], row=1, col=2)
    fig.update_yaxes(title_text=label, row=1, col=2)
  else:
    fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))

  return

def plot_model(df, features, weights, bias, fig):
  df['FARE_PREDICTED'] = bias[0]

  for index, feature in enumerate(features):
    df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

  if len(features) == 1:
    model = px.line(df, x=features[0], y='FARE_PREDICTED')
    model.update_traces(line_color='#ff0000', line_width=3)
  else:
    z_name, y_name = "FARE_PREDICTED", features[1]
    z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
    y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
    x = []
    for i in range(len(y)):
      x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

    plane=pd.DataFrame({'x':x, 'y':y, 'z':[z] * 3})

    light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
    model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
                                      colorscale=light_yellow))

  fig.add_trace(model.data[0], row=1, col=2)

  return

def model_info(feature_names, label_name, model_output):
  weights = model_output[0]
  bias = model_output[1]

  nl = "\n"
  header = "-" * 80
  banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header

  info = ""
  equation = label_name + " = "

  for index, feature in enumerate(feature_names):
    info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
    equation = equation + "{:.3f} * {} + ".format(weights[index][0], feature)

  info = info + "Bias: {:.3f}\n".format(bias[0])
  equation = equation + "{:.3f}\n".format(bias[0])

  return banner + nl + info + nl + equation

print("SUCCESS: defining plotting functions complete.")


# create a simple linear regression model

def build_model(my_learning_rate, num_features):
  # describes the topology of our model
  inputs = keras.Input(shape = (num_features))
  outputs = keras.layers.Dense(units=1)(inputs)
  model = keras.Model(input=inputs, outputs=outputs)

  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[keras.metrics.RootMeanSquaredError()])
  return model

def train_model(model, features, label, epochs, batch_size):
  """Train the model by feeding it data"""

  # the feature and label, the model will train for a specific number of epochs
  history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs)

  # gather the trained models weight and bias
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # the list of epochs is stored separately from the rest of the history
  epochs = history.epoch

  # Isolate the error for each epoch
  hist = pd.DataFrame(history.history)

  # to track the progression of the trainging, we are going to take a snapshot 
  # of the models root mean squrare error at each epoch
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse

def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):
  print("Info: starting the training experiment with features={} and label={}\n".format(feature_names, label_name))

  num_features = len(feature_names)
  features = df.loc[:, feature_names].values
  label = df[label_name].values

  model = build_model(learning_rate, num_features)
  model_output = train_model(model, features, label, epochs, batch_size)
  print("Experiment Successful")
  print(model_info(feature_names, label_name, model_output))
  make_plots(df, feature_names, label_name, model_output)

  return model

print("Linear Regression Complete")

#title -- experiment 1

# hyperparameters
learning_rate = 0.001
epochs = 20
batch_size = 50

# specify our feature and label
features = ["TRIP_MILES"]
label = 'FARE'

model_1 = run_experiment(training_df, features, label, learning_rate, epochs, batch_size)

  


