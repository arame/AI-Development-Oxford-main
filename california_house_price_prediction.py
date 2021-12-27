

# Step -1 - Import Packages
import sys
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn import metrics
from config import EModel, Hyper, Scalar
from helper import Helper
plt.rcParams["figure.figsize"] = (10, 10)



# Step - 2 - Define the main function
def main():
    # Get data

    ### To Do Assignment: try changing the data from Boston housing to California housing dataset 
    ### You can load the datasets as follows::
    ###    from sklearn.datasets import fetch_california_housing
    ###    housing = fetch_california_housing()
    ###  Refer this link for more detatils: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
    california = fetch_california_housing()
    california_X = pd.DataFrame(california.data, columns = california.feature_names)
    california_y = california.target
    features = california.feature_names
    
    ## Data Exploration
    Helper.printline(f'The features in dataset are: {features}')
    #print(f'Data description\n {boston_X.describe()}')
    
    #Plots
    plot_data(california_X, california_y, features)

    ## Remove Outliers
    california_X, california_y = remove_outliers(california_X,california_y, features)
    
    X_train, y_train, X_test, y_test = preprocess(california_X, california_y, features)

    if Hyper.model_type == EModel.isSVC:
        best_model = best_SVM(X_train, y_train, X_test, y_test)
    elif Hyper.model_type == EModel.isMLP:
        best_model = best_MLP(X_train, y_train, X_test, y_test)
    else:
        sys.exit("Model not selected")
        
    best_model = train(best_model, X_train, y_train)
    title = '\n\nBest Model Performance on Test Dataset:\n'
    evaluate (best_model, X_test, y_test, title)

def best_SVM(X_train, y_train, X_test, y_test):
    model = SVR() 

    model = train(model, X_train, y_train)
    title = '\n\nBaseline Model Performance on Test Dataset:\n'
    evaluate(model, X_test, y_test, title)

    best_params = optimize_models(X_train, y_train)
    Helper.printline(best_params)

    ## Build Best Model
    best_C= Helper.printline(f"Best parameters = {best_params['C']}")
    best_kernel = best_params['kernel']

    best_model = SVR(kernel = best_kernel, C= best_C)
    return best_model

def best_MLP(X_train, y_train, X_test, y_test):
    model = MLPRegressor(early_stopping=True, random_state=42)

    model = train(model, X_train, y_train)
    title = '\n\nBaseline Model Performance on Test Dataset:\n'
    evaluate(model, X_test, y_test, title)

    best_params = optimize_models(X_train, y_train)
    Helper.printline(best_params)

    ## Build Best Model
    # params = {"hidden_layer_sizes":(30,15), 'learning_rate_init':[0.01]}
    best_lr= best_params['learning_rate_init']
    best_hidden_layers = best_params['hidden_layer_sizes']

    best_model = MLPRegressor(hidden_layer_sizes=best_hidden_layers, learning_rate_init=best_lr,early_stopping=True, random_state=42)
    return best_model

    

 
# Step - 3 - Plot graphs to understand data
def plot_data(x_df, y_df,features):
    X = x_df.values
    plt.figure(figsize=(10,10))
    plt.title("Price Distribution")
    plt.hist(y_df, bins=30)
    plt.show()
    #cols = x_df.columns()
    fig, ax = plt.subplots(1, len(features), sharey=True, figsize=(20,5))
    plt.title("Relationship between different input features and price")
    ax = ax.flatten()
    for i, col in enumerate(features):
        x = X[:,i]
        y = y_df
        ax[i].scatter(x, y, marker='o')
        ax[i].set_title(col)
        ax[i].set_xlabel(col)
        ax[i].set_ylabel('MEDV')
    plt.show()

    if Hyper.is_correlation:
        fig.clear(True)
        plot_features_correlation(x_df, features)

def plot_features_correlation(x_df, features):
    ### To Do Add the code to find and display correlation among
    ### different features
    
    X = x_df.values
    number_of_subplots = 3
    fig, ax = plt.subplots(1, number_of_subplots, sharey=True, figsize=(20,5))
    plt.title("Corelations between different input features")
    ax = ax.flatten()
    ax[0] = set_subplot(features, X, ax, "HouseAge", "Population", 0)
    ax[1] = set_subplot(features, X, ax, "AveRooms", "Population", 1)
    ax[2] = set_subplot(features, X, ax, "HouseAge", "AveRooms", 2)
    plt.show()
    fig.clear(True)

def set_subplot(features, X, ax, x_feature, y_feature, index):
    index_x = features.index(x_feature)
    index_y = features.index(y_feature)
    x = X[:,index_x]
    y = X[:,index_y]
    ax[index].scatter(x, y, marker='o')
    ax[index].set_title(x_feature)
    ax[index].set_xlabel(x_feature)
    ax[index].set_ylabel(y_feature)
    return ax[index]

# Step - 4 - Preprocess data
# Step -4a : Remove outliers
def remove_outliers(x,y, features):
    #remove null
    x_df = x.copy(deep=True)
    x_df['MEDV'] = y
    x_df.dropna(inplace=True)
    return x_df[features], x_df['MEDV']
    
    
# Step -4b : Normalize data
def scale_numeric(df):
    x = df.values
    ### To Do Assignment instead of StandardScaler use MinMaxScaler, 
    ### Also observe if scaling influences the results
    scaler = get_scaler()   # Checks the config.py file to determine which scaler to get
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def get_scaler():
    if Hyper.scalar_type == Scalar.is_standard: 
        return preprocessing.StandardScaler()
    
    if Hyper.scalar_type == Scalar.is_minmax:            
        return preprocessing.MinMaxScaler()
    
    sys.exit("Scalar not selected")

    

# Step -4b : Preprocess data
def preprocess(x, y, features):
    x_df = x[features].copy(deep=True)
    x_df = scale_numeric(x_df)
    #print(len(x_df),len(y))
    # Split data into train, test
    X_train, X_test, y_train, y_test = train_test_split(x_df,y, test_size=0.3, random_state=1)
    return X_train, y_train, X_test, y_test
    
    
    
    
# Step - 5 - train model 
def train(model,X_train, y_train):
    model.fit(X_train, y_train)
    return model
    
    
# Step - 6 - Evaluate Model
def evaluate(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    if Hyper.print_results:
      print(title)
      Helper.printline(f'R^2: {metrics.r2_score(y_test, y_pred)}')
      Helper.printline(f'MAE: {metrics.mean_absolute_error(y_test, y_pred)}')
      Helper.printline(f'MSE: {metrics.mean_squared_error(y_test, y_pred)}')
      Helper.printline(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')

    if Hyper.plot:
      plt.scatter(y_test, y_pred)
      plt.xlabel("Prices")
      plt.ylabel("Predicted prices")
      plt.title("Prices vs Predicted prices")
      plt.show()
    return 
    
    
    
    
# Step - 7 - Improve Model
def optimize_models(X_train, y_train):
  ### To Do Assignment Change the model to MLP  and accordingly change Grid search params
  params,  model = getmodel()
  clf = GridSearchCV(model, params)
  clf.fit(X_train, y_train)
  return (clf.best_params_)

def getmodel():
    if Hyper.model_type == EModel.isSVC:
        params = {'kernel':['linear', 'rbf'], 'C':[1, 10]}
        model = SVR() 
    elif Hyper.model_type == EModel.isMLP:
        params = {"hidden_layer_sizes":[(20,20), (50, 50)], 'learning_rate_init':[0.001, 0.0001]}
        model = MLPRegressor(early_stopping=True, random_state=42) 
    else:
        sys.exit("Model not selected")
        
    return params, model
        

# call the main finction
if __name__ == '__main__':
    main()
    
    
    
