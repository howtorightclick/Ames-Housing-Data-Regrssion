"""
Regression on housing data using random forest
"""

import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def read_csv_from_file():
    """
    Read CSV into dataframe and clean data
    """
    df_ames_data = pd.read_csv("AmesHousing.csv", sep=",")
    df_train_data = pd.read_csv("train.csv", sep=",")

    # Drop redundant columns and rename columns to match train.csv
    for (col_name, _) in df_ames_data.items():
        new_name = re.sub(r'[^a-zA-Z0-9]', '', col_name)
        #new_name: str = col_name.replace(" ", "")
        df_ames_data.rename(columns={col_name: new_name}, inplace=True)

    df_ames_data.drop(columns=['PID'], inplace=True)
    df_ames_data.rename(columns={"Order": "Id"}, inplace=True)

    # Join training data sets
    combined_training_data = pd.concat([df_ames_data, df_train_data])

    # Drop rows with NaN
    #combined_training_data = combined_training_data.select_dtypes(include=[np.number])
    #print(combined_training_data.info())
    # Encode non numeric data

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', categories='auto')
    categorical_columns = combined_training_data.select_dtypes(include='object')
    
    feature_arr = one_hot_encoder.fit_transform(categorical_columns).toarray()
    feature_labels = one_hot_encoder.get_feature_names_out()
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_arr, columns=feature_labels)
    
    #print(features.info())
    # Add one-hot encoded columns to numerical features
    #new_X_train = pd.concat([numeric_X_train, OH_cols_train], axis=1)
    #new_X_valid = pd.concat([numeric_X_valid, OH_cols_valid], axis=1)
    #print(new_X_train)

    #df_encoded = combined_training_data.apply(one_hot_encoder.fit_transform)

    #combined_training_data = combined_training_data.dropna()

    #print(combined_training_data.info())

    return None

def fit_linear_regression(df: pd.DataFrame):
    """
    Fits multilinear regression model
    """
    df_test = pd.read_csv("test.csv", sep=",")
    df_test = df_test.select_dtypes(include=[np.number])

    x_train = df.drop(columns=["SalePrice"])
    y_train = df["SalePrice"]
    x_train, x_test = x_train.align(df_test, join='inner', axis=1)

    print(f"X_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {x_test.shape}")

    x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train_split, y_train_split)
    y_val_pred = model.predict(x_val)

    # Calculate performance metrics
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)

    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R² Score:", r2)

def plot_graph(df):
    """
    Plot sale price distribution and correlation heat map
    """
    sns.histplot(df["SalePrice"], kde=True, bins=30)
    plt.title('Distribution of SalePrice')
    plt.xlabel('SalePrice')
    plt.show()

    corr = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', cbar=True)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def main():
    """
    Execute
    """
    print("Executing main")
    training_df = read_csv_from_file()
    #fit_linear_regression(training_df)


    #predicted_price = regr.predict([[3, 2, 3]])
    #print(predicted_price)

if __name__ == "__main__":
    main()
