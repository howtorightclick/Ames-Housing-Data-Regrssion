"""
Regression on housing data using random forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def read_train_data_csv():
    """
    Read train data into dataframe
    """
    #df_ames_data = pd.read_csv("AmesHousing.csv", sep=",")
    df_train_data = pd.read_csv("train.csv", sep=",")

    # Drop redundant columns and rename columns to match train.csv
    #for (col_name, _) in df_ames_data.items():
    #    new_name = re.sub(r'[^a-zA-Z0-9]', '', col_name)
    #    #new_name: str = col_name.replace(" ", "")
    #    df_ames_data.rename(columns={col_name: new_name}, inplace=True)

    #df_ames_data.drop(columns=['PID'], inplace=True)
    #df_ames_data.rename(columns={"Order": "Id"}, inplace=True)

    # Join training data sets
    #combined_training_data = pd.concat([df_ames_data, df_train_data])

    return df_train_data

def read_test_data_csv():
    """
    Reads the test data into dataframe
    """
    df_test = pd.read_csv("test.csv", sep=",")
    df_test['SalePrice'] = 0
    return df_test

def fit_model(training_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fits multilinear regression model
    """

    x_train = training_df.drop(columns=["remainder__SalePrice"])
    y_train = training_df["remainder__SalePrice"]
    x_train, x_test = x_train.align(test_df, join='inner', axis=1)

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

    return model

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
    training_df = read_train_data_csv()
    test_df = read_test_data_csv()

    column_transformer = ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore"), 
        list(training_df.select_dtypes(include="object").columns))
    ], remainder="passthrough")
    training_df = column_transformer.fit_transform(training_df)
    training_df = pd.DataFrame(training_df.toarray(), columns=column_transformer.get_feature_names_out())

    test_df = column_transformer.transform(test_df)
    test_df = pd.DataFrame(test_df.toarray(), columns=column_transformer.get_feature_names_out())

    model = fit_model(training_df, test_df)
    test_df.drop('remainder__SalePrice', axis=1, inplace=True)
    preds = model.predict(test_df)

    sample_submission_df = pd.read_csv('./sample_submission.csv')
    sample_submission_df['SalePrice'] = preds
    sample_submission_df.to_csv('./submission.csv', index=False)
    sample_submission_df.head()

if __name__ == "__main__":
    main()
