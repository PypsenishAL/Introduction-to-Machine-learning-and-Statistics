import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv(r'https://raw.githubusercontent.com/Joshwen7947/Machine-Learning-Pipeline/refs/heads/main/ML_pipeline/vehicle_emissions.csv')
print(df.head())

X = df.drop(columns=['CO2_Emissions'])
y = df['CO2_Emissions']
numercial_columns = X.select_dtypes(exclude='object').columns
categorical_columns = X.select_dtypes(include='object').columns

# создали пайплан для нумерикал
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler())
    ])

# создали пайплан для категориальных
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                                ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([('numerical_pipeline', numerical_pipeline, numercial_columns), 
                                  ('categorical_pipeline', categorical_pipeline, categorical_columns)])

# объединили два препроцессор и пайплан в единую модель
pipeline = Pipeline([('preprocessor', preprocessor), 
                     ('model', RandomForestRegressor())])


X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(x_test)