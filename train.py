# import numpy as np
# import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Drop redundant columns
titanic = titanic.drop(columns=['embark_town', 'alive', 'who', 'class', 'adult_male'])

# Fill missing deck with most frequent value
most_common_deck = titanic['deck'].mode()[0]
titanic['deck'] = titanic['deck'].fillna(most_common_deck)

# Split features and target
X = titanic.drop(columns='survived')
y = titanic['survived']

# Define column types
numeric_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combined preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline with Logistic Regression
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# Optional test accuracy
accuracy = clf.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")

# Save trained pipeline
joblib.dump(clf, "titanic_pipeline.joblib")
print("✅ Pipeline saved as titanic_pipeline.joblib")
