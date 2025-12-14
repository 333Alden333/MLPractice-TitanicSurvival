# Imports
# Data
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Create and read data frame
data = pd.read_csv("titanic.csv")
data.info() # finds data type
print(data.isnull().sum()) # finds missing values and sum of them

# Data cleaning
def preprocess_data(df):
    df.drop(columns = ["PassengerId", "Name", "Ticket", "Cabin"], inplace = True)

    df["Embarked"] = df["Embarked"].fillna("S")
    # had to change this line (25) from: df["Embarked"].fillna("S", inplace = True)
    # in order to avoid warnings about future changes in pandas 3.0
    df.drop(columns = ["Embarked"], inplace = True)

    fill_missing_ages(df)

    # Fill missing fare values with median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    # had to change this line (34) from: df["Fare"].fillna(df["Fare"].median(), inplace = True)
    # in order to avoid warnings about future changes in pandas 3.0

    
    # Convert gender
    df["Sex"] = df["Sex"].map({'male': 1, 'female': 0})

    # Feature engineering - create new columns
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels = False)
    df["AgeBin"] = pd.cut(df["Age"], bins = [0, 12, 20, 40, 60, np.inf], labels = False)

    return df

# Fill in missing ages
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()

    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"],
                         axis = 1)
    
data = preprocess_data(data)

# Create features / target variables - view these as flashcards (x front, y back)
x = data.drop(columns = ["Survived"])
y = data["Survived"]
# testing for 25% of data so about 105 passengers
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

# ML preprocessing
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Hyperparameter tuning - knn model
def tune_model(x_train, y_train):
    param_grid = {
        "n_neighbors": range(1,21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv = 5, n_jobs = -1)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_

best_model = tune_model(x_train, y_train)

# Predictions
def evaluate_model(model, x_test, y_test):
    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, x_test, y_test)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Confusion Matrix:')
print(matrix)

# Plot
def plot_model(matrix):
    plt.figure(figsize = (10, 7))
    sns.heatmap(matrix, annot = True, fmt = "d", xticklabels = ["Survived", "Not Survived"],
                yticklabels = ["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.show()

plot_model(matrix)