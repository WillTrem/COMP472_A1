import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import GridSearchCV
from graphviz import Source
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np


# STEP 5 - Model Performance


def model_performance_to_txt(
    fileName, modelName, y_test, y_pred, classes, hyperParameters=""
):
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    with open(fileName, "a") as file:
        file.write(
            "(A)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            + modelName
            + " - "
            + hyperParameters
            + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        )
        file.write("\n(B) Confusion Matrix")
        file.write("\n" + str(conf_matrix))
        file.write("\n(C) Precision, recall & F1 values for each class")
        for className in classes:
            file.write("\nClass " + className + ":")
            file.write("    \nPrecision: " + str(class_report[className]["precision"]))
            file.write("    \nRecall: " + str(class_report[className]["recall"]))
            file.write("    \nF1: " + str(class_report[className]["f1-score"]))
        file.write("\n(D) Accuracy, marcro-average & weighted-average F1")
        file.write("\nAccuracy: " + str(class_report["accuracy"]))
        file.write("\nMacro-average F1: " + str(class_report["macro avg"]["f1-score"]))
        file.write(
            "\nWeighted-average F1: " + str(class_report["weighted avg"]["f1-score"])
        )
        file.write("\n\n\n")


# STEP 1
# Load both data sets
penguin_data = pd.read_csv("penguins.csv")
abalone_data = pd.read_csv("abalone.csv")

penguin_classes = penguin_data["species"].unique()
abalone_classes = abalone_data["Type"].unique()

# Converting island and sex features with dummy-coded data for Peguin dataset
penguin_data_dummy = pd.get_dummies(
    penguin_data, columns=["island", "sex"]
)  # cam add 'sex'

# Converting island and sex features to categories for Peguin dataset
penguin_data_categorical = penguin_data

penguin_data_categorical["sex"] = pd.Categorical(penguin_data["sex"]).codes
penguin_data_categorical["island"] = pd.Categorical(penguin_data["island"]).codes

# Converting type features to categories for Abalone dataset
abalone_data_categorical = abalone_data
abalone_data_categorical["Type"] = pd.Categorical(abalone_data["Type"])

# Calculate the percentage of instances in each output class
penguin_class_counts = penguin_data["species"].value_counts()
penguin_class_percentage = penguin_class_counts / len(penguin_data) * 100

abalone_class_counts = abalone_data["Type"].value_counts()
abalone_class_percentage = abalone_class_counts / len(abalone_data) * 100

# STEP 2
# Plot and save the graphic for both datasets
plt.figure(figsize=(8, 6))
penguin_class_percentage.plot(kind="bar")
plt.title("Penguin Species Distribution")
plt.xlabel("Species")
plt.ylabel("Percentage (%)")
plt.savefig("penguin-classes.png")
plt.show()

# Convert the PNG file to a GIF
image = Image.open("penguin-classes.png")
image.save("penguin-classes.gif")
plt.show()

plt.figure(figsize=(8, 6))
abalone_class_percentage.plot(kind="bar")
plt.title("Abalone Sex Distribution")
plt.xlabel("Sex")
plt.ylabel("Percentage (%)")
plt.savefig("abalone-classes.png")
plt.show()

image2 = Image.open("abalone-classes.png")
image2.save("abalone-classes.gif")
plt.show()

# STEP 3
# Assuming "species" is the target variable
X_penguin = penguin_data_categorical.drop(columns=["species"])  # Features
y_penguin = penguin_data_categorical["species"]  # Target

# Assuming "type" is the target variable
X_abalone = abalone_data_categorical.drop(columns=["Type"])  # Features
y_abalone = abalone_data_categorical["Type"]  # Target


X_train_penguin, X_test_penguin, y_train_penguin, y_test_penguin = train_test_split(
    X_penguin, y_penguin, random_state=42
)
X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone = train_test_split(
    X_abalone, y_abalone, random_state=42
)

# STEP 4 - ABALONE
# (a) Base-DT

# Create a Decision Tree classifier with default parameters
clf = DecisionTreeClassifier(max_depth=5)

# Fit the Decision Tree classifier to your training data
clf.fit(X_train_abalone, y_train_abalone)

# Visualize the Decision Tree
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X_abalone.columns,
    class_names=["Male", "Female", "Infant"],  # Replace with actual class names
    filled=True,
    rounded=True,
    special_characters=True,
)

graph = graphviz.Source(dot_data)
graph.render("abalone_decision_tree")  # Saves the graph to "abalone_decision_tree.pdf"
# graph.view("abalone_decision_tree")    # Opens a viewer for the graph

# Tests the model with the testing data
y_pred_abalone = clf.predict(X_test_abalone)

model_performance_to_txt(
    "abalone-performance.txt",
    "Base-DT",
    y_test_abalone,
    y_pred_abalone,
    abalone_classes,
    "max_depth=5",
)

# (b) Top-DT
# Define the hyperparameters grid
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 5, 10],
}

# Create a Decision Tree classifier
dt = DecisionTreeClassifier(max_depth=3)

# Perform GridSearch to find the best hyperparameters
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train_abalone, y_train_abalone)

# Get the best estimator
best_dt = grid_search.best_estimator_

# Visualize the Decision Tree
dot_data = export_graphviz(
    best_dt,
    out_file=None,
    feature_names=X_abalone.columns,
    class_names=abalone_classes,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = Source(dot_data)
graph.render("abalone_best_decision_tree")

# Tests the model with the testing data
y_pred_abalone = best_dt.predict(X_test_abalone)

hyper_params = best_dt.get_params()
subset_params = {
    param: hyper_params[param]
    for param in ["criterion", "max_depth", "min_samples_split"]
}

model_performance_to_txt(
    "abalone-performance.txt",
    "Top-DT",
    y_test_abalone,
    y_pred_abalone,
    abalone_classes,
    ", ".join(map(str, subset_params.values())),
)

# (c) Base-MLP

# Create a MLP Classifier with two hidden layers of 100 neurons each, logistic activation function and stochastic gradient descent
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100, 100), activation="logistic", solver="sgd"
)

# Train the model
mlp_clf.fit(X_train_abalone, y_train_abalone)

# Test the model
y_pred_abalone = mlp_clf.predict(X_test_abalone)


model_performance_to_txt(
    "abalone-performance.txt",
    "Base-MLP",
    y_test_abalone,
    y_pred_abalone,
    abalone_classes,
    "(100, 100), logistic, sgd",
)
# (d) Top-MLP
top_mlp_clf = MLPClassifier()

mlp_param_grid = {
    "activation": ["logistic", "tanh", "relu"],
    "hidden_layer_sizes": [(25, 25), (15, 15, 15)],
    "solver": ["adam", "sgd"],
}

mlp_grid_search = GridSearchCV(top_mlp_clf, mlp_param_grid)
mlp_grid_search.fit(X_train_abalone, y_train_abalone)

best_mlp_clf = mlp_grid_search.best_estimator_

y_pred_abalone = best_mlp_clf.predict(X_test_abalone)

hyper_params = best_mlp_clf.get_params()
subset_params = {
    param: hyper_params[param]
    for param in ["hidden_layer_sizes", "activation", "solver"]
}


model_performance_to_txt(
    "abalone-performance.txt",
    "Top-MLP",
    y_test_abalone,
    y_pred_abalone,
    abalone_classes,
    ", ".join(map(str, subset_params.values())),
)

# STEP 4 - PENGUIN
# (a) Base-DT

# Create a Decision Tree classifier with default parameters
clf = DecisionTreeClassifier()

# Fit the Decision Tree classifier to your training data
clf.fit(X_train_penguin, y_train_penguin)

# Visualize the Decision Tree
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X_penguin.columns,
    class_names=["Male", "Female", "Infant"],  # Replace with actual class names
    filled=True,
    rounded=True,
    special_characters=True,
)

graph = graphviz.Source(dot_data)
graph.render("penguin_decision_tree")  # Saves the graph to "penguin_decision_tree.pdf"
# graph.view("penguin_decision_tree")    # Opens a viewer for the graph

# Tests the model with the testing data
y_pred_penguin = clf.predict(X_test_penguin)

model_performance_to_txt(
    "penguin-performance.txt",
    "Base-DT",
    y_test_penguin,
    y_pred_penguin,
    penguin_classes,
)

# (b) Top-DT

# Create a Decision Tree classifier
dt = DecisionTreeClassifier(max_depth=3)

# Perform GridSearch to find the best hyperparameters
# Uses the same param grid as abalone
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train_penguin, y_train_penguin)

# Get the best estimator
best_dt = grid_search.best_estimator_

# Visualize the Decision Tree
dot_data = export_graphviz(
    best_dt,
    out_file=None,
    feature_names=X_penguin.columns,
    class_names=best_dt.classes_,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = Source(dot_data)
graph.render("penguin_best_decision_tree")

# Tests the model with the testing data
y_pred_penguin = best_dt.predict(X_test_penguin)

hyper_params = best_dt.get_params()
subset_params = {
    param: hyper_params[param]
    for param in ["criterion", "max_depth", "min_samples_split"]
}

model_performance_to_txt(
    "penguin-performance.txt",
    "Top-DT",
    y_test_penguin,
    y_pred_penguin,
    penguin_classes,
    ", ".join(map(str, subset_params.values())),
)
# (c) Base-MLP
# Create a MLP Classifier with two hidden layers of 100 neurons each, logistic activation function and stochastic gradient descent
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100, 100), activation="logistic", solver="sgd"
)

# Train the model
mlp_clf.fit(X_train_penguin, y_train_penguin)

# Test the model
y_pred_penguin = mlp_clf.predict(X_test_penguin)

model_performance_to_txt(
    "penguin-performance.txt",
    "Base-MLP",
    y_test_penguin,
    y_pred_penguin,
    penguin_classes,
    "(100, 100), logistic, sgd",
)
# (d) Top-MLP
top_mlp_clf = MLPClassifier()

# Perform GridSearch to find the best hyperparameters.
# Uses the same param grid as abalone
mlp_grid_search = GridSearchCV(top_mlp_clf, mlp_param_grid)
mlp_grid_search.fit(X_train_penguin, y_train_penguin)

# Gets the best estimator
best_mlp_clf = mlp_grid_search.best_estimator_

# Test the model
y_pred_penguin = best_mlp_clf.predict(X_test_penguin)

hyper_params = best_mlp_clf.get_params()
subset_params = {
    param: hyper_params[param]
    for param in ["hidden_layer_sizes", "activation", "solver"]
}

model_performance_to_txt(
    "penguin-performance.txt",
    "Top-MLP",
    y_test_penguin,
    y_pred_penguin,
    penguin_classes,
    ", ".join(map(str, subset_params.values())),
)


# Part 6 - Repeating steps 4 and 5 five times for each model and appending metrics in performance file
def repeat_model_evaluation(
    X_train,
    X_test,
    y_train,
    y_test,
    model,
    model_name,
    classes,
    hyper_parameters,
    repetitions=5,
):
    accuracies = []
    macro_f1_scores = []
    weighted_f1_scores = []

    for i in range(repetitions):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # To calculate the performance metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = class_report["accuracy"]
        macro_f1 = class_report["macro avg"]["f1-score"]
        weighted_f1 = class_report["weighted avg"]["f1-score"]

        accuracies.append(accuracy)
        macro_f1_scores.append(macro_f1)
        weighted_f1_scores.append(weighted_f1)

        # Here we want to append to the performance file these metrics
        model_performance_to_txt(
            "performance.txt", model_name, y_test, y_pred, classes, hyper_parameters
        )

    # Calculating avg, variance (sigma)
    avg_accuracy = np.mean(accuracies)
    var_accuracy = np.var(accuracies)
    avg_macro_f1 = np.mean(macro_f1_scores)
    var_macro_f1 = np.var(macro_f1_scores)
    avg_weighted_f1 = np.mean(weighted_f1_scores)
    var_weighted_f1 = np.var(weighted_f1_scores)

    # Adding these calculations to the performance file shown above
    with open("performance.txt", "a") as file:
        file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        file.write(
            f"\n(A)Average Accuracy for {model_name}: {avg_accuracy}\n Average Variance for {model_name}: {var_accuracy}"
        )
        file.write(
            f"\n(B)Average Macro F1 for {model_name}: {avg_macro_f1}\n Average Variance for {model_name}: {var_macro_f1}"
        )
        file.write(
            f"\n(C)Average Weighted F1 for {model_name}: {avg_weighted_f1}\n Average Variance for {model_name}: {var_weighted_f1}\n"
        )


# Repeat model evaluation for Base-DT, Top-DT, Base-MLP, Top-MLP on abalone dataset
for _ in range(5):
    # (a) Base-DT
    clf = DecisionTreeClassifier(max_depth=5)
    repeat_model_evaluation(
        X_train_abalone,
        X_test_abalone,
        y_train_abalone,
        y_test_abalone,
        clf,
        "Base-DT",
        abalone_classes,
        "max_depth=5",
    )

    # (b) Top-DT
    dt = DecisionTreeClassifier(max_depth=3)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train_abalone, y_train_abalone)
    best_dt = grid_search.best_estimator_
    repeat_model_evaluation(
        X_train_abalone,
        X_test_abalone,
        y_train_abalone,
        y_test_abalone,
        best_dt,
        "Top-DT",
        abalone_classes,
        ", ".join(map(str, subset_params.values())),
    )

    # (c) Base-MLP
    mlp_clf_base = MLPClassifier(
        hidden_layer_sizes=(100, 100), activation="logistic", solver="sgd"
    )
    repeat_model_evaluation(
        X_train_abalone,
        X_test_abalone,
        y_train_abalone,
        y_test_abalone,
        mlp_clf_base,
        "Base-MLP",
        abalone_classes,
        "(100, 100), logistic, sgd",
    )

    # (d) Top-MLP
    top_mlp_clf = MLPClassifier()
    mlp_grid_search = GridSearchCV(top_mlp_clf, mlp_param_grid)
    mlp_grid_search.fit(X_train_abalone, y_train_abalone)
    best_mlp_clf = mlp_grid_search.best_estimator_
    repeat_model_evaluation(
        X_train_abalone,
        X_test_abalone,
        y_train_abalone,
        y_test_abalone,
        best_mlp_clf,
        "Top-MLP",
        abalone_classes,
        ", ".join(map(str, subset_params.values())),
    )

# Repeat model evaluation for Base-DT, Top-DT, Base-MLP, Top-MLP on penguin dataset
for _ in range(5):
    # (a) Base-DT
    clf_penguin_base = DecisionTreeClassifier()
    repeat_model_evaluation(
        X_train_penguin,
        X_test_penguin,
        y_train_penguin,
        y_test_penguin,
        clf_penguin_base,
        "Base-DT",
        penguin_classes,
    )

    # (b) Top-DT
    dt_penguin = DecisionTreeClassifier(max_depth=3)
    grid_search_penguin = GridSearchCV(dt_penguin, param_grid, cv=5, scoring="accuracy")
    grid_search_penguin.fit(X_train_penguin, y_train_penguin)
    best_dt_penguin = grid_search_penguin.best_estimator_
    repeat_model_evaluation(
        X_train_penguin,
        X_test_penguin,
        y_train_penguin,
        y_test_penguin,
        best_dt_penguin,
        "Top-DT",
        penguin_classes,
        ", ".join(map(str, subset_params.values())),
    )

    # (c) Base-MLP
    mlp_clf_penguin_base = MLPClassifier(
        hidden_layer_sizes=(100, 100), activation="logistic", solver="sgd"
    )
    repeat_model_evaluation(
        X_train_penguin,
        X_test_penguin,
        y_train_penguin,
        y_test_penguin,
        mlp_clf_penguin_base,
        "Base-MLP",
        penguin_classes,
        "(100, 100), logistic, sgd",
    )

    # (d) Top-MLP
    top_mlp_clf_penguin = MLPClassifier()
    mlp_grid_search_penguin = GridSearchCV(top_mlp_clf_penguin, mlp_param_grid)
    mlp_grid_search_penguin.fit(X_train_penguin, y_train_penguin)
    best_mlp_clf_penguin = mlp_grid_search_penguin.best_estimator_
    repeat_model_evaluation(
        X_train_penguin,
        X_test_penguin,
        y_train_penguin,
        y_test_penguin,
        best_mlp_clf_penguin,
        "Top-MLP",
        penguin_classes,
        ", ".join(map(str, subset_params.values())),
    )
