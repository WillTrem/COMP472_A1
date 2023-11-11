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
from sklearn.metrics import accuracy_score

# STEP 1
# Load both data sets
penguin_data = pd.read_csv("penguins.csv")
abalone_data = pd.read_csv("abalone.csv")

# Converting island and sex features with dummy-coded data for Peguin dataset
penguin_data_dummy = pd.get_dummies(penguin_data, columns=['island', 'sex']) #cam add 'sex'

# Converting island and sex features to categories for Peguin dataset
penguin_data_categorical = penguin_data

penguin_data_categorical['sex'] = pd.Categorical(penguin_data['sex']).codes
penguin_data_categorical['island'] = pd.Categorical(penguin_data['island']).codes

# Converting type features to categories for Abalone dataset
abalone_data['Type'] = pd.Categorical(abalone_data['Type'])

# Calculate the percentage of instances in each output class
penguin_class_counts = penguin_data['species'].value_counts()
penguin_class_percentage = penguin_class_counts / len(penguin_data) * 100

abalone_class_counts = abalone_data['Type'].value_counts()
abalone_class_percentage = abalone_class_counts / len(abalone_data) * 100

# STEP 2
# Plot and save the graphic for both datasets
plt.figure(figsize=(8, 6))
penguin_class_percentage.plot(kind='bar')
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
abalone_class_percentage.plot(kind='bar')
plt.title("Abalone Sex Distribution")
plt.xlabel("Sex")
plt.ylabel("Percentage (%)")
plt.savefig("abalone-classes.png")
plt.show()

image2 = Image.open("abalone-classes.png")
image2.save("abalone-classes.gif")
plt.show()

#STEP 3
# Assuming "species" is the target variable
X_penguin = penguin_data_categorical.drop(columns=['species'])  # Features
y_penguin = penguin_data_categorical['species']  # Target

# Assuming "type" is the target variable
X_abalone = abalone_data.drop(columns=['Type'])  # Features
y_abalone = abalone_data['Type']  # Target


X_train_penguin, X_test_penguin, y_train_penguin, y_test_penguin = train_test_split(X_penguin, y_penguin, random_state=42)
X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone = train_test_split(X_abalone, y_abalone, random_state=42)

# STEP 4 - ABALONE
# (a) Base-DT

# Create a Decision Tree classifier with default parameters
clf = DecisionTreeClassifier(max_depth=5)

# Fit the Decision Tree classifier to your training data
clf.fit(X_train_abalone, y_train_abalone)

# Visualize the Decision Tree
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=X_abalone.columns,  
                           class_names=['Male', 'Female', 'Infant'],  # Replace with actual class names
                           filled=True, rounded=True, special_characters=True)  

graph = graphviz.Source(dot_data)
graph.render("abalone_decision_tree")  # Saves the graph to "abalone_decision_tree.pdf"
# graph.view("abalone_decision_tree")    # Opens a viewer for the graph

# Tests the model with the testing data
y_pred_abalone = clf.predict(X_test_abalone)

print("Abalone Base-DT Accuracy: ", accuracy_score(y_test_abalone, y_pred_abalone))

# (b) Top-DT
# Define the hyperparameters grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, None],  
    'min_samples_split': [2, 5, 10]  
}

# Create a Decision Tree classifier
dt = DecisionTreeClassifier(max_depth=3)

# Perform GridSearch to find the best hyperparameters
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_abalone, y_train_abalone)

# Get the best estimator
best_dt = grid_search.best_estimator_

# Visualize the Decision Tree
dot_data = export_graphviz(best_dt, out_file=None, 
                           feature_names=X_abalone.columns,
                           class_names=best_dt.classes_,
                           filled=True, rounded=True, special_characters=True)
graph = Source(dot_data)
graph.render("abalone_best_decision_tree")

# Tests the model with the testing data
y_pred_abalone = best_dt.predict(X_test_abalone)

print("Abalone Top-DT Accuracy: ", accuracy_score(y_test_abalone, y_pred_abalone))

# (c) Base-MLP
# Create a MLP Classifier with two hidden layers of 100 neurons each, logistic activation function and stochastic gradient descent
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')

# Train the model
mlp_clf.fit(X_train_abalone, y_train_abalone)

# Test the model 
y_pred_abalone = mlp_clf.predict(X_test_abalone)

# Print the accuracy 
print("Abalone Base-MLP Accuracy: ", accuracy_score(y_test_abalone, y_pred_abalone))

# (d) Top-MLP
top_mlp_clf = MLPClassifier()

mlp_param_grid = {
	'activation':['logistic', 'tanh', 'relu'],
	'hidden_layer_sizes':[(25, 25), (15, 15, 15)],
	'solver':['adam', 'sgd']
}

mlp_grid_search = GridSearchCV(top_mlp_clf, mlp_param_grid)
mlp_grid_search.fit(X_train_abalone, y_train_abalone)

best_mlp_clf = mlp_grid_search.best_estimator_

y_pred_abalone = best_mlp_clf.predict(X_test_abalone)

print("Abalone Top-MLP Accuracy: ", accuracy_score(y_test_abalone, y_pred_abalone))


# STEP 4 - PENGUIN
# (a) Base-DT

# Create a Decision Tree classifier with default parameters
clf = DecisionTreeClassifier()

# Fit the Decision Tree classifier to your training data
clf.fit(X_train_penguin, y_train_penguin)

# Visualize the Decision Tree
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=X_penguin.columns,  
                           class_names=['Male', 'Female', 'Infant'],  # Replace with actual class names
                           filled=True, rounded=True, special_characters=True)  

graph = graphviz.Source(dot_data)
graph.render("penguin_decision_tree")  # Saves the graph to "penguin_decision_tree.pdf"
# graph.view("penguin_decision_tree")    # Opens a viewer for the graph

# Tests the model with the testing data
y_pred_penguin = clf.predict(X_test_penguin)

print("Penguin Base-DT Accuracy: ", accuracy_score(y_test_penguin, y_pred_penguin))

# (b) Top-DT

# Create a Decision Tree classifier
dt = DecisionTreeClassifier(max_depth=3)

# Perform GridSearch to find the best hyperparameters
# Uses the same param grid as abalone
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_penguin, y_train_penguin)

# Get the best estimator
best_dt = grid_search.best_estimator_

# Visualize the Decision Tree
dot_data = export_graphviz(best_dt, out_file=None, 
                           feature_names=X_penguin.columns,
                           class_names=best_dt.classes_,
                           filled=True, rounded=True, special_characters=True)
graph = Source(dot_data)
graph.render("penguin_best_decision_tree")

# Tests the model with the testing data
y_pred_penguin = best_dt.predict(X_test_penguin)

print("Penguin Top-DT Accuracy: ", accuracy_score(y_test_penguin, y_pred_penguin))

# (c) Base-MLP
# Create a MLP Classifier with two hidden layers of 100 neurons each, logistic activation function and stochastic gradient descent
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')

# Train the model
mlp_clf.fit(X_train_penguin, y_train_penguin)

# Test the model 
y_pred_penguin = mlp_clf.predict(X_test_penguin)

# Print the accuracy 
print("Penguin Base-MLP Accuracy: ", accuracy_score(y_test_penguin, y_pred_penguin))

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

# Print the accuracy
print("Penguin Top-MLP Accuracy: ", accuracy_score(y_test_penguin, y_pred_penguin))
