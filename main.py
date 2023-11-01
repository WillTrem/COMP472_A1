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

# STEP 1
# Load both data sets
penguin_data = pd.read_csv("penguins.csv")
abalone_data = pd.read_csv("abalone.csv")

# Converting island and sex features with dummy-coded data for Peguin dataset
penguin_data = pd.get_dummies(penguin_data, columns=['island']) #cam add 'sex'

# Converting island and sex features to categories for Peguin dataset
penguin_data['sex'] = pd.Categorical(penguin_data['sex'])

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

# Assuming "species" is the target variable
X_penguin = penguin_data.drop(columns=['species'])  # Features
y_penguin = penguin_data['species']  # Target

# Assuming "type" is the target variable
X_abalone = abalone_data.drop(columns=['Type'])  # Features
y_abalone = abalone_data['Type']  # Target

#STEP 3
X_train, X_test, y_train, y_test = train_test_split(X_penguin, y_penguin, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_abalone, y_abalone, random_state=42)

# STEP 4 (a) Base-DT
# Create a Decision Tree classifier with default parameters
clf = DecisionTreeClassifier(max_depth=5)

# Fit the Decision Tree classifier to your data
clf.fit(X_abalone, y_abalone)

# Visualize the Decision Tree
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=X_abalone.columns,  
                           class_names=['Male', 'Female', 'Infant'],  # Replace with actual class names
                           filled=True, rounded=True, special_characters=True)  

graph = graphviz.Source(dot_data)
graph.render("abalone_decision_tree")  # Saves the graph to "abalone_decision_tree.pdf"
graph.view("abalone_decision_tree")    # Opens a viewer for the graph

# STEP 4 (b) Top-DT
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
grid_search.fit(X_train, y_train)

# Get the best estimator
best_dt = grid_search.best_estimator_

# Visualize the Decision Tree
dot_data = export_graphviz(best_dt, out_file=None, 
                           feature_names=X_train.columns,
                           class_names=best_dt.classes_,
                           filled=True, rounded=True, special_characters=True)
graph = Source(dot_data)
graph.render("best_decision_tree")