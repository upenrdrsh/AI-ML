#imoprt streamlit
import streamlit as st

# import dataset
import sklearn
from sklearn.datasets import load_iris
iris = load_iris()

# Convert dataset to Dataframe
import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
X = iris.data
y = iris.target

# Model selection
from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# Train the model
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=3)
classifier_knn.fit(X_train, y_train)



# Get input from streamlit slider
sepal_length = st.slider("Select sepal length", 0.0, 10.0, 5.0)
sepal_width = st.slider("Select sepal width", 0.0, 10.0, 5.0)
petal_length = st.slider("Select petal length", 0.0, 10.0, 5.0)
petal_width = st.slider("Select petal width", 0.0, 10.0, 5.0)


# Create a function for prediction
import numpy as np
def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = classifier_knn.predict(sample)
    species = iris.target_names[prediction[0]]
    return species

#Prediction

if st.button("Predict"):
    pred_species = predict_iris_species(sepal_length, sepal_width, petal_length, petal_width)
    st.write(f"The predicted species is: {pred_species}")
