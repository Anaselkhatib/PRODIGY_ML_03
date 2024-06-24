Prodigy ML 03: Dogs vs Cats Image Classification Using SVM

This project implements a Support Vector Machine (SVM) model with Principal Component Analysis (PCA) to classify images of dogs and cats.
Table of Contents

    Overview
    Datasets
    Requirements
    Installation
    Usage
    Model Training and Evaluation
    Results
    Contributing

Overview

The goal of this project is to classify images of dogs and cats. We use an SVM model with PCA for dimensionality reduction to achieve this. The project includes data preprocessing, model training, evaluation, and saving the trained model.
Datasets

The datasets used in this project include images of dogs and cats. The images are stored in the train and test1 directories and should be structured as follows:

markdown

dogs-vs-cats/
├── train/
│   └── train/
│       ├── cat.0.jpg
│       ├── cat.1.jpg
│       └── ...
└── test1/
    └── test1/
        ├── cat.0.jpg
        ├── cat.1.jpg
        └── ...

Requirements

    Python 3.x
    numpy
    scikit-learn
    matplotlib
    tqdm
    joblib
    opencv-python
    seaborn

You can install the required packages using:

bash

pip install numpy scikit-learn matplotlib tqdm joblib opencv-python seaborn

Installation

    Clone the repository:

    bash

git clone https://github.com/your-username/dogs-vs-cats.git
cd dogs-vs-cats

Install the required packages:

bash

    pip install -r requirements.txt

Usage

    Ensure your dataset is in the correct format and place it in the project directory as shown above.

    Run the script:

    bash

    python script.py

Model Training and Evaluation

The model training and evaluation process includes the following steps:

    Loading and preprocessing the dataset: Handling missing values and normalizing the data.
    Splitting the data: Dividing the dataset into training and testing sets.
    Applying PCA and SVM: Using PCA for dimensionality reduction and training an SVM model.
    Hyperparameter tuning: Using GridSearchCV for finding the best hyperparameters.
    Evaluating the model: Assessing the model using accuracy, classification report, and confusion matrix.

Example code snippet for applying PCA and SVM:

python

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pca = PCA(n_components=0.8, random_state=42)
svm = SVC()

pipeline = Pipeline([
    ('pca', pca),
    ('svm', svm)
])

param_grid = {
    'pca__n_components': [2, 1, 0.9],
    'svm__kernel': ['linear', 'sigmoid'],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=4)
grid_search.fit(X_train, y_train)

Results

The performance of the classification model is evaluated based on:

    Accuracy: Overall accuracy on the test set.
    Classification Report: Detailed performance metrics for each class.
    Confusion Matrix: Visualization of the confusion matrix.

Example output:

yaml

Best Parameters: {'pca__n_components': 0.9, 'svm__kernel': 'linear'}
Best Score: 0.85
Accuracy: 0.83
Classification Report:
              precision    recall  f1-score   support

         Cat       0.84      0.82      0.83       100
         Dog       0.83      0.85      0.84       100

    accuracy                           0.83       200
   macro avg       0.83      0.83      0.83       200
weighted avg       0.83      0.83      0.83       200

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.