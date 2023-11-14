import warnings
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import moabb
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

# Suppress warnings and set MOABB log level
warnings.filterwarnings("ignore")
moabb.set_log_level("info")

# Instantiate the dataset
dataset = BNCI2014_001()
dataset.subject_list = dataset.subject_list  # Use all available subjects

# Define the paradigm
paradigm = MotorImagery()

# Create Pipeline
pipeline = make_pipeline(CSP(n_components=8), LDA())

# Load data and labels
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list)

# Flatten the data as they are nested lists
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

# Extract group information (subject/session) for LOGO
groups = metadata['subject']

# Instantiate the LeaveOneGroupOut cross-validator
logo = LeaveOneGroupOut()

# Metrics initialization
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# LOGO Cross-validation
for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and store metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

# Print average metrics
print("Average Accuracy:", np.mean(accuracy_scores))
print("Average Precision:", np.mean(precision_scores))
print("Average Recall:", np.mean(recall_scores))
print("Average F1 Score:", np.mean(f1_scores))

# For confusion matrix, you might want to predict on the whole dataset
y_pred_all = cross_val_predict(pipeline, X, y, cv=logo.split(X, y, groups))
conf_mat = confusion_matrix(y, y_pred_all)
sns.heatmap(conf_mat, annot=True)
plt.show()