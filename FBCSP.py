

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest


lowcut = 7
highcut = 32
fs = 250
interval = 4

filtered_data = {}
for start in np.arange(lowcut, highcut, step=interval):
    band = "{:02d}_{:02d}".format(start, start+interval)
    filtered_data[band] = butter_bandpass_filter(X, start, start+interval, fs)

# Apply CSP to each band to get spatial filters and features
CSP_data = {}
for band in filtered_data:
    CSP_data[band] = {}
    CSP_data[band]['W'] = spatial_filter(*decompose_S(*compute_S(compute_cov(filtered_data[band]), white_matrix(*decompose_cov(compute_cov(filtered_data[band]))))))

# Prepare the feature vectors for classification
features = np.hstack([feat_vector(compute_Z(CSP_data[band]['W'], filtered_data[band], m=2)) for band in filtered_data])

# Select the most informative features using Mutual Information
X_selected = SelectKBest(mutual_info_classif, k=4).fit_transform(features, y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train an SVM classifier
svm = SVC(gamma='scale')
svm.fit(X_train, y_train)

# Evaluate the classifier
train_accuracy = svm.score(X_train, y_train) * 100
test_accuracy = svm.score(X_test, y_test) * 100
print(f"Train Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# If you want to perform cross-validation on the training set
cv_scores = cross_val_score(svm, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores*100}")
