import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


num_trials, num_channels, num_time_points = X_new.shape
data_reshaped = X_new.reshape(num_trials, num_channels * num_time_points)
X_train, X_test, y_train, y_test = train_test_split(data_reshaped, labels, test_size=0.3, random_state=42)


svc = SVC(kernel="linear")
selector = RFE(svc, n_features_to_select=6, step=1)
selector = selector.fit(X_train, y_train)


feature_ranking = selector.ranking_.reshape(num_channels, num_time_points)
top_channels = np.argwhere(feature_ranking == 1)
unique_top_channels = np.unique(top_channels[:, 0])


print("Indices of the top 6 channels:", unique_top_channels)


selected_data = X_new[:, unique_top_channels, :]
selected_data_reshaped = selected_data.reshape(num_trials, len(unique_top_channels) * num_time_points)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_data_reshaped, labels, test_size=0.3, random_state=42)

# Retrain SVM with only selected channels
svc.fit(X_train, y_train)
predictions = svc.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy with top 6 channels: {accuracy}")
'''
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train1, y_train1)

# Get feature importances and reshape back to (num_channels, num_time_points)
importances = rf.feature_importances_.reshape(num_channels, num_time_points)

# Sum importance scores for each channel
channel_importances = np.sum(importances, axis=1)

# Select top 6 channels based on importance
top_n_channels = 6
important_channels_indices = np.argsort(channel_importances)[-top_n_channels:]

# Print the indices of the top 6 channels
print("Indices of the top 6 channels:", important_channels_indices)

# Optional: You can retrain and evaluate the model using only the selected channels
# This step is to verify the performance with the selected channels
selected_data = X_new[:, important_channels_indices, :]
selected_data_reshaped = selected_data.reshape(num_trials, top_n_channels * num_time_points)
X_train, X_test, y_train, y_test = train_test_split(selected_data_reshaped, labels, test_size=0.3, random_state=42)

rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy with top {top_n_channels} channels: {accuracy}")


'''