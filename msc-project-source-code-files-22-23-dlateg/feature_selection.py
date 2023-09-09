import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


train_file = "C:/Users/denla/Project/UNSW_NB15_training-set.csv"
test_file = "C:/Users/denla/Project/UNSW_NB15_testing-set.csv"

# Load the dataset into a pandas dataframe
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Check for missing values
print(train_data.isnull().sum())
print(test_data.isnull().sum())

print(train_data.head())
print(test_data.head())


def categorical_columns(data):
    # Select the categorical columns
    categorical_columns = data.select_dtypes(include=["object"]).columns 
    
    # Create an instance of One-hot-encoder
    onehot_encoder = OneHotEncoder()
    
    # Passing encoded columns
    for col in categorical_columns:
       # Encode categorical columns
       encoded_col = onehot_encoder.fit_transform(data[[col]])

       # New df using encoded values and appropriate features
       encoded_df = pd.DataFrame(encoded_col.toarray(), columns=onehot_encoder.get_feature_names_out([col]))
       
       # Concatenate new encoded df with orginal data and drop original categorical columns
       data = pd.concat([data, encoded_df], axis=1)
       data.drop(columns=[col], inplace=True)
    
    print(categorical_columns)
    print(data)

    return data


train_data = categorical_columns(train_data)
test_data = categorical_columns(test_data)


def feature_selection(data):

    # Create a features and target variable
    X = data.drop("label", axis=1)  
    y = data["label"]  

    num_features_to_select = 6  
    """Selectkbest is a feature selection method which selects the top k features using the mutual_info_classif scoring function
    mutual_info_classif calculates the mutual info between feature and variable x and y """
    selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)
    selector.fit(X, y)


    # Get indices of selected features 
    selected_feature_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_feature_indices]

    # Drop id column if its selected as its an incremental value that can falsely impact accuracy of the model
    id_index = np.where(selected_features == 'id')[0][0]
    selected_features = selected_features.drop('id')
    selector.scores_ = np.delete(selector.scores_[selected_feature_indices], id_index)

    print(selected_features)

    # Plotting the selected features
    plt.figure(figsize=(10, 6))
    plt.bar(selected_features, selector.scores_)
    plt.xlabel('Features')
    # Information gain is the amount of info gained or lost when making a decision
    plt.ylabel('Information Gain')
    plt.title('Selected Features - Information Gain')
    plt.xticks(rotation=90)
    plt.show()
    
    return X[selected_features]

selected_features = feature_selection(train_data)
# Use the same columnsretrieved from train data to make it consistent
selected_features_test = test_data[selected_features.columns]


def scaled_features(selected_features):

    scaler = RobustScaler()

    # Fit the scaler on the selected features and transform them
    scaled_features = scaler.fit_transform(selected_features)

    print(scaled_features[:5])

    # Converting the scaled_features array back to a dataframe so we can create plots
    scaled_df = pd.DataFrame(scaled_features, columns=selected_features.columns)

    # Scatter plots to visualise the data
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(selected_features.columns):
        plt.subplot(2, 3, i + 1)
        plt.scatter(range(len(selected_features)), selected_features[feature], c='blue', label='Original', alpha=0.7)
        plt.scatter(range(len(scaled_df)), scaled_df[feature], c='orange', label='Scaled', alpha=0.7)
        plt.xlabel('Original ' + feature)
        plt.ylabel('Scaled ' + feature)
        plt.title('Scatter Plot: Original vs Scaled ' + feature)
        plt.legend()

    plt.tight_layout()
    plt.show()
    
    return scaled_features

scaled_train_data = scaled_features(selected_features)
scaled_test_data = scaled_features(selected_features_test)

# Creating X and y variables to be used in training and testing the model
X_train = scaled_train_data
y_train = train_data["label"] 

X_test = scaled_test_data
y_test = test_data["label"]


# Training and evaluating the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Using tqdm to monitor the progress of the prediction
svm_predictions = []
for X_test_sample in tqdm(X_test):
    prediction = svm_model.predict([X_test_sample])
    svm_predictions.append(prediction)

svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)

# Calculate the confusion matrix
confusion = confusion_matrix(y_test, svm_predictions)

TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]

# Calculate the False Positive Rate (FPR)
SVM_FPR = FP / (FP + TN)

print("SVM False Positive Rate (FPR):",  SVM_FPR)

# Training and evaluating the KNN model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Using tqdm to monitor the progress of the prediction
knn_predictions = []
for X_test_sample in tqdm(X_test):
    prediction = knn_model.predict([X_test_sample])
    knn_predictions.append(prediction)

knn_accuracy = accuracy_score(y_test, knn_predictions)
print("KNN Accuracy:", knn_accuracy)

# Calculate the confusion matrix
confusion = confusion_matrix(y_test, knn_predictions)

TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]

# Calculate the False Positive Rate (FPR)
kNN_FPR = FP / (FP + TN)

print("kNN False Positive Rate (FPR):", kNN_FPR)

# Accuracy comparison
models = ['SVM', 'kNN']
accuracies = [svm_accuracy, knn_accuracy]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of SVM and kNN')
plt.ylim([0, 1])
plt.show()


# False Positive Rate comparison
models = ['SVM', 'kNN']
fpr_values = [SVM_FPR, kNN_FPR]

plt.figure(figsize=(6, 4))
plt.bar(models, fpr_values)
plt.xlabel('Models')
plt.ylabel('False Positive Rate (FPR)')
plt.title('FPR Comparison of SVM and kNN - FPR')
plt.ylim([0, 0.2])
plt.show()