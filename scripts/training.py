import pandas as pd
import argparse
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import time
import pickle
import os

def compute_score(y_test, y_predict, model_name):
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    specificity = tn / (tn + fp)
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {round(accuracy_score(y_test, y_predict)*100,2)}%")
    print(f"Recall: {round(recall_score(y_test, y_predict, average='macro')*100,2)}%")
    print(f"Precision: {round(precision_score(y_test, y_predict, average='macro')*100,2)}%")
    print(f"Specificity: {round(specificity*100,2)}%")

def show_training_progress(description):
    """Shows a progress bar for the training process"""
    pbar = tqdm(total=100, desc=description, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    for i in range(100):
        time.sleep(0.01)  # Simulate training progress
        pbar.update(1)
    pbar.close()

def train_decision_tree(x_train, y_train, x_test, y_test):
    # print("\nTraining Decision Tree Model...")
    params = {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 5}
    model = DecisionTreeClassifier(**params)
    
    # show_training_progress("Training Decision Tree")
    model.fit(x_train, y_train)
    
    # print("Making predictions...")
    y_predict = model.predict(x_test)
    compute_score(y_test, y_predict, "Decision Tree")
    return model

def train_knn(x_train, y_train, x_test, y_test):
    # print("\nTraining KNN Model...")
    params = {'n_neighbors': 1, 'weights': 'uniform'}
    model = KNeighborsClassifier(**params)
    
    # show_training_progress("Training KNN")
    model.fit(x_train, y_train)
    
    # print("Making predictions...")
    y_predict = model.predict(x_test)
    compute_score(y_test, y_predict, "KNN")
    return model

def train_random_forest(x_train, y_train, x_test, y_test):
    # print("\nTraining Random Forest Model...")
    params = {'max_depth': 12, 'n_estimators': 100, 'random_state': 42}
    model = RandomForestClassifier(**params)
    
    # show_training_progress("Training Random Forest")
    model.fit(x_train, y_train)
    
    # print("Making predictions...")
    y_predict = model.predict(x_test)
    compute_score(y_test, y_predict, "Random Forest")
    return model

def train_logistic_regression(x_train, y_train, x_test, y_test):
    # print("\nTraining Logistic Regression Model...")
    params = {'solver': 'sag', 'random_state': 42, 'max_iter': 100}
    model = LogisticRegression(**params)
    
    # show_training_progress("Training Logistic Regression")
    model.fit(x_train, y_train)
    
    # print("Making predictions...")
    y_predict = model.predict(x_test)
    compute_score(y_test, y_predict, "Logistic Regression")
    return model

def train_naive_bayes(x_train, y_train, x_test, y_test):
    # print("\nTraining Naive Bayes Model...")
    model = GaussianNB()
    
    # show_training_progress("Training Naive Bayes")
    model.fit(x_train, y_train)
    
    # print("Making predictions...")
    y_predict = model.predict(x_test)
    compute_score(y_test, y_predict, "Naive Bayes")
    return model

def main():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection')
    parser.add_argument('train_file', help='Path to training CSV file')
    parser.add_argument('test_file', help='Path to test CSV file')
    parser.add_argument('model_type', choices=['dt', 'knn', 'rf', 'lr', 'nb'], 
                        help='Model type: dt (Decision Tree), knn (K-Nearest Neighbors), '
                             'rf (Random Forest), lr (Logistic Regression), nb (Naive Bayes)')
    
    args = parser.parse_args()

    # Load data
    # print("\nLoading datasets...")
    try:
        # with tqdm(total=2, desc="Loading Data") as pbar:
        train_df = pd.read_csv(args.train_file)
        # pbar.update(1)
        test_df = pd.read_csv(args.test_file)
        # pbar.update(1)
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return
    except pd.errors.EmptyDataError:
        print("Error: One of the CSV files is empty")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # print(f"\nDataset shapes:")
    # print(f"Training set: {train_df.shape}")
    # print(f"Test set: {test_df.shape}")

    # Prepare data
    # print("\nPreparing data...")
    X_train = train_df.drop('is_fraud', axis=1)
    Y_train = train_df['is_fraud']
    X_test = test_df.drop('is_fraud', axis=1)
    Y_test = test_df['is_fraud']

    # Train model based on type
    model_functions = {
        'dt': train_decision_tree,
        'knn': train_knn,
        'rf': train_random_forest,
        'lr': train_logistic_regression,
        'nb': train_naive_bayes
    }

    try:
        model = model_functions[args.model_type](X_train, Y_train, X_test, Y_test)
        #print(f"\nModel training completed successfully!")
        
        # Save the model to a pickle file
        model_filename = os.path.join('temp', f"fraud_detection_{args.model_type}_model.pkl")
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        #print(f"Model saved as: {model_filename}")
    except Exception as e:
        print(f"Error during model training: {e}")

if __name__ == "__main__":
    main()