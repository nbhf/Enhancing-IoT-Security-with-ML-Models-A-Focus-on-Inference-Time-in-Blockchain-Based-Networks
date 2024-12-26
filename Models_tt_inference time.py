import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load datasets============================================================================================================
def load_datasets():
    print("================================= LOADING DATASETS =================================")
    column_name = pd.read_csv("Field Names.csv", header=None)
    new_columns = list(column_name[0].values) + ['class', 'difficulty']
    
    nsl_kdd_path = "KDDmerged.csv"
    kdd_cup_path = "kddcup.data.corrected.csv"
    
    print("NSL KDD:Loading...")
    nsl_kdd = pd.read_csv(nsl_kdd_path, names=new_columns)
    
    print("KDD Cup-99:Loading...")
    kdd_cup = pd.read_csv(kdd_cup_path, names=new_columns)
    
    return nsl_kdd, kdd_cup



#preprocessing============================================================================================================
from sklearn.preprocessing import LabelEncoder
def preprocess_data(dataset):
    print("================================= DATA PREPROCESSING =================================")
    
    # Remove duplicate rows
    dataset = dataset.drop_duplicates()
    
    # Initialize label encoder for categorical columns
    encoder = LabelEncoder()
    for col in ['protocol_type', 'service', 'flag']:
        if col in dataset.columns:
            dataset.loc[:, col] = encoder.fit_transform(dataset[col])  # Use .loc to avoid the warning
    
    # Separate features and target
    X = dataset.iloc[:, :-2]  # Excluding 'class' and 'difficulty'
    y = encoder.fit_transform(dataset['class'])  # Encode the 'class' column
    print(X)
    print (y)
    return X, y


# 3. Train Random Forest model============================================================================================================
def train_and_evaluate_rf(X_train, y_train, dataset_name):
    print("================================= TRAINING RANDOM FOREST =================================")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    start_time = time.time()  # Start timing
    rf_model.fit(X_train, y_train)
    end_time = time.time()  # End timing
    
    training_time = end_time - start_time
    print(f"Random Forest training completed on {len(X_train)} samples in {training_time:.2f} seconds for {dataset_name}.")
    return rf_model, training_time



# 4. Train SVM model============================================================================================================
def train_and_evaluate_svm(X_train, y_train, dataset_name):
    print("================================= TRAINING SVM =================================")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    
    start_time = time.time()  # Start timing
    svm_model.fit(X_train, y_train)
    end_time = time.time()  # End timing
    
    training_time = end_time - start_time
    print(f"SVM training completed on {len(X_train)} samples in {training_time:.2f} seconds for {dataset_name}.")
    return svm_model, training_time, scaler




#KNN ===========================================================================================================================================
from sklearn.neighbors import KNeighborsClassifier

def train_and_evaluate_knn(X_train, y_train, dataset_name):
    print("================================= TRAINING k-NN =================================")
    # Hyperparameter: Number of neighbors
    n_neighbors = 5  # Default value, can be tuned
    
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Initialize k-NN model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Measure training time
    start_time = time.time()
    knn_model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"k-NN training completed on {len(X_train)} samples in {training_time:.2f} seconds for {dataset_name}.")
    return knn_model, training_time, scaler


#DecisionTree ===========================================================================================================================================
from sklearn.tree import DecisionTreeClassifier
def train_and_evaluate_dt(X_train, y_train, dataset_name):
    print("================================= TRAINING DECISION TREE =================================")
    dt_model = DecisionTreeClassifier(random_state=42)
    
    start_time = time.time()  # Start timing
    dt_model.fit(X_train, y_train)
    end_time = time.time()  # End timing
    
    training_time = end_time - start_time
    print(f"Decision Tree training completed on {len(X_train)} samples in {training_time:.2f} seconds for {dataset_name}.")
    return dt_model, training_time

#LogisticRegression ===========================================================================================================================================
from sklearn.linear_model import LogisticRegression
def train_and_evaluate_lr(X_train, y_train, dataset_name):
    print("================================= TRAINING LOGISTIC REGRESSION =================================")
    
    # Normalisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Initialisation du modèle
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Entraînement du modèle
    start_time = time.time()
    lr_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Logistic Regression training completed on {len(X_train)} samples in {training_time:.2f} seconds for {dataset_name}.")
    return lr_model, training_time, scaler


#AdaBoost ===========================================================================================================================================
from sklearn.ensemble import AdaBoostClassifier
def train_and_evaluate_adaboost(X_train, y_train, dataset_name):
    print("================================= TRAINING ADABOOST =================================")
    
    # Initialisation du modèle AdaBoost
    adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42,algorithm='SAMME')
    
    # Entraînement du modèle
    start_time = time.time()
    adaboost_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"AdaBoost training completed on {len(X_train)} samples in {training_time:.2f} seconds for {dataset_name}.")
    return adaboost_model, training_time


#naive_bayes ===========================================================================================================================================
from sklearn.naive_bayes import GaussianNB
def train_and_evaluate_nb(X_train, y_train, dataset_name):
    print("================================= TRAINING NAIVE BAYES =================================")
    
    # Initialisation du modèle
    nb_model = GaussianNB()
    
    # Entraînement du modèle
    start_time = time.time()
    nb_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Naive Bayes training completed on {len(X_train)} samples in {training_time:.2f} seconds for {dataset_name}.")
    return nb_model, training_time




# 5. Test the model============================================================================================================
def test_model(model, X_test, y_test, dataset_name, model_name, scaler=None):
    print(f"================================= TESTING {model_name} ON {dataset_name} =================================")
    
    #!!!!!!! this modifies the accuracy value for SVM !!!!!!!!!!
    if scaler:
        X_test = scaler.transform(X_test)  # Scale test data for SVM
    
    start_time = time.time()  # Start timing
    y_pred = model.predict(X_test)
    end_time = time.time()  # End timing
    
    inference_time = (end_time - start_time)/len(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Inference time for {len(X_test)} samples: {inference_time:.6f} seconds per sample ")
    #print(f"Classification Report:\n{classification_report(y_test, y_pred,zero_division=1)}")
    
    return accuracy             



# 6. Plot comparison=============================================================================================================
def plot_comparison(models, accuracies, dataset_name):
    plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Performance Comparison on {dataset_name}')
    plt.show()



# Main program=========================================================================================================================
if __name__ == "__main__":
    # Load datasets
    nsl_kdd, kdd_cup = load_datasets()
    
    # Preprocess datasets
    X_nsl, y_nsl = preprocess_data(nsl_kdd)
    X_kdd, y_kdd = preprocess_data(kdd_cup)
    
    # Split datasets into training and test sets
    X_train_nsl, X_test_nsl, y_train_nsl, y_test_nsl = train_test_split(X_nsl, y_nsl, test_size=0.15, random_state=42)
    X_train_kdd, X_test_kdd, y_train_kdd, y_test_kdd = train_test_split(X_kdd, y_kdd, test_size=0.15, random_state=42)
    


    # Train and test models on NSL-KDD 
    rf_model_nsl, rf_train_time_nsl = train_and_evaluate_rf(X_train_nsl, y_train_nsl, "NSL-KDD")
    accuracy_nsl_rf = test_model(rf_model_nsl, X_test_nsl, y_test_nsl, "NSL-KDD", "Random Forest")

    # Train and test Decision Tree on NSL-KDD
    dt_model_nsl, dt_train_time_nsl = train_and_evaluate_dt(X_train_nsl, y_train_nsl, "NSL-KDD")
    accuracy_nsl_dt = test_model(dt_model_nsl, X_test_nsl, y_test_nsl, "NSL-KDD", "Decision Tree")

    #svm
    svm_model_nsl, svm_train_time_nsl, scaler_nsl = train_and_evaluate_svm(X_train_nsl, y_train_nsl, "NSL-KDD")
    accuracy_nsl_svm = test_model(svm_model_nsl, X_test_nsl, y_test_nsl, "NSL-KDD", "SVM", scaler=scaler_nsl)

    # Train and test k-NN on NSL-KDD
    knn_model_nsl, knn_train_time_nsl, knn_scaler_nsl = train_and_evaluate_knn(X_train_nsl, y_train_nsl, "NSL-KDD")
    accuracy_nsl_knn = test_model(knn_model_nsl, X_test_nsl, y_test_nsl, "NSL-KDD", "k-NN", scaler=knn_scaler_nsl)
    
    # Train and test Logistic Regression on NSL-KDD
    lr_model_nsl, lr_train_time_nsl, lr_scaler_nsl = train_and_evaluate_lr(X_train_nsl, y_train_nsl, "NSL-KDD")
    accuracy_nsl_lr = test_model(lr_model_nsl, X_test_nsl, y_test_nsl, "NSL-KDD", "Logistic Regression", scaler=lr_scaler_nsl)

    # Train and test AdaBoost on NSL-KDD
    adaboost_model_nsl, adaboost_train_time_nsl = train_and_evaluate_adaboost(X_train_nsl, y_train_nsl, "NSL-KDD")
    accuracy_nsl_adaboost = test_model(adaboost_model_nsl, X_test_nsl, y_test_nsl, "NSL-KDD", "AdaBoost")


    # Train and test Naïve Bayes on NSL-KDD
    nb_model_nsl, nb_train_time_nsl = train_and_evaluate_nb(X_train_nsl, y_train_nsl, "NSL-KDD")
    accuracy_nsl_nb = test_model(nb_model_nsl, X_test_nsl, y_test_nsl, "NSL-KDD", "Naive Bayes")



    # Train and test models on KDD Cup-99
    rf_model_kdd, rf_train_time_kdd = train_and_evaluate_rf(X_train_kdd, y_train_kdd, "KDD Cup-99")
    accuracy_kdd_rf = test_model(rf_model_kdd, X_test_kdd, y_test_kdd, "KDD Cup-99", "Random Forest")

    # Train and test Decision Tree on KDD Cup-99
    dt_model_kdd, dt_train_time_kdd = train_and_evaluate_dt(X_train_kdd, y_train_kdd, "KDD Cup-99")
    accuracy_kdd_dt = test_model(dt_model_kdd, X_test_kdd, y_test_kdd, "KDD Cup-99", "Decision Tree")
    
    #svm
    svm_model_kdd, svm_train_time_kdd, scaler_kdd = train_and_evaluate_svm(X_train_kdd, y_train_kdd, "KDD Cup-99")
    accuracy_kdd_svm = test_model(svm_model_kdd, X_test_kdd, y_test_kdd, "KDD Cup-99", "SVM", scaler=scaler_kdd)

     # Train and test k-NN on KDD Cup-99
    knn_model_kdd, knn_train_time_kdd, knn_scaler_kdd = train_and_evaluate_knn(X_train_kdd, y_train_kdd, "KDD Cup-99")
    accuracy_kdd_knn = test_model(knn_model_kdd, X_test_kdd, y_test_kdd, "KDD Cup-99", "k-NN", scaler=knn_scaler_kdd)

    
    # Train and test Logistic Regression on KDD Cup-99
    lr_model_kdd, lr_train_time_kdd, lr_scaler_kdd = train_and_evaluate_lr(X_train_kdd, y_train_kdd, "KDD Cup-99")
    accuracy_kdd_lr = test_model(lr_model_kdd, X_test_kdd, y_test_kdd, "KDD Cup-99", "Logistic Regression", scaler=lr_scaler_kdd)

    # Train and test AdaBoost on KDD Cup-99
    adaboost_model_kdd, adaboost_train_time_kdd = train_and_evaluate_adaboost(X_train_kdd, y_train_kdd, "KDD Cup-99")
    accuracy_kdd_adaboost = test_model(adaboost_model_kdd, X_test_kdd, y_test_kdd, "KDD Cup-99", "AdaBoost")
   
    # Train and test Naïve Bayes on KDD Cup-99
    nb_model_kdd, nb_train_time_kdd = train_and_evaluate_nb(X_train_kdd, y_train_kdd, "KDD Cup-99")
    accuracy_kdd_nb = test_model(nb_model_kdd, X_test_kdd, y_test_kdd, "KDD Cup-99", "Naive Bayes")

   

  


    
    models = ['Random Forest', 'SVM', 'k-NN', 'Decision Tree', 'Logistic Regression', 'Naïve Bayes', 'AdaBoost']
    # Comparaison des modèles pour NSL-KDD
    accuracies_nsl = [accuracy_nsl_rf * 100, accuracy_nsl_svm * 100, accuracy_nsl_knn * 100, 
                    accuracy_nsl_dt * 100, accuracy_nsl_lr * 100, accuracy_nsl_nb * 100, accuracy_nsl_adaboost * 100]
    plot_comparison(models, accuracies_nsl, "NSL-KDD")


    # Comparaison des modèles pour KDD Cup-99
    accuracies_kdd = [accuracy_kdd_rf * 100, accuracy_kdd_svm * 100, accuracy_kdd_knn * 100, 
                    accuracy_kdd_dt * 100, accuracy_kdd_lr * 100, accuracy_kdd_nb * 100, accuracy_kdd_adaboost * 100]
    plot_comparison(models, accuracies_kdd, "KDD Cup-99")



   