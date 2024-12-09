# Import ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from visuals import create_learning_curves, create_conf_matirx

def grid_search(param_grid, X, y, estimator):
    '''
    Performs hyperparameter tuning using GridSearchCV.

    This function splits the dataset into training and testing sets, then uses GridSearchCV to find the best hyperparameters for the provided estimator based on the specified parameter grid. It fits the grid search to the training data and returns a formatted string of the best parameters found.

    Parameters:
        param_grid (dict or list of dicts): The parameter grid to search over.
        X (pd.DataFrame): The feature dataset.
        y (pd.Series): The target variable.
        estimator (sklearn estimator): The machine learning model or pipeline for which to perform hyperparameter tuning.

    Returns:
        str: A formatted string displaying the best parameters found by GridSearchCV.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=estimator, 
                               param_grid=param_grid, 
                               cv=5, 
                               n_jobs=-1, 
                               verbose=2)

    # Fit grid search to the data
    grid_search.fit(X_train, y_train)

    # Output the best hyperparameters
    output = f"Best parameters: {grid_search.best_params_}"
    return output

def decision_tree(X, y, tree_model):
    '''
    Train and evaluate a Decision Tree classifier.

    Splits the data into training and testing sets, fits the provided decision tree model
    on the training data, makes predictions on the test data, and prints the accuracy
    and classification report. Also generates learning curves and heatmap of the confusion matrix.

    Parameters:
        X (pandas.DataFrame): The feature dataset.
        y (pandas.Series): The target variable.
        tree_model (sklearn.tree.DecisionTreeClassifier): Decision Tree classifier instance.

    Returns:
        sklearn.tree.DecisionTreeClassifier: Trained Decision Tree model.
    '''
    # Split the data into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fitting on training data
    tree_model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = tree_model.predict(X_test)

    # Evaluation
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    create_learning_curves(tree_model, X, y, "Decision Tree")
    create_conf_matirx(y_test, y_pred, y.unique(), "Decision Tree")
    
    return tree_model

def random_forest(X, y, rf_model): 
    '''
    Train and evaluate a Random Forest classifier.

    Splits the data into training and testing sets, fits the provided random forest model
    on the training data, makes predictions on the test data, and prints the accuracy
    and classification report. Also generates learning curves and heatmap of the confusion matrix

    Parameters:
        X (pandas.DataFrame): The feature dataset.
        y (pandas.Series): the target variable
        rf_model (sklearn.ensemble.RandomForestClassifier): Random Forest classifier instance.
    '''

    # Split the data into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fitting the training data 
    rf_model.fit(X_train, y_train)
    
    # Predict on the test data 
    y_pred = rf_model.predict(X_test)
    
    # Evaluation
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    create_learning_curves(rf_model, X, y, "Random Forest")
    create_conf_matirx(y_test, y_pred, y.unique(), "Random Forest")
    
    return rf_model

def gradient_boosting(X, y, gb_model): 
    '''
    Train and evaluate a Gradient Boosting classifier.

    Splits the data into training and testing sets, fits the provided gradient boosting model
    on the training data, makes predictions on the test data, prints the accuracy and classification report,
    and generates learning curves and heatmap of the confusion matrix

    Parameters:
        X (pandas.DataFrame): The feature dataset
        y (pandas.Series): the target variable.
        gb_model (sklearn.ensemble.GradientBoostingClassifier): Gradient Boosting classifier instance.

    Returns:
        sklearn.ensemble.GradientBoostingClassifier: Trained Gradient Boosting model.
    '''
    # Split the data into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model on the reduced training dataset
    gb_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = gb_model.predict(X_test)

    # Evaluate the model's performance
    accuracy_gb = accuracy_score(y_test, y_pred)
    report_gb = classification_report(y_test, y_pred, zero_division=0)

    print(f"Gradient Boosting Accuracy: {accuracy_gb:.2f}")
    print("Classification Report:\n", report_gb)
    
    create_learning_curves(gb_model, X, y, "Gradient Boosting")
    create_conf_matirx(y_test, y_pred, y.unique(), "Gradient Boosting")
    
    return gb_model
