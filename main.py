# Import Libraries for pre processing and transformation
import pandas as pd
import numpy as np

# Import ML libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Import methods from other .py files 
from pre_process import pre_process_visuals, pre_process_ML, remove_high_corr
from ml_models import grid_search, decision_tree, random_forest, gradient_boosting
from visuals import plot_uniq_disease_count, symptoms_melted_df_plots, create_learning_curves, top_15_features

def main ():
    print("--" * 50)
    # Load the Datasets
    df = pd.read_csv('data/dataset.csv')
    severity = pd.read_csv('data/Symptom-severity.csv')
    
    print("--" * 50)
    # Pre-Process the data 
    unique_rows, symptoms_melted = pre_process_visuals(df)
    binary_data = pre_process_ML(df, severity)
    
    print("--" * 50)
    # Exploratory Data Analysis and Visulaizations
    '''
    # Plot all the Unique Rows in the data 
    # Figure 1
    plot_uniq_disease_count(unique_rows)
    
    # Plot the top 15 most common symptom frequencies 
    # Figure 2
    symptoms_melted_df_plots(symptoms_melted, "common")
    
    # Plot the number of diseases associated with Top 15 common symptoms 
    # Figure 3 
    symptoms_melted_df_plots(symptoms_melted, "count")
    
    # Plot the heatmap of the Top 15 common symptoms and Diseases to seee their relations
    # Figure 4
    symptoms_melted_df_plots(symptoms_melted, "heatmap")
    
    # Plot the number of unique symptoms associated with the disease 
    # Figure 5 
    symptoms_melted_df_plots(symptoms_melted, "symptom_disease")
    '''

    print("--" * 50)
    # Machine Learning models 
    ## Define X and y variables 
    X = binary_data.drop(columns=['Disease'])
    y = binary_data['Disease']
    
    print("--" * 50)
    ## Decision Tree
    ### w/o fine tuning
    tree_model1 = DecisionTreeClassifier(random_state=42)
    decision_tree(X, y, tree_model1) # Produces Figure 6 
    
    print("--" * 50)
    ### Use Grid Search
    '''
    param_grid_tree = {
        'max_depth': [7, 10, 15],
        'min_samples_split': [2, 10, 20],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search(X, y, tree_model1)
    '''
    
    print("--" * 50)
    ### fine tune with results from the Grid Search 
    tree_model2 = DecisionTreeClassifier(random_state=42, max_depth=15, criterion='entropy', min_samples_split=2)
    decision_tree(X, y, tree_model2) # Produces Figure 7 
    
    print("--" * 50)
    ## Random Forest
    rf1 = RandomForestClassifier(random_state=42, n_estimators=10)
    rf_model1 = random_forest(X, y , rf1) # Produces Figure 8 
    
    ### Find the most important features based on RF model
    feature_importances = rf_model1.feature_importances_
    features = X.columns
    
    top_15_features(feature_importances, features) # Produces Figure 9 
    
    ### Remove one of the features from a feature pair with high correlation
    X_reduced = remove_high_corr(X)
    
    print("--" * 50)
    ### Fine tune and use the X_reduced, with less features
    rf2 = RandomForestClassifier(
        n_estimators=60,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )
    
    rf_model2 = random_forest(X_reduced, y, rf2) # Produces Figure 10 
    ### See how the features changed 
    feature_importances = rf_model2.feature_importances_
    features = X_reduced.columns
    
    top_15_features(feature_importances, features) # Produces Figure 11
    
    print("--" * 50)
    ## Gradient Boosting 
    gb = GradientBoostingClassifier(random_state=42)
    
    ### Start with Grid search
    '''
    param_grid_gb = {
        'n_estimators': [50, 75, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }
    
    grid_search(param_grid_gb, X_reduced, y, gb)
    '''
    
    print("--" * 50)
    ### Fine tune the GB model with the results from Grid search
    gb_model = GradientBoostingClassifier(
        n_estimators=50,          
        learning_rate=0.05,         
        max_depth=7,               
        random_state=42, 
    )
    
    # Fit the gradient boosting model 
    gradient_boosting(X_reduced, y, gb_model) # Produces Figure 12 
    
    
if __name__ == "__main__":
    main()
    
    



