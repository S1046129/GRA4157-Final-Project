# Import Libraries for pre processing and transformation
import pandas as pd

# Import ML libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Import methods from other .py files 
from pre_process import pre_process_visuals, pre_process_ML, remove_high_corr
from ml_models import grid_search, decision_tree, random_forest, gradient_boosting
from visuals import plot_uniq_disease_count, symptoms_melted_df_plots, top_15_features

def main ():
    print("--" * 50)
    # Load the Datasets
    df = pd.read_csv('data/dataset.csv')
    severity = pd.read_csv('data/Symptom-severity.csv')
    
    # Pre-Process the data 
    unique_rows, symptoms_melted = pre_process_visuals(df)
    binary_data = pre_process_ML(df, severity)
    
    # Exploratory Data Analysis and Visualizations
    
    # Plot the top 15 most common symptom frequencies 
    # Figure 1 in the report 
    symptoms_melted_df_plots(symptoms_melted, "common")
    
    # Plot all the Unique Rows in the data 
    # Figure 2 in the report
    plot_uniq_disease_count(unique_rows)
    
    # Plot the heatmap of the Top 15 common symptoms and Diseases to seee their relations
    # Figure 3 in the report
    symptoms_melted_df_plots(symptoms_melted, "heatmap")
    '''
    # Plot the number of diseases associated with Top 15 common symptoms 
    symptoms_melted_df_plots(symptoms_melted, "count")
    
    # Plot the number of unique symptoms associated with the disease 
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
    decision_tree(X, y, tree_model1) 
    
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
    decision_tree(X, y, tree_model2) # Figure 4 and 7 in the report 
    
    print("--" * 50)
    ## Random Forest
    rf1 = RandomForestClassifier(random_state=42, n_estimators=10)
    rf_model1 = random_forest(X, y , rf1) 
    
    ### Find the most important features based on RF model
    feature_importances = rf_model1.feature_importances_
    features = X.columns
    
    top_15_features(feature_importances, features) 
    
    ### Remove one of the features from a feature pair with high correlation
    X_reduced = remove_high_corr(X)
    
    print("--" * 50)
    ### Fine tune and use the X_reduced, with less features
    rf2 = RandomForestClassifier(
        n_estimators=80,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )
    
    rf_model2 = random_forest(X_reduced, y, rf2) # Figure 5 and a heatmap of conf matrix 
    ### See how the features changed 
    feature_importances = rf_model2.feature_importances_
    features = X_reduced.columns
    
    top_15_features(feature_importances, features)
    
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
    gradient_boosting(X_reduced, y, gb_model) # Figure 6 and a heatmap of conf matrix 
    
    
if __name__ == "__main__":
    main()
    
    



