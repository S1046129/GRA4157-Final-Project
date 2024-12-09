# Import ML Libraries
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix

# Import plotting Libraries 
import matplotlib.pyplot as plt 
import seaborn as sns

# Import Data Transformation Libraries 
import numpy as np
import pandas as pd

def plot_uniq_disease_count(unique_rows_df):
    '''
    Plots the count of unique diseases using a bar chart.

    This function takes a DataFrame containing unique disease entries, calculates the count of each disease, and visualizes the counts using a seaborn bar plot.

    Parameters:
        unique_rows_df (pd.DataFrame): The DataFrame containing unique disease entries with a 'Disease' column.

    Returns:
        None. Displays a bar plot of disease counts.
    '''
    disease_counts_uniq = unique_rows_df.groupby("Disease").size().reset_index(name="Disease_Count")
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.barplot(data=disease_counts_uniq ,x="Disease_Count", y="Disease", ax=ax)
    plt.show()

def symptoms_melted_df_plots(symptoms_melted_df, plot_type): 
    '''
    Generates various plots based on the specified plot type using the melted symptoms DataFrame.

    This function creates different visualizations depending on the value of the `plot_type` parameter. The available plot types are:
    - "common": Displays a bar chart of the top 15 most common symptoms.
    - "count": Shows a bar chart of the number of diseases associated with the top 15 common symptoms.
    - "heatmap": Presents a heatmap of the top 15 common symptoms associated with diseases.
    - "symptom_disease": Illustrates a horizontal bar chart of the number of unique symptoms per disease.

    Parameters:
        symptoms_melted_df (pd.DataFrame): The melted DataFrame containing 'Disease' and 'Symptom' columns.
        plot_type (str): The type of plot to generate. Must be one of "common", "count", "heatmap", or "symptom_disease".

    Returns:
        None. Displays the specified plot.

    Raises:
        ValueError: If an invalid `plot_type` is provided.
    '''
    
    overall_symptom_count = symptoms_melted_df.groupby('Symptom').size().reset_index(name='Total_Count')

    # get the top 15 most common symptoms
    top_symptoms = overall_symptom_count.nlargest(15, 'Total_Count')
    
    if plot_type == "common":
        # plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_symptoms, x='Symptom', y='Total_Count')
        plt.xticks(rotation=45)
        plt.title('Top 15 Common Symptom Frequencies')
        plt.tight_layout()
        plt.show()
    elif plot_type == "count": 
        symptom_count = symptoms_melted_df[symptoms_melted_df['Symptom'].isin(top_symptoms['Symptom'])]
        # Count the number of unique diseases for each symptom
        disease_count_per_symptom = symptom_count.groupby('Symptom')['Disease'].nunique().reset_index(name='Disease_Count')

        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=disease_count_per_symptom, x='Symptom', y='Disease_Count')
        plt.xticks(rotation=45)
        plt.title('Number of Diseases Associated with Top 15 Common Symptoms')
        plt.xlabel('Symptom')
        plt.ylabel('Number of Diseases')
        plt.tight_layout()
        plt.show()
    elif plot_type == "heatmap": 
        symptom_count = symptoms_melted_df[symptoms_melted_df['Symptom'].isin(top_symptoms['Symptom'])]
        symptom_count = symptom_count.groupby(['Disease', 'Symptom']).size().unstack(fill_value=0)

        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(symptom_count, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Count'})
        plt.title('Top 15 Common Symptoms Associated with Diseases')
        plt.xlabel('Symptom')
        plt.ylabel('Disease')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    elif plot_type == "symptom_disease":
        # Count the number of symptoms per disease
        symptoms_per_disease = symptoms_melted_df.groupby('Disease')['Symptom'].nunique().reset_index(name='Symptom_Count')

        # Sort by number of symptoms
        symptoms_per_disease = symptoms_per_disease.sort_values('Symptom_Count', ascending=True)

        # Plot horizontal bar chart
        plt.figure(figsize=(10, 12))
        sns.barplot(data=symptoms_per_disease, x='Symptom_Count', y='Disease')
        plt.title('Number of Unique Symptoms per Disease')
        plt.xlabel('Number of Unique Symptoms')
        plt.ylabel('Disease')
        plt.tight_layout()
        plt.show()
    else: 
        print("Invalid Input for plot type!")
        

def create_learning_curves(model, X, y, title):
    '''
    Generates and plots learning curves for a given machine learning model.

    This function computes the training and cross-validation scores for different sizes of the training dataset using the specified model. It then plots the learning curves to help evaluate the model's performance and detect potential issues like overfitting or underfitting.

    Parameters:
        model (sklearn estimator): The machine learning model to evaluate.
        X (pd.DataFrame or np.ndarray): The feature dataset.
        y (pd.Series or np.ndarray): The target variable.
        title (str): The title for the learning curves plot.

    Returns:
        None. Displays a plot of the learning curves.
    '''
    # Generate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, random_state=42, verbose=0
    )

    # Calculate the mean and standard deviation of the training and test scores
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', color='blue')
    plt.plot(train_sizes, test_mean, label='Cross-Validation Score', color='orange')

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='orange', alpha=0.1)

    plt.title(f'Learning Curves - {title}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim(0,4000)
    plt.ylim(0.8,1.01)
    plt.show()

def top_15_features(feature_importances, features):
    '''
    Plots the top 15 most important features for disease prediction.

    This function creates a horizontal bar chart displaying the top 15 symptoms based on their importance scores. It sorts the features by their importance and visualizes the most influential symptoms in predicting diseases.

    Parameters:
        feature_importances (array-like): An array or list of feature importance scores.
        features (array-like): A list of feature names corresponding to the importance scores.

    Returns:
        None. Displays a horizontal bar chart of the top 15 most important symptoms.
    '''
    # Create a DataFrame to display the feature importances
    importance_df = pd.DataFrame({
        'Symptom': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Plot the top 15 most important symptoms
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Symptom'][:15], importance_df['Importance'][:15])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Top 15 Most Important Symptoms for Disease Prediction')
    plt.show()

def create_conf_matirx(y_test, y_pred, class_names, model_name):
    '''
    Plots a heatmap of the confusion matrix.

    Parameters:
    y_test (array-like): True class labels.
    y_pred (array-like): Predicted class labels.
    class_names (list): List of class names corresponding to the labels.

    Returns:
    None
    '''
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Classes")
    plt.ylabel("True Classes")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
