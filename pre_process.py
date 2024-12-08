# Import pandas
import pandas as pd


def pre_process_visuals(disease_data_df):
    '''
    Preprocesses the disease data DataFrame for visualization purposes.

    This function performs several preprocessing steps on the input DataFrame, including:
    - Printing the shape of the DataFrame.
    - Displaying the data types of each column.
    - Checking for and summarizing NaN values.
    - Providing descriptive statistics of the DataFrame.
    - Identifying and counting duplicate rows grouped by the 'Disease' column.
    - Removing duplicate rows to obtain unique entries.
    - Melting the DataFrame to transform symptom columns into a long format suitable for visualization.

    Parameters:
        disease_data_df (pd.DataFrame): The DataFrame containing disease-related data with symptom columns.

    Returns:
        tuple:
            unique_rows (pd.DataFrame): A DataFrame with duplicate rows removed, containing unique disease entries.
            symptoms_melted (pd.DataFrame): A melted DataFrame where symptom columns are transformed into a single 'Symptom' column.
    '''
    # Print the shape 
    print("Data Shape: ", disease_data_df.shape)

    # Print the data types for each column 
    print("Data Types: \n", disease_data_df.dtypes)

    # Check for NaN values 
    print("Check for NaN values: \n", disease_data_df.isna().sum())

    # Check som dataframe statistics 
    print("Data Description: \n", disease_data_df.describe())

    # Check for duplicate rows grouped by diseases
    duplicate_counts = disease_data_df[disease_data_df.duplicated()].groupby('Disease').size().reset_index(name='Count')
    print(duplicate_counts)

    # Get unique rows
    unique_rows = disease_data_df.drop_duplicates()
    unique_count = unique_rows.shape[0]
    print(f"\nNumber of unique rows: {unique_count}")

    # Flatten the DataFrame
    symptoms_melted = disease_data_df.melt(id_vars=['Disease'],
                          value_vars=[col for col in disease_data_df.columns if 'Symptom' in col],
                          var_name='Symptom_Type',
                          value_name='Symptom')

    # remove any NaN
    symptoms_melted = symptoms_melted.dropna()
    
    # Returns 2 different dataframes for visuals 
    return unique_rows, symptoms_melted


def pre_process_ML(disease_data_df, symptom_df):
    '''
    Preprocesses disease and symptom data for machine learning applications.

    This function transforms the input disease DataFrame into a binary format suitable for machine learning models. 
    It performs the following steps:
    - Extracts unique symptoms from the symptom DataFrame.
    - Creates a binary DataFrame where each column represents a unique symptom and each row corresponds to a disease entry.
    - Populates the binary DataFrame with 1s indicating the presence of a symptom for a disease and 0s otherwise.

    Parameters:
        disease_data_df (pd.DataFrame): The DataFrame containing disease-related data with symptom columns.
        symptom_df (pd.DataFrame): The DataFrame containing a list of symptoms.

    Returns:
        pd.DataFrame: A binary DataFrame where each column represents a unique symptom and each row indicates 
        the presence (1) or absence (0) of symptoms for each disease.
    '''
    unique_symptoms = symptom_df["Symptom"].unique()
    # Transforming the data to binary checks for symptoms
    
    binary_data = pd.DataFrame(0, index=disease_data_df.index, columns=unique_symptoms)
    binary_data.insert(0, "Disease", disease_data_df["Disease"])

    # Populate the binary DataFrame and print debug information
    for idx, row in disease_data_df.iterrows():
        # Gather symptoms for the current row, removing NaNs and empty strings within the row
        symptoms_present = [symptom.strip() for symptom in row[1:].dropna() if symptom.strip() != '']
        # Debuggers
        # print(f"Row {idx}, Disease: {row['Disease']}")
        # print("Symptoms to add:", symptoms_present)

        # Set binary indicator to 1 for each symptom in the current row
        for symptom in symptoms_present:
            if symptom in binary_data.columns:
                binary_data.loc[idx, symptom] = 1
            
            # Debugger 
            # print(f"  Added symptom '{symptom}' to binary_data[{idx}]")
    
    return binary_data

def remove_high_corr(X):
    '''
    Removes highly correlated features from the dataset.

    This function identifies and removes features in the input DataFrame that have a correlation coefficient 
    higher than a specified threshold (default is 0.5) with any other feature. 
    It helps in reducing multicollinearity by retaining only one feature from each pair of highly correlated features.

    Parameters:
        X (pd.DataFrame): The input DataFrame from which highly correlated features will be removed.

    Returns:
        pd.DataFrame: A reduced DataFrame with highly correlated features removed.
    '''
    # Create the correlation matrix
    correlation_matrix = X.corr()

    # Identify highly correlated features (correlation > 0.5)
    high_corr_features = set()
    threshold = 0.5

    # Find pairs of features with correlation above the threshold
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                high_corr_features.add(correlation_matrix.columns[i])

    # Remove the highly correlated features from the dataset
    X_reduced = X.drop(columns=high_corr_features)
    return X_reduced