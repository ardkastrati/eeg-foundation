import pandas as pd
# Load the Excel file
file_path = '/itet-stor/kard/deepeye_storage/foundation/MI_LR/ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100295/Questionnaire_results_of_52_subjects.xlsx'


data = pd.read_excel(file_path)

# Extract the relevant rows for sex, handedness, and age
sex = data.iloc[5, 7:59].values
handedness = data.iloc[3, 7:59].values
age = data.iloc[4, 7:59].values

# Replace the sex and handedness values with appropriate labels
sex = ['Female' if s == 0.0 else 'Male' for s in sex]
handedness = ['Left' if h == 0.0 else 'Right' if h == 1.0 else 'Both' for h in handedness]

# Create a structured dataframe with formatted subject labels
subjects = [f's{i:02d}.mat' for i in range(1, 53)]
structured_data = pd.DataFrame({
    'Sex': sex,
    'Handedness': handedness,
    'Age': age
}, index=subjects)

# Display the dataframe
print(structured_data)

# If you are using Jupyter Notebook or similar, you can display the dataframe as follows:
# from IPython.display import display
# display(structured_data)

# Example of accessing data using the formatted index
print(structured_data.loc['s01.mat']["Sex"])
