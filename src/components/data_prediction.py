import pandas as pd
import pickle
import torch
import torch.nn as nn
import numpy as np
from model import BiLSTM



data = {
    'OFFENSE_CODE': 1402,
    'DAY_OF_WEEK' : 'Sunday',
    'DISTRICT' : 'C11', 
    'UCR_PART' : 'Part One', 
    'STREET' : 'LINCOLN ST',
    'REPORTING_AREA_STR': '808.0',
    'LATITUDE' : 42.357791, 
    'LONGITUDE' : -71.060300, 
    'YEAR' : 2018, 
    'MONTH' : 7, 
    'DAY' : 4, 
    'HOUR' : 15 
}

Offence_code_group = ['Larceny',
 'Vandalism',
 'Towed',
 'Investigate Property',
 'Motor Vehicle Accident Response',
 'Auto Theft',
 'Verbal Disputes',
 'Robbery',
 'Fire Related Reports',
 'Other',
 'Property Lost',
 'Medical Assistance',
 'Assembly or Gathering Violations',
 'Larceny From Motor Vehicle',
 'Residential Burglary',
 'Simple Assault',
 'Restraining Order Violations',
 'Violations',
 'Harassment',
 'Ballistics',
 'Property Found',
 'Police Service Incidents',
 'Drug Violation',
 'Warrant Arrests',
 'Disorderly Conduct',
 'Property Related Damage',
 'Missing Person Reported',
 'Investigate Person',
 'Fraud',
 'Aggravated Assault',
 'License Plate Related Incidents',
 'Firearm Violations',
 'Other Burglary',
 'Arson',
 'Bomb Hoax',
 'Harbor Related Incidents',
 'Counterfeiting',
 'Liquor Violation',
 'Firearm Discovery',
 'Landlord/Tenant Disputes',
 'Missing Person Located',
 'Auto Theft Recovery',
 'Service',
 'Operating Under the Influence',
 'Confidence Games',
 'Search Warrants',
 'License Violation',
 'Commercial Burglary',
 'HOME INVASION',
 'Recovered Stolen Property',
 'Offenses Against Child / Family',
 'Prostitution',
 'Evading Fare',
 'Prisoner Related Incidents',
 'Homicide',
 'Embezzlement',
 'Explosives',
 'Criminal Harassment',
 'Phone Call Complaints',
 'Aircraft',
 'Biological Threat',
 'Manslaughter',
 'Gambling',
 'INVESTIGATE PERSON',
 'HUMAN TRAFFICKING',
 'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE',
 'Burglary - No Property Taken']

# Convert the data dictionary to a Pandas DataFrame
data_df = pd.DataFrame([data])


df = pd.read_csv('artifacts/transformed_data.csv', encoding='latin-1')


# Append the df DataFrame to the existing dataset using pd.concat()
new_df = pd.concat([df, data_df], ignore_index=True)

# Open the file containing the pickled data
with open('artifacts/proprocessor.pkl', 'rb') as file:
    # Load the pickled data
    preprocessor = pickle.load(file)

# Convert the 'REPORTING_AREA_STR' column to string data type
new_df['REPORTING_AREA_STR'] = new_df['REPORTING_AREA_STR'].astype(str)
# tranform the new data  
new_arr = preprocessor.fit_transform(new_df)

#extract the last row
last_row = new_arr[-1,:]
last_row_reshape = torch.Tensor(last_row).unsqueeze(0).unsqueeze(0)


# Open the file containing the pickled data
with open('artifacts/model.pkl', 'rb') as file:
    # Load the pickled data
    model = pickle.load(file)


# predict the output
with torch.no_grad():
    single_output = model(last_row_reshape)

# Print the output tensor
#print(single_output)

# get top 5 crimes committed
single_output_arr = single_output.numpy()
# Get the indices of the top 5 values from the array
top5_indices = np.argsort(single_output_arr)[0, -5:]

# Print the indices of the top 5 values
#print(top5_indices)

# top 5 crimes name
top_crimes = []
for num in top5_indices:
    top_crimes.append(Offence_code_group[num])

print(top_crimes)