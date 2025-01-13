import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument('source', help="Name of the source file")
args = parser.parse_args()

data_url = args.source
combined_df = pd.read_csv(data_url, index_col=[0])

# Convert string to number
combined_df['trans_date_trans_time'] = pd.to_datetime(combined_df['trans_date_trans_time'])
combined_df['dob'] = pd.to_datetime(combined_df['dob'])

"""Process hour attributes"""

# Extract the hour from the transaction dateTime, and introduce a new column
combined_df['trans_time'] = combined_df['trans_date_trans_time'].dt.hour

# Categorize hour into session
def categorize_hour(hour):
  if 0 <= hour <= 4 or 22 <= hour <=23:
    return 'Late Night'
  elif 5 <= hour <= 11:
    return 'Morning'
  elif 12 <= hour <= 17:
    return 'Afternoon'
  elif 18 <= hour <= 21:
    return 'Night'

combined_df['trans_session'] = combined_df['trans_time'].apply(categorize_hour)

"""Process age attributes"""

# Age Group
combined_df['age'] = combined_df['trans_date_trans_time'].dt.year - combined_df['dob'].dt.year

# Categorize into age group
def categorize_age(age):
  if 0 <= age <= 12:
    return 'kids'
  elif 13 <= age <= 20:
    return 'teenagers'
  elif 21 <= age <= 50:
    return 'adults'
  elif 51 <= age <= 80:
    return 'veterans'
  else:
    return 'senior veterans'

combined_df['trans_ageGroup'] = combined_df['age'].apply(categorize_age)

"""Process location attributes"""

combined_df['latitudinal_distance'] = abs(round(combined_df['merch_lat'] - combined_df['lat'],3))
combined_df['longitudinal_distance'] = abs(round(combined_df['merch_long'] - combined_df['long'],3))

"""Drop unnecessary attributes"""

combined_df.drop(['cc_num', 'first', 'last', 'trans_num'], axis = 1, inplace = True)
combined_df.drop(['long', 'lat', 'zip' ], axis = 1, inplace = True)
combined_df.drop(['gender', 'street', 'city', 'state'], axis = 1, inplace = True)
combined_df.drop(['trans_date_trans_time', 'dob', 'unix_time', 'merch_lat', 'merch_long', 'trans_session', 'trans_ageGroup'], axis = 1, inplace = True)

combined_df.head()

"""One Hot Encoding and Label Encoding"""

non_numeric_columns = combined_df.select_dtypes(exclude=['number']).columns

for column in non_numeric_columns:
    if len(combined_df[column].unique()) <= 3:
        combined_df = pd.get_dummies(combined_df, columns=[column])
        #print("Using one hot encoding: ", column)
    else:
        combined_df[column] = pd.factorize(combined_df[column])[0]
        #print("Using label encoding: ", column)

"""## Export Cleaned Data"""

# Split the combined dataset into train and test sets
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

combined_df.to_csv(os.path.join('temp', 'cleanData.csv'))
train_df.to_csv(os.path.join('temp', 'cleanTrain.csv'))
test_df.to_csv(os.path.join('temp', 'cleanTest.csv'))