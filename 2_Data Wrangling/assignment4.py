import pandas as pd

# Loading up the table, and extracting the dataset
# out of it. 
#
df = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2', header = 1)[0]

# Renaming the columns so that they are similar to the
# column definitions provided on the website.
# 
df.columns = ['RK','PLAYER','TEAM','GP','G','A','PTS','+/-','PIM','PTS/G','SOG','PCT','GWG','PPG','PPA','SHG','SHA']

# Getting rid of any row that has at least 4 NANs in it,
# 
df = df.dropna(axis=0, thresh = 4)

# Looking through the dataset by printing
# it. 
#
print(df)
df=df.drop(df.index[[10,21,32]])

# Getting rid of the 'RK' column
#
df = df.drop(labels = ['RK'], axis = 1)

# Ensuring there are no holes in the index by resetting
# it.
df = df.reset_index(drop=True)

# Checking the data type of all columns, and ensuring those
# that should be numeric are numeric
df.GP = pd.to_numeric(df.GP, errors = 'coerce')
df.G = pd.to_numeric(df.G, errors = 'coerce')
df.A = pd.to_numeric(df.A, errors = 'coerce')
df.PTS = pd.to_numeric(df.PTS, errors = 'coerce')
df['+/-'] = pd.to_numeric(df['+/-'], errors = 'coerce')
df.PIM = pd.to_numeric(df.PIM, errors = 'coerce')
df['PTS/G'] = pd.to_numeric(df['PTS/G'], errors = 'coerce')
df.SOG = pd.to_numeric(df.SOG, errors = 'coerce')
df.PCT = pd.to_numeric(df.PCT, errors = 'coerce')
df.GWG = pd.to_numeric(df.GWG, errors = 'coerce')
df.PPG = pd.to_numeric(df.PPG, errors = 'coerce')
df.PPA = pd.to_numeric(df.PPA, errors = 'coerce')
df.SHG = pd.to_numeric(df.SHG, errors = 'coerce')
df.SHA = pd.to_numeric(df.SHA, errors = 'coerce')
print(df.dtypes)

print(df)
print(df.PCT.unique())