from operator import le
import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_file_gry = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion\\"
f = files[0]

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

data = pd.read_csv(f)

data['participant'] = participant
data['label'] = label
data['category'] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
df_acc = pd.DataFrame()
df_gyr = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
    data = pd.read_csv(f)
    
    data['participant'] = participant
    data['label'] = label
    data['category'] = category
    
    if "Accelerometer" in f:
        data['set'] = acc_set
        acc_set+=1
        df_acc = pd.concat([df_acc, data])
        
    if "Gyroscope" in f:
        data['set'] = gyr_set
        gyr_set+=1
        df_gyr = pd.concat([df_gyr, data])
        
df_acc[df_acc['set'] == 1]
        
    

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
df_acc.info()

pd.to_datetime(data['epoch (ms)'], unit="ms")

df_acc.index = pd.to_datetime(df_acc['epoch (ms)'], unit="ms")
df_gyr.index = pd.to_datetime(df_gyr['epoch (ms)'], unit="ms")

df_acc.drop(columns=['epoch (ms)', 'time (01:00)', 'elapsed (s)'], inplace=True)
df_gyr.drop(columns=['epoch (ms)', 'time (01:00)', 'elapsed (s)'], inplace=True)


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
def read_data_from_files(files):
    df_acc = pd.DataFrame()
    df_gyr = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        
        data = pd.read_csv(f)
        
        data['participant'] = participant
        data['label'] = label
        data['category'] = category
        
        if "Accelerometer" in f:
            data['set'] = acc_set
            acc_set+=1
            df_acc = pd.concat([df_acc, data])
            
        if "Gyroscope" in f:
            data['set'] = gyr_set
            gyr_set+=1
            df_gyr = pd.concat([df_gyr, data])
            
    df_acc[df_acc['set'] == 1]
    
    df_acc.index = pd.to_datetime(df_acc['epoch (ms)'], unit="ms")
    df_gyr.index = pd.to_datetime(df_gyr['epoch (ms)'], unit="ms")

    df_acc.drop(columns=['epoch (ms)', 'time (01:00)', 'elapsed (s)'], inplace=True)
    df_gyr.drop(columns=['epoch (ms)', 'time (01:00)', 'elapsed (s)'], inplace=True)

    return df_acc, df_gyr

df_acc, df_gyr = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_marged = pd.concat([df_acc.iloc[:,:3], df_gyr], axis=1)
data_marged.dropna()

data_marged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set"
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last"
}

data_marged[:1000].resample(rule="200ms").apply(sampling)

days = [g for n, g in data_marged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled['set'] = data_resampled['set'].astype("int")
data_resampled.info()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")