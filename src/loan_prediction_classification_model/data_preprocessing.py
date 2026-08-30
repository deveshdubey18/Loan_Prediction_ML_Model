# Step2: Data Preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats.mstats import winsorize
def preprocessing(df):

  # Remove Duplicates from Dataset
  df.drop_duplicates()
 
  # Segregate Categorical and Numerical Columns
  categorical_data = df.select_dtypes(include = 'object')
  numerical_data = df.select_dtypes(exclude = 'object')

  # Using Winsorization Technique : To Cap the Outliers
  for i in numerical_data.columns:
    df[i] = winsorize(df[i],limits = [0.05,0.05])

  # Encoding Categorical Data

  le = LabelEncoder()
  for i in categorical_data.columns:
    df[i] = le.fit_transform(df[i])

  # Split the Datatset into X and y
  X = df.drop(columns = ['loan_status'])
  y = df['loan_status']

  # Split the Dataset into Train and Test i.e. Seen Data and Unseen Data
  X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state = 1)

  
  # Use Scaling Technique

  sc = MinMaxScaler()
  X_train = sc.fit_transform(X_train) # Seen Data
  X_test = sc.transform(X_test)       # Unseen Data
  
  # Use SMOTE Technique
  sm = SMOTE()
  X_train,y_train = sm.fit_resample(X_train,y_train) # type: ignore


  return X_train,X_test,y_train,y_test