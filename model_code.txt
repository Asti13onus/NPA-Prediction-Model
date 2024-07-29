import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset from the specified file path
file_path = "C:/Users/91807/Downloads/MSME Pulse Reports Data - Sheet1 (1).csv"
df = pd.read_csv(file_path)

# Display first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# summary of dataset to check data types and missing values
print("\nDataset information:")
print(df.info())

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Remove columns with all missing values
df.dropna(axis=1, how='all', inplace=True)

# Print column names to identify correct target column
print("\nColumn names in the dataset:")
print(df.columns)

# Data Cleaning

# Select numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns

# Fill missing values with the mean for numerical columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Drop rows with missing values 
df.dropna(inplace=True)

# Convert data types if not done already
for col in df.columns:
    if df[col].dtype == 'object':  # Convert object columns to categorical if necessary
        df[col] = df[col].astype('category').cat.codes

# Print first few rows after cleaning
print("\nFirst few rows after cleaning:")
print(df.head())

# Specify the actual target column
target_column = 'Industry (Micro, Small, Medium and Large) - Total NPA %'

# Check if the target column is present
if target_column not in df.columns:
    print(f"\nError: Target column '{target_column}' not found in the dataset. Please update the target_column variable.")
else:
    # Identify features and target
    features = df.drop(columns=target_column)
    target = df[target_column]

    # Normalize/Scale data
    # 1. Standardization
    scaler_standard = StandardScaler()
    features_standardized = scaler_standard.fit_transform(features)

    # 2. Min-Max Scaling
    scaler_minmax = MinMaxScaler()
    features_minmax_scaled = scaler_minmax.fit_transform(features)

    print("\nFirst few rows of standardized features:")
    print(pd.DataFrame(features_standardized, columns=features.columns).head())

    # display min-max scaled features
    print("\nFirst few rows of Min-Max scaled features:")
    print(pd.DataFrame(features_minmax_scaled, columns=features.columns).head())


#Applying on the dataset with no empty cells beforehand, focusing on only few columns
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset from the specified file path
file_path = "C:/Users/91807/Downloads/MSME Pulse Reports Data-Filtered - Sheet1.csv"
df = pd.read_csv(file_path)

# Display first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display summary of the dataset to check data types and missing values
print("\nDataset information:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Remove columns with all missing values
df.dropna(axis=1, how='all', inplace=True)

# Print column names to identify correct target column
print("\nColumn names in the dataset:")
print(df.columns)

# Data Cleaning

# Select numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns

# Fill missing values with the mean for numerical columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Drop rows with missing values 
df.dropna(inplace=True)

# Convert data types if necessary
for col in df.columns:
    if df[col].dtype == 'object':  # Convert object columns to categorical if necessary
        df[col] = df[col].astype('category').cat.codes


print("\nFirst few rows after cleaning:")
print(df.head())

target_column = 'Industry (Micro, Small, Medium and Large) - Total NPA %'


if target_column not in df.columns:
    print(f"\nError: Target column '{target_column}' not found in the dataset. Please update the target_column variable.")
else:
    # Identify features and target
    features = df.drop(columns=target_column)
    target = df[target_column]

    # Normalize/Scale data

    #Standardization
    scaler_standard = StandardScaler()
    features_standardized = scaler_standard.fit_transform(features)

    # Min-Max Scaling
scaler_minmax = MinMaxScaler()
features_minmax_scaled = scaler_minmax.fit_transform(features)

    # Print first few rows of standardized features
print("\nFirst few rows of standardized features:")
print(pd.DataFrame(features_standardized, columns=features.columns).head())

    # Min-Max scaled features
print("\nFirst few rows of Min-Max scaled features:")
print(pd.DataFrame(features_minmax_scaled, columns=features.columns).head())


correlation_matrix = df.corr()
print(correlation_matrix['Industry (Micro, Small, Medium and Large) - Total NPA %'].sort_values(ascending=False))


from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(df['Industry (Micro, Small, Medium and Large) - Total NPA %'])

chi2_scores, p_values = chi2(features, encoded_target)


from sklearn.ensemble import RandomForestRegressor
import pandas as pd

model = RandomForestRegressor()
model.fit(features, target)

importance = model.feature_importances_
feature_importance = pd.Series(importance, index=features.columns).sort_values(ascending=False)

print("Feature Importances:")
print(feature_importance)


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

model = RandomForestRegressor()

# Initialize RFE with model and no. of features to select
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(features, target)

# print selected features
print("Selected features:", fit.support_)
print("Feature ranking:", fit.ranking_)


#using Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

model = LinearRegression()

# Initialize RFE with model and no. of features to select
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(features, target)

# print selected features
print("Selected features:", fit.support_)
print("Feature ranking:", fit.ranking_)


from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k='all')
fit = selector.fit(features, target)
scores = pd.Series(fit.scores_, index=features.columns)
print(scores.sort_values(ascending=False))


# check if required columns exist
if 'Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)' in df.columns and 'Industry (Micro, Small, Medium and Large) -  Total o/s Advances (in million rupees)' in df.columns:
    # Calculate Growth Rate
    df['Growth_Rate'] = (df['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)'] / df['Industry (Micro, Small, Medium and Large) -  Total o/s Advances (in million rupees)'] - 1) * 100
    print("Growth Rate calculated successfully.")
else:
    print("Required columns are missing.")

print(df)
print(df[['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)', 'Industry (Micro, Small, Medium and Large) -  Total o/s Advances (in million rupees)', 'Growth_Rate']])
print(df[['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)', 'Industry (Micro, Small, Medium and Large) -  Total o/s Advances (in million rupees)', 'Growth_Rate']].head())
print(df['Growth_Rate'].describe())
df.to_csv("report_with_growth_rate.csv", index=False)


df['Ratio'] = df['Industry (Micro, Small, Medium and Large) -  Total o/s Advances (in million rupees)'] / df['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)']
print(df)
print(df[['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)', 'Industry (Micro, Small, Medium and Large) -  Total o/s Advances (in million rupees)', 'Ratio']])
print(df[['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)', 'Industry (Micro, Small, Medium and Large) -  Total o/s Advances (in million rupees)', 'Ratio']].head())
print(df['Ratio'].describe())
df.to_csv("report_with_ratio.csv", index=False)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # Reduce to 2 dimensions
principal_components = pca.fit_transform(features)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])


import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
features = df.drop(columns=['Industry (Micro, Small, Medium and Large) - Total NPA %']) #our target column

# check no. of samples
num_samples = features.shape[0]
print("Number of samples:", num_samples)

# adjust perplexity or use PCA for dimensionality reduction before t-SNE
if num_samples > 30:
    tsne = TSNE(n_components=2, perplexity=min(30, num_samples - 1))
    tsne_results = tsne.fit_transform(features)
    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
else:
    # Use UMAP if t-SNE is not feasible
    umap_model = umap.UMAP(n_components=2)
    umap_results = umap_model.fit_transform(features)
    tsne_df = pd.DataFrame(data=umap_results, columns=['UMAP1', 'UMAP2'])

print(tsne_df.head())


import pandas as pd
from sklearn.model_selection import train_test_split
features = df.drop(columns=['Industry (Micro, Small, Medium and Large) - Total NPA %'])  
target = df['Industry (Micro, Small, Medium and Large) - Total NPA %'] 

# split data into training and test sets (85% training, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.15, random_state=42)

# split temporary set into validation and test sets (50% validation, 50% test of the 15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)
print("Test set size:", X_test.shape)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20]
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predictions
y_val_pred = best_model.predict(X_val_scaled)
y_test_pred = best_model.predict(X_test_scaled)

# Evaluation metrics
print("Validation MAE:", mean_absolute_error(y_val, y_val_pred))
print("Validation MSE:", mean_squared_error(y_val, y_val_pred))
print("Validation R^2:", r2_score(y_val, y_val_pred))

print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))


importances = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(importance_df.sort_values(by='Importance', ascending=False))


import matplotlib.pyplot as plt
import seaborn as sns

# Plot histogram for each feature
df.hist(figsize=(12, 10), bins=30)
plt.tight_layout()
plt.show()


# Scatter plot between two features
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Industry (Micro, Small, Medium and Large) - Total NPA %', y='Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)', data=df)
plt.title('Scatter Plot between Industry (Micro, Small, Medium and Large) - Total NPA % and Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)')
plt.show()



# Scatter plot between two features
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Industry (Micro, Small, Medium and Large) - Total NPA %', y='Growth_Rate', data=df)
plt.title('Scatter Plot between Industry (Micro, Small, Medium and Large) - Total NPA % and Growth_Rate')
plt.show()


# Box plot for a feature
plt.figure(figsize=(10, 6))
sns.boxplot(x='Industry (Micro, Small, Medium and Large) - Total NPA %', data=df)
plt.title('Box Plot of Feature1')
plt.show()


# Box plot for a feature
plt.figure(figsize=(10, 6))
sns.boxplot(x='Growth_Rate', data=df)
plt.title('Box Plot of Feature1')
plt.show()

# Assuming 'Date' is a datetime column and 'Target' is the target variable
plt.figure(figsize=(12, 6))
df.set_index('Year')['Industry (Micro, Small, Medium and Large) - Total NPA %'].plot()
plt.title('Time Series Plot of Target Variable')
plt.xlabel('Date')
plt.ylabel('Industry (Micro, Small, Medium and Large) - Total NPA %')
plt.show()


# Assuming 'Date' is a datetime column and 'Target' is the target variable
plt.figure(figsize=(12, 6))
df.set_index('Year')['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)'].plot()
plt.title('Time Series Plot of Target Variable')
plt.xlabel('Date')
plt.ylabel('Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)')
plt.show()


# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# Correlation of features with target variable
target_corr = corr_matrix['Industry (Micro, Small, Medium and Large) - Total NPA %'].sort_values(ascending=False)
print("Correlation with Target:\n", target_corr)


# Correlation of features with target variable
target_corr = corr_matrix['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)'].sort_values(ascending=False)
print("Correlation with Target:\n", target_corr)


# Correlation of features with target variable
target_corr = corr_matrix['Industry (Micro, Small, Medium and Large) -  Total o/s Advances (in million rupees)'].sort_values(ascending=False)
print("Correlation with Target:\n", target_corr)


# Correlation of features with target variable
target_corr = corr_matrix['Growth_Rate'].sort_values(ascending=False)
print("Correlation with Target:\n", target_corr)

# Correlation of features with target variable
target_corr = corr_matrix['Ratio'].sort_values(ascending=False)
print("Correlation with Target:\n", target_corr)

# Descriptive statistics
print(df.describe())


from scipy import stats

# T-test between two groups (example)
group1 = df[df['Industry (Micro, Small, Medium and Large) - Total NPA %'] == 'A']['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)']
group2 = df[df['Industry (Micro, Small, Medium and Large) - Total NPA %'] == 'B']['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)']
t_stat, p_val = stats.ttest_ind(group1, group2)
print(f"T-statistic: {t_stat}, P-value: {p_val}")


from scipy.stats import chi2_contingency

# Chi-Square test (example)
contingency_table = pd.crosstab(df['Industry (Micro, Small, Medium and Large) - Total NPA %'], df['Industry (Micro, Small, Medium and Large) -  Total Gross NPAs (in million rupees)'])
chi2_stat, p_val, dof, ex = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2_stat}, P-value: {p_val}")


from scipy import stats
# Perform ANOVA
anova_result = stats.f_oneway(
    df[df['Year'] == 'A']['Industry (Micro, Small, Medium and Large) - Total NPA %'],
    df[df['Year'] == 'B']['Industry (Micro, Small, Medium and Large) - Total NPA %'],
    df[df['Year'] == 'C']['Industry (Micro, Small, Medium and Large) - Total NPA %']
)

# Extracting the results
f_statistic = anova_result.statistic
p_value = anova_result.pvalue

print(f"ANOVA F-Statistic: {f_statistic}")
print(f"ANOVA P-Value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("There is a significant difference between the means of the groups.")
else:
    print("There is no significant difference between the means of the groups.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping


import numpy as np
data = pd.DataFrame({
    'Year': np.random.rand(100),
    'Industry (Micro, Small, Medium and Large) - Total NPA %': np.random.rand(100)
})

# Features and target
X = data[['Year']].values
y = data['Industry (Micro, Small, Medium and Large) - Total NPA %'].values

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Function to create sequences for LSTM
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

# Parameters
seq_length = 10

# Create sequences
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

# Split data
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]


from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

# Define the model
model = Sequential()

# Define the input shape using Input layer
model.add(Input(shape=(16,4)))  # Adjust `timesteps` and `num_features` to your data

# Add LSTM layer
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))

# Add Dense layer
model.add(Dense(1))  # Adjust the number of units based on your output

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()



print(X_train.shape)  # Should print (num_samples, timesteps, num_features)



from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

# Define the model
model = Sequential()

# Define the input shape
model.add(Input(shape=(10,1 )))  # Ensure timesteps and num_features match your data

# Add LSTM layers
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))

# Add Dense layer for the output
model.add(Dense(1))  # Adjust the number of units based on your output

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()


# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)


# Predict on test data
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mae = mean_absolute_error(y_test_original, y_pred)
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)

print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R-squared: {r2:.4f}')


plt.figure(figsize=(12, 6))

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


import gym
from gym import spaces
import numpy as np

class FinancialEnv(gym.Env):
    def __init__(self, data):
        super(FinancialEnv, self).__init__()
        
        # Define action and observation space
        # Actions: Buy, Hold, Sell
        self.action_space = spaces.Discrete(3)  
        
        # Observation space: State features (e.g., market indicators)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)
        
        # Initialize data
        self.data = data
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        # Apply action and get reward
        reward = self._take_action(action)
        
        # Move to next state
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Get next observation
        obs = self.data[self.current_step] if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, {}

    def _take_action(self, action):
        # Define logic to calculate reward based on action
        # Example: reward based on profit/loss
        return np.random.rand()  # Placeholder for reward calculation

    def render(self, mode='human'):
        # Implement visualization if needed
        pass


from stable_baselines3 import DQN

# Initialize the environment with your dataset
data = np.random.rand(1000, 10)  # Example dataset
env = FinancialEnv(data)

# Create DQN model
model = DQN('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)


# Train the RL agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("dqn_financial_model")


# Load the model
model = DQN.load("dqn_financial_model")

# Test the agent
obs = env.reset()
for _ in range(len(data)):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

# Example data (replace this with your actual dataset)
# Suppose df is your DataFrame and 'target' is your target column
df = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
df['Industry (Micro, Small, Medium and Large) - Total NPA %'] = np.random.randint(0, 2, size=(100,))

# Split into features and target
X = df.drop(columns=['Industry (Micro, Small, Medium and Large) - Total NPA %'])
y = df['Industry (Micro, Small, Medium and Large) - Total NPA %']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values for the training data
shap_values = explainer.shap_values(X_train)

# Plot SHAP values
shap.summary_plot(shap_values, X_train)



from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load example dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the base model
base_model = DecisionTreeClassifier()

# Initialize the bagging model without base_estimator
bagging_model = BaggingClassifier(base_model, n_estimators=50, random_state=42)

# Train the bagging model
bagging_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of Bagging Classifier:", accuracy)


from sklearn.ensemble import GradientBoostingClassifier

# Initialize and train the model
boosting_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
boosting_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = boosting_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Boosting Model Accuracy: {accuracy:.2f}')


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the model
model = RandomForestClassifier()

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and score
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_:.2f}')


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the model
model = GradientBoostingClassifier()

# Define the parameter distribution
param_dist = {
    'n_estimators': randint(50, 200),
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': randint(3, 10)
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best parameters and score
print(f'Best Parameters: {random_search.best_params_}')
print(f'Best Score: {random_search.best_score_:.2f}')


import matplotlib.pyplot as plt
import numpy as np

# Assuming y_test and y_pred are your actual and predicted values
# For demonstration purposes, let's create dummy data
y_test = np.random.randint(0, 2, size=100)  # Actual values
y_pred = np.random.randint(0, 2, size=100)  # Predicted values

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual values')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted values', alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Predicted vs Actual Values')
plt.legend()
plt.show()


import matplotlib.pyplot as plt

# Create example time-series data
dates = np.arange('2023-01', '2024-01', dtype='datetime64[M]')
historical_data = np.sin(np.linspace(0, 10, len(dates)))  # Example historical data
predicted_trends = np.sin(np.linspace(0, 10.5, len(dates)))  # Example future trends

plt.figure(figsize=(12, 6))
plt.plot(dates, historical_data, label='Historical Data', color='blue')
plt.plot(dates, predicted_trends, label='Predicted Trends', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Historical Data vs Predicted Trends')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assuming y_test and y_pred are your actual and predicted values
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Assuming y_test and y_pred_prob are from a multiclass classifier
# Binarize the output
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Adjust classes as needed
y_pred_prob_bin = model.predict_proba(X_test)

# Plot ROC curve for each class
plt.figure()
for i in range(y_test_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob_bin[:, i])
    roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_prob_bin[:, i])
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Multiclass')
plt.legend(loc='lower right')
plt.show()


import seaborn as sns

# Assuming df is your DataFrame with features and labels
sns.pairplot(df, hue='Industry (Micro, Small, Medium and Large) - Total NPA %')
plt.show()


from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1)

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', color='red', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.grid(True)
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Train a model (e.g., RandomForest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Plot partial dependence for each class
features = [0, 1]  # Features to plot
for target_class in range(model.n_classes_):
    fig, ax = plt.subplots(figsize=(10, 6))
    display = PartialDependenceDisplay.from_estimator(
        model, X_train, features, ax=ax, target=target_class
    )
    plt.title(f"Partial Dependence for Class {target_class}")
    plt.show()


