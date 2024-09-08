import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Loading the data from the csv file
df = pd.read_csv('ai_job_market_insights.csv')
# Salary chart
fig = px.line(df,
              y='Salary_USD',
              title='Line Chart of Salary in USD',
              hover_data={'Salary_USD': True, 'Job_Title': True, 'Location': True})
fig.show()

# Job title distribution pie chart
job_title_distribution = df['Job_Title'].value_counts()
job_title_distribution.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.title('Distribution of Job Titles')
plt.ylabel('')
plt.show()

# Remote friendly distribution pie chart
remote_friendly_distribution = df['Remote_Friendly'].value_counts()
remote_friendly_distribution.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.title('Remote Friendly Distribution')
plt.ylabel('')
plt.show()

# Features and target
X = df[['Job_Title', 'Remote_Friendly', 'Company_Size', 'Location', 'Automation_Risk', 'AI_Adoption_Level', 'Job_Growth_Projection', 'Industry']]
y = df['Salary_USD']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Job_Title', 'Remote_Friendly', 'Company_Size', 'Location', 'Automation_Risk', 'AI_Adoption_Level', 'Job_Growth_Projection', 'Industry'])
    ])

# Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X, y)
# Train-test split with a smaller test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the best model
best_pipeline = grid_search.best_estimator_
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

# Evaluate model performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
accuracy_percentage = r2 * 100

print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")
print(f"Mean Absolute Error: {mae:.2f}")
