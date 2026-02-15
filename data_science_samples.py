#!/usr/bin/env python3
"""
Data Science & Analytics Sample Data
------------------------------------
Comprehensive sample documents for building a data science RAG system.
"""

from add_documents import DocumentIngester

def get_data_science_samples():
    """Return comprehensive data science and analytics content."""
    
    return [
        {
            'content': """
# Data Analysis Fundamentals

Data analysis is the process of inspecting, cleansing, transforming, and modeling data to discover useful information, inform conclusions, and support decision-making.

## Key Concepts

**Exploratory Data Analysis (EDA)**: The initial investigation of data to discover patterns, spot anomalies, test hypotheses, and check assumptions through summary statistics and graphical representations.

**Data Cleaning**: The process of detecting and correcting (or removing) corrupt or inaccurate records from a dataset. Common tasks include handling missing values, removing duplicates, fixing structural errors, and addressing outliers.

**Feature Engineering**: The process of using domain knowledge to extract features from raw data. This includes creating new variables, transforming existing ones, encoding categorical variables, and scaling numerical features.

## Common Techniques

1. **Univariate Analysis**: Analyzing single variables using histograms, box plots, and summary statistics
2. **Bivariate Analysis**: Examining relationships between two variables using scatter plots, correlation coefficients, and cross-tabulations
3. **Multivariate Analysis**: Studying multiple variables simultaneously through techniques like PCA, clustering, and regression

## Tools

- Python: pandas, numpy, scipy
- Visualization: matplotlib, seaborn, plotly
- Statistical: statsmodels, scikit-learn
""",
            'metadata': {'source': 'data_analysis_fundamentals', 'domain': 'datascience', 'topic': 'analysis'}
        },
        
        {
            'content': """
# Statistical Methods in Data Science

Statistics provides the theoretical foundation for data science. Understanding statistical methods is crucial for making valid inferences from data.

## Descriptive Statistics

**Measures of Central Tendency**:
- Mean: Average of all values
- Median: Middle value when data is ordered
- Mode: Most frequently occurring value

**Measures of Dispersion**:
- Variance: Average squared deviation from mean
- Standard Deviation: Square root of variance
- Interquartile Range (IQR): Spread of middle 50% of data

## Inferential Statistics

**Hypothesis Testing**: A method for testing claims about population parameters using sample data. Steps include:
1. State null and alternative hypotheses
2. Choose significance level (α)
3. Calculate test statistic
4. Determine p-value
5. Make decision: reject or fail to reject null hypothesis

**Common Tests**:
- t-test: Compare means of two groups
- ANOVA: Compare means of three or more groups
- Chi-square test: Test relationships between categorical variables
- Correlation tests: Measure strength of relationships

**Confidence Intervals**: Range of values that likely contains the true population parameter with specified confidence level (e.g., 95% CI)

## Regression Analysis

**Linear Regression**: Models relationship between dependent variable and one or more independent variables. Assumes linear relationship, independence, homoscedasticity, and normality of residuals.

**Logistic Regression**: Used for binary classification problems. Predicts probability of outcome using logistic function.

**Key Metrics**:
- R-squared: Proportion of variance explained by model
- RMSE: Root Mean Square Error for continuous predictions
- Coefficients: Indicate strength and direction of relationships
""",
            'metadata': {'source': 'statistics_methods', 'domain': 'datascience', 'topic': 'statistics'}
        },
        
        {
            'content': """
# Data Visualization Best Practices

Effective data visualization transforms complex data into intuitive visual representations that facilitate understanding and decision-making.

## Principles of Good Visualization

**1. Know Your Audience**: Tailor complexity and detail to viewer's expertise level

**2. Choose Appropriate Chart Types**:
- Line charts: Trends over time
- Bar charts: Comparisons across categories
- Scatter plots: Relationships between variables
- Heatmaps: Patterns in matrix data
- Box plots: Distribution and outliers
- Pie charts: Parts of a whole (use sparingly)

**3. Design for Clarity**:
- Remove chart junk and unnecessary elements
- Use consistent color schemes
- Label axes clearly with units
- Provide informative titles
- Include legends when needed

## Color Usage

- Use color purposefully to highlight key information
- Consider color blindness (avoid red-green combinations only)
- Limit palette to 5-7 distinct colors
- Use sequential colors for continuous data
- Use diverging colors for data with meaningful midpoint

## Common Mistakes to Avoid

1. **Misleading axes**: Always start y-axis at zero for bar charts
2. **3D effects**: They distort perception without adding value
3. **Too much information**: Overcrowding reduces comprehension
4. **Poor aspect ratio**: Can exaggerate or minimize trends
5. **No context**: Always provide scale, units, and reference points

## Interactive Visualizations

Modern tools enable interactive exploration:
- Plotly: Web-based interactive plots
- Bokeh: Interactive visualization for large datasets
- Dash: Full dashboards with callbacks
- Altair: Declarative statistical visualization
""",
            'metadata': {'source': 'data_visualization', 'domain': 'datascience', 'topic': 'visualization'}
        },
        
        {
            'content': """
# Pandas Library Guide

Pandas is the essential Python library for data manipulation and analysis, providing data structures and operations for manipulating numerical tables and time series.

## Core Data Structures

**Series**: One-dimensional labeled array capable of holding any data type
```python
import pandas as pd
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
```

**DataFrame**: Two-dimensional labeled data structure with columns of potentially different types
```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})
```

## Essential Operations

**Data Loading**:
- `pd.read_csv()`: Load CSV files
- `pd.read_excel()`: Load Excel files
- `pd.read_sql()`: Load from databases
- `pd.read_json()`: Load JSON data

**Data Selection**:
- `.loc[]`: Label-based indexing
- `.iloc[]`: Integer-based indexing
- `.query()`: SQL-like filtering
- Boolean indexing: `df[df['age'] > 25]`

**Data Manipulation**:
- `.groupby()`: Group data for aggregation
- `.merge()`: Join dataframes
- `.pivot_table()`: Create pivot tables
- `.apply()`: Apply functions to rows/columns
- `.fillna()`: Handle missing values
- `.drop_duplicates()`: Remove duplicates

**Aggregations**:
- `.mean()`, `.sum()`, `.count()`: Basic statistics
- `.describe()`: Summary statistics
- `.value_counts()`: Frequency counts
- `.agg()`: Multiple aggregations

## Performance Tips

1. Use vectorized operations instead of loops
2. Use categorical dtype for low-cardinality string columns
3. Read only needed columns: `usecols` parameter
4. Use chunking for large files: `chunksize` parameter
5. Optimize dtypes: `int32` instead of `int64` when appropriate
""",
            'metadata': {'source': 'pandas_guide', 'domain': 'datascience', 'topic': 'pandas'}
        },
        
        {
            'content': """
# SQL for Data Analysis

SQL (Structured Query Language) is fundamental for data analysts to extract, transform, and analyze data stored in relational databases.

## Core SQL Concepts

**SELECT Statement**: Retrieve data from database
```sql
SELECT column1, column2
FROM table_name
WHERE condition
ORDER BY column1 DESC
LIMIT 10;
```

**Filtering with WHERE**:
- Comparison: `=`, `>`, `<`, `>=`, `<=`, `<>`
- Pattern matching: `LIKE '%pattern%'`
- Range: `BETWEEN value1 AND value2`
- List: `IN (value1, value2, value3)`
- Null checks: `IS NULL`, `IS NOT NULL`

## Aggregation and Grouping

**Aggregate Functions**:
- `COUNT()`: Count rows
- `SUM()`: Sum values
- `AVG()`: Calculate average
- `MIN()` / `MAX()`: Find extremes
- `GROUP_CONCAT()`: Concatenate values

**GROUP BY**: Group rows sharing property
```sql
SELECT department, AVG(salary) as avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 50000;
```

## Joins

**INNER JOIN**: Returns matching rows from both tables
**LEFT JOIN**: All rows from left table, matching from right
**RIGHT JOIN**: All rows from right table, matching from left
**FULL OUTER JOIN**: All rows from both tables

Example:
```sql
SELECT orders.id, customers.name, orders.total
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id;
```

## Window Functions

Perform calculations across rows related to current row:
- `ROW_NUMBER()`: Assign unique number to each row
- `RANK()`: Rank with gaps for ties
- `DENSE_RANK()`: Rank without gaps
- `LAG()` / `LEAD()`: Access previous/next row
- `PARTITION BY`: Create separate windows

Example:
```sql
SELECT name, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees;
```

## Common Table Expressions (CTEs)

Improve query readability with temporary named result sets:
```sql
WITH high_performers AS (
    SELECT employee_id, AVG(performance_score) as avg_score
    FROM reviews
    GROUP BY employee_id
    HAVING AVG(performance_score) > 4.0
)
SELECT e.name, hp.avg_score
FROM employees e
JOIN high_performers hp ON e.id = hp.employee_id;
```

## Query Optimization

1. **Indexes**: Create indexes on frequently queried columns
2. **Limit results**: Use LIMIT to restrict row count
3. **Select specific columns**: Avoid `SELECT *`
4. **Efficient joins**: Join on indexed columns
5. **Avoid subqueries in WHERE**: Use joins instead when possible
""",
            'metadata': {'source': 'sql_guide', 'domain': 'datascience', 'topic': 'sql'}
        },
        
        {
            'content': """
# A/B Testing and Experimentation

A/B testing is a statistical method for comparing two versions of a variable to determine which performs better. It's essential for data-driven decision making.

## Experimental Design

**Key Components**:
1. **Hypothesis**: Clear statement of expected outcome
2. **Metrics**: Quantifiable measures of success
3. **Sample size**: Sufficient participants for statistical power
4. **Duration**: Long enough to capture variation
5. **Randomization**: Unbiased assignment to groups

**Types of Tests**:
- A/B test: Compare two versions
- A/B/n test: Compare multiple versions
- Multivariate test: Test multiple variables simultaneously

## Statistical Foundations

**Null Hypothesis (H0)**: No difference between groups
**Alternative Hypothesis (H1)**: Significant difference exists

**Type I Error (False Positive)**: Concluding difference when none exists
- Controlled by significance level (α), typically 0.05

**Type II Error (False Negative)**: Missing real difference
- Related to statistical power (1-β), typically aim for 0.80

**Sample Size Calculation**:
Depends on:
- Baseline conversion rate
- Minimum detectable effect (MDE)
- Significance level (α)
- Statistical power (1-β)

## Analysis Methods

**T-test**: Compare means of two groups
- Independent samples t-test for A/B testing
- Check assumptions: normality, equal variance

**Proportion Test**: Compare conversion rates
- Z-test for proportions
- Chi-square test for categorical outcomes

**Confidence Intervals**: Estimate range of true effect
- 95% CI means 95% probability true value lies within interval

## Best Practices

1. **Pre-register experiments**: Document design before running
2. **Avoid peeking**: Don't check results early and stop test
3. **Account for multiple testing**: Use Bonferroni correction
4. **Consider practical significance**: Not just statistical significance
5. **Segment analysis**: Examine effects in subgroups
6. **Long-term effects**: Monitor beyond immediate impact

## Common Pitfalls

- **Insufficient sample size**: Leads to inconclusive results
- **Short duration**: Doesn't capture weekly patterns
- **Selection bias**: Non-random assignment
- **Novelty effect**: Temporary boost from change
- **Interference**: One group affects another (spillover)

## Tools and Platforms

- Google Optimize: Free A/B testing for web
- Optimizely: Enterprise experimentation platform
- VWO: Conversion optimization platform
- Python libraries: scipy.stats, statsmodels
- R packages: pwr, broom
""",
            'metadata': {'source': 'ab_testing', 'domain': 'datascience', 'topic': 'experimentation'}
        },
        
        {
            'content': """
# Time Series Analysis

Time series analysis involves methods for analyzing temporal data to extract meaningful statistics and characteristics, often for forecasting.

## Time Series Components

**1. Trend**: Long-term increase or decrease in data
**2. Seasonality**: Regular pattern that repeats over fixed period
**3. Cyclical**: Fluctuations without fixed frequency
**4. Irregular/Residual**: Random variation

**Decomposition**: Separating time series into components
- Additive: Y = T + S + C + I
- Multiplicative: Y = T × S × C × I

## Stationarity

**Stationary Series**: Statistical properties don't change over time
- Constant mean
- Constant variance
- Autocovariance depends only on lag

**Tests for Stationarity**:
- Augmented Dickey-Fuller (ADF) test
- KPSS test
- Visual inspection: Rolling statistics

**Making Series Stationary**:
- Differencing: Remove trend
- Log transformation: Stabilize variance
- Detrending: Subtract trend component

## Forecasting Models

**Moving Average (MA)**: Average of recent observations
- Simple MA: Equal weights
- Weighted MA: Different weights for observations
- Exponential smoothing: Exponentially decreasing weights

**ARIMA (AutoRegressive Integrated Moving Average)**:
- AR(p): Regression on past values
- I(d): Differencing to achieve stationarity
- MA(q): Regression on past errors

**Seasonal ARIMA (SARIMA)**: Extends ARIMA for seasonal patterns

**Prophet**: Facebook's forecasting tool
- Handles missing data and outliers
- Accounts for holidays and events
- Provides uncertainty intervals

## Evaluation Metrics

**Point Forecast Accuracy**:
- MAE: Mean Absolute Error
- RMSE: Root Mean Square Error
- MAPE: Mean Absolute Percentage Error

**Forecast Intervals**: Range of predicted values with confidence level

## Python Implementation

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ARIMA example
model = ARIMA(data, order=(1,1,1))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=30)

# Prophet example
df = pd.DataFrame({'ds': dates, 'y': values})
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

## Applications

- Sales forecasting
- Demand planning
- Stock price prediction
- Weather forecasting
- Economic indicators
- Website traffic prediction
""",
            'metadata': {'source': 'time_series', 'domain': 'datascience', 'topic': 'time_series'}
        },
        
        {
            'content': """
# Machine Learning for Data Scientists

Machine learning enables computers to learn from data and make predictions or decisions without being explicitly programmed.

## Supervised Learning

**Classification**: Predict categorical labels
- Logistic Regression
- Decision Trees and Random Forests
- Support Vector Machines (SVM)
- Neural Networks
- k-Nearest Neighbors (k-NN)

**Regression**: Predict continuous values
- Linear Regression
- Ridge and Lasso Regression
- Random Forest Regressor
- Gradient Boosting (XGBoost, LightGBM)

## Unsupervised Learning

**Clustering**: Group similar data points
- K-Means: Partition into k clusters
- DBSCAN: Density-based clustering
- Hierarchical: Build cluster tree
- Gaussian Mixture Models: Probabilistic clustering

**Dimensionality Reduction**:
- PCA: Principal Component Analysis
- t-SNE: Visualization of high-dimensional data
- UMAP: Uniform Manifold Approximation

## Model Development Process

**1. Data Preparation**:
- Train/validation/test split (60/20/20 or 70/15/15)
- Handle missing values
- Encode categorical variables
- Scale/normalize features

**2. Feature Engineering**:
- Create interaction terms
- Polynomial features
- Domain-specific features
- Feature selection techniques

**3. Model Selection**:
- Start with simple baseline
- Try multiple algorithms
- Consider computational cost
- Balance interpretability vs performance

**4. Hyperparameter Tuning**:
- Grid Search: Exhaustive search
- Random Search: Sample parameter space
- Bayesian Optimization: Intelligent search

**5. Model Evaluation**:
- Cross-validation: k-fold CV
- Metrics selection based on problem
- Check for overfitting/underfitting

## Evaluation Metrics

**Classification**:
- Accuracy: Overall correctness
- Precision: True positives / predicted positives
- Recall: True positives / actual positives
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under ROC curve
- Confusion Matrix: Detailed error analysis

**Regression**:
- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- R²: Coefficient of determination
- MAPE: Mean Absolute Percentage Error

## Best Practices

1. **Establish baseline**: Simple model to compare against
2. **Cross-validation**: Don't rely on single split
3. **Feature importance**: Understand what drives predictions
4. **Model explainability**: Use SHAP, LIME for interpretation
5. **Monitor performance**: Track metrics over time in production
6. **A/B testing**: Validate real-world impact

## Common Pitfalls

- **Data leakage**: Information from test set enters training
- **Overfitting**: Model memorizes training data
- **Underfitting**: Model too simple to capture patterns
- **Ignoring class imbalance**: Accuracy misleading with imbalanced classes
- **Not considering business context**: Technical metrics vs business value
""",
            'metadata': {'source': 'ml_for_ds', 'domain': 'datascience', 'topic': 'machine_learning'}
        }
    ]


def main():
    """Add all data science samples to knowledge base."""
    
    print("="*70)
    print("📚 ADDING DATA SCIENCE SAMPLE CONTENT")
    print("="*70)
    print()
    
    ingester = DocumentIngester()
    samples = get_data_science_samples()
    
    print(f"Processing {len(samples)} data science documents...")
    print()
    
    total_chunks = 0
    for i, sample in enumerate(samples, 1):
        topic = sample['metadata']['topic']
        print(f"[{i}/{len(samples)}] Adding {topic}...")
        
        try:
            chunks = ingester._add_documents_to_kb([{
                'text': sample['content'],
                'metadata': sample['metadata']
            }])
            total_chunks += chunks
            print(f"  ✅ Added {chunks} chunks (Total: {total_chunks})")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print()
    print("="*70)
    print(f"✅ Successfully added {total_chunks} chunks to knowledge base")
    print("="*70)
    print()
    print("💡 Topics covered:")
    for sample in samples:
        print(f"  • {sample['metadata']['topic'].replace('_', ' ').title()}")
    
    return total_chunks


if __name__ == '__main__':
    main()
