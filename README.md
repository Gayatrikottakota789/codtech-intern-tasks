# codtech-intern-tasks
This is the internship in data analytics cousre from april-10 to may-10
In this internship there are 4 tasks in data analytics concepts . In which we have to complete all the 4 tasks.
**Task-1: Big Data analysis**
In this task I performed analysis on big data that is sales-data using** pyspark.**
PySpark is the Python API for Apache Spark — a fast, in-memory data processing engine used for big data analytics.
It helps process huge datasets quickly by distributing the data across many machines (called clusters).
A Python script or Jupyter notebook that performs scalable data analysis.
Should include:
Dataset loading
Distributed processing (e.g., filtering, aggregation)
Insights and visualizations if applicable
Recommended Tools:
PySpark, Dask, pandas, matplotlib
**Taks-2 : Predictive analysis using machine learning:**
1. Data Loading
Loaded the CSV file using pandas.
Displayed the first 5 rows to understand the structure.
2. Exploratory Data Analysis
Used df.info() to examine data types and null values.
Plotted a correlation heatmap using seaborn to visualize feature relationships.
3. Data Preparation
Defined:
Features (X): All columns except WillPass.
Target (y): The WillPass column (binary classification).
4. Train-Test Split
Split the data into training and test sets with an 80:20 ratio using train_test_split.
5. Model Training
Used RandomForestClassifier from sklearn.ensemble.
Trained the model on the training dataset.
6. Model Evaluation
Evaluated predictions on the test set:
Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)
7. Feature Importance
Extracted and visualized feature importances using a horizontal bar chart.
📌 Libraries Used:
pandas — for data handling
matplotlib, seaborn — for visualization
sklearn — for model training and evaluation

**Task-3:Dashboard development using powerBI**
In this i developed a interactive dashboard using sales data
in this i want to analyse the which city , segment , product arev having high sales. 
Very intresting task among the 4 **
**Task-4: sentimental analysis**
1. Data Loading
Loaded a dataset of movie reviews using pandas.
Limited the dataset to 1000 reviews for fast processing
2. Sentiment Function
Defined a function using TextBlob to:
Analyze polarity of each review.
Classify review as:
Positive (polarity > 0)
Negative (polarity < 0)
Neutral (polarity = 0)
3. Sentiment Labeling
Applied the sentiment function to each review using .apply().
Added a new column Sentiment to the dataset.
4. Analysis & Visualization
Counted how many reviews fall under each sentiment.
Visualized the sentiment distribution using seaborn bar plot.
🔧 Tools & Libraries Used:
pandas — for data manipulation
textblob — for sentiment analysis
matplotlib & seaborn — for visualization

