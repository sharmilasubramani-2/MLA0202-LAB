# MLA 0202 - FUNDAMENTALS OF MACHINE LEARNING

**Course Code:** MLA 0202  
**Course Name:** Fundamentals of Machine Learning  
**Student Name:** Sharmila S  
**Register Number:** 192525011  

---

## TABLE OF CONTENTS

1. [Exp 01: Find-S Algorithm for Concept Learning](#exp-01-find-s-algorithm-for-concept-learning)
2. [Exp 02: Candidate Elimination Algorithm](#exp-02-candidate-elimination-algorithm)
3. [Exp 03: Decision Tree using ID3 Algorithm](#exp-03-decision-tree-using-id3-algorithm)
4. [Exp 04: Linear Regression using Gradient Descent](#exp-04-linear-regression-using-gradient-descent)
5. [Exp 05: Logistic Regression for Binary Classification](#exp-05-logistic-regression-for-binary-classification)
6. [Exp 06: Artificial Neural Network with Backpropagation](#exp-06-artificial-neural-network-with-backpropagation)
7. [Exp 07: Naïve Bayes Classifier for Document Classification](#exp-07-naïve-bayes-classifier-for-document-classification)
8. [Exp 08: Bayesian Network for Medical Diagnosis](#exp-08-bayesian-network-for-medical-diagnosis)
9. [Exp 09: K-Means Clustering Algorithm](#exp-09-k-means-clustering-algorithm)
10. [Exp 10: K-Nearest Neighbors (KNN) on Benchmark Dataset](#exp-10-k-nearest-neighbors-knn-on-benchmark-dataset)
11. [Exp 11: Credit Score Classification using Random Forest](#exp-11-credit-score-classification-using-random-forest)
12. [Exp 12: Iris Flower Classification using KNN](#exp-12-iris-flower-classification-using-knn)
13. [Exp 13: Car Price Prediction using Linear Regression](#exp-13-car-price-prediction-using-linear-regression)
14. [Exp 14: House Price Prediction using Linear Regression](#exp-14-house-price-prediction-using-linear-regression)
15. [Exp 15: Iris Flower Classification using Naïve Bayes](#exp-15-iris-flower-classification-using-naïve-bayes)
16. [Exp 16: Performance Comparison of Classification Algorithms](#exp-16-performance-comparison-of-classification-algorithms)
17. [Exp 17: Mobile Price Prediction using Random Forest](#exp-17-mobile-price-prediction-using-random-forest)
18. [Exp 18: Single-Layer Perceptron for Iris Classification](#exp-18-single-layer-perceptron-for-iris-classification)
19. [Exp 19: Bank Loan Eligibility Prediction using Naïve Bayes](#exp-19-bank-loan-eligibility-prediction-using-naïve-bayes)
20. [Exp 20: Future Sales Forecasting using Linear Regression](#exp-20-future-sales-forecasting-using-linear-regression)

---

### Exp 01: Find-S Algorithm for Concept Learning

**Aim:**  
To implement the Find-S algorithm in Python to determine the most specific hypothesis consistent with given positive training instances.

**Pseudocode:**
```text
1. Initialize hypothesis H = ['phi', 'phi', 'phi', 'phi', 'phi', 'phi']
2. For each training instance x with target value 'Yes':
     If H is all 'phi':
       Set H = list of feature values of x
     Else:
       For each feature index i:
         If H[i] != x[i]:
           H[i] = '?'
3. Return the maximally specific hypothesis H
Sample Input:PlaintextTraining Instances:
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No']
['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
Sample Output / Result:Plaintext=== FIND-S ALGORITHM ===
Maximally Specific Hypothesis:
['Sunny', 'Warm', '?', 'Strong', '?', '?']
SHARMILA S (192525011)
Exp 02: Candidate Elimination AlgorithmAim:To implement the Candidate Elimination algorithm to compute the General (G) and Specific (S) hypothesis boundaries for a given training dataset.Pseudocode:Plaintext1. Initialize S to the first positive instance; G to [['?', '?', '?', '?', '?', '?']].
2. For each training instance x:
     If x is positive:
       Prune from G any hypothesis inconsistent with x.
       Generalize S to satisfy x.
     If x is negative:
       Prune from S any hypothesis consistent with x.
       Specialize G to exclude x.
3. Return final version space boundaries S and G.
Sample Input:PlaintextPositive: ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
Positive: ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same']
Negative: ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change']
Positive: ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change']
Sample Output / Result:Plaintext=== CANDIDATE ELIMINATION RESULT ===
Specific Boundary (S):
[['Sunny', 'Warm', '?', 'Strong', '?', '?']]
General Boundary (G):
[['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?']]
SHARMILA S (192525011)
Exp 03: Decision Tree using ID3 AlgorithmAim:To construct a Decision Tree classifier using the ID3 algorithm based on Information Gain and Entropy metrics.Pseudocode:Plaintext1. Calculate Entropy of target class: H(S) = -Sum(p_i * log2(p_i)).
2. For each attribute A, compute Information Gain: Gain(S, A) = H(S) - Sum((|S_v|/|S|) * H(S_v)).
3. Select attribute with maximum Information Gain as root node.
4. Recursively repeat steps for subsets until all attributes are exhausted or nodes become pure.
Sample Input:PlaintextFeatures: [Outlook, Temperature, Humidity, Wind]
Target: PlayTennis [Yes, No] (14 rows)
Sample Output / Result:Plaintext=== DECISION TREE (ID3) CLASSIFIER ===
Root Node Selected: Outlook (Information Gain: 0.246)
Tree Construction: Completed
Classification Accuracy: 100.00%
SHARMILA S (192525011)
Exp 04: Linear Regression using Gradient DescentAim:To implement univariate Linear Regression and optimize model parameters using Gradient Descent to minimize Mean Squared Error.Pseudocode:Plaintext1. Initialize weight m = 0, bias c = 0, learning rate alpha = 0.01, epochs = 1000.
2. For each epoch:
     Compute y_pred = m * X + c.
     Compute error = y_pred - y.
     Update m = m - alpha * (2/n) * Sum(X * error).
     Update c = c - alpha * (2/n) * Sum(error).
3. Output final optimized parameters m and c.
Sample Input:PlaintextX = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
Sample Output / Result:Plaintext=== GRADIENT DESCENT LINEAR REGRESSION ===
Optimized Slope (m): 1.9998
Optimized Intercept (c): 0.0006
Final Mean Squared Error: 0.000001
SHARMILA S (192525011)
Exp 05: Logistic Regression for Binary ClassificationAim:To implement Logistic Regression using the Sigmoid activation function to perform binary classification on student performance data.Pseudocode:Plaintext1. Define Sigmoid function: sigma(z) = 1 / (1 + exp(-z)).
2. Compute linear equation: z = X * w + b.
3. Apply Sigmoid to obtain class probabilities: y_hat = sigma(z).
4. Update weights using Binary Cross-Entropy gradient descent.
5. Predict class 1 if y_hat >= 0.5, else class 0.
Sample Input:PlaintextHours Studied: [1.5, 2.0, 3.0, 4.5, 5.0, 6.0, 7.5, 8.0]
Exam Result (Pass/Fail): [0, 0, 0, 0, 1, 1, 1, 1]
Sample Output / Result:Plaintext=== LOGISTIC REGRESSION RESULT ===
Model Accuracy: 100.00%
Predicted Outcome for 5.5 Study Hours: Pass (Class 1)
SHARMILA S (192525011)
Exp 06: Artificial Neural Network with BackpropagationAim:To implement a Multi-Layer Perceptron (MLP) Neural Network using the Backpropagation algorithm and Sigmoid activation.Pseudocode:Plaintext1. Initialize random weights and biases for input-hidden and hidden-output layers.
2. Forward Propagation:
     h = sigmoid(X * W1 + b1)
     y_hat = sigmoid(h * W2 + b2)
3. Backward Propagation:
     delta_out = (y - y_hat) * y_hat * (1 - y_hat)
     delta_hid = (delta_out * W2.T) * h * (1 - h)
4. Update weights and biases via gradient descent step.
Sample Input:PlaintextInput Features: [[0, 0], [0, 1], [1, 0], [1, 1]] (XOR Logic)
Target: [0, 1, 1, 0]
Sample Output / Result:Plaintext=== BACKPROPAGATION ANN RESULT ===
Epochs: 10000 | Final Loss: 0.0021
Test Predictions:
[0, 0] -> 0.03 (Class 0)
[0, 1] -> 0.97 (Class 1)
[1, 0] -> 0.96 (Class 1)
[1, 1] -> 0.04 (Class 0)
SHARMILA S (192525011)
Exp 07: Naïve Bayes Classifier for Document ClassificationAim:To implement a Multinomial Naïve Bayes classifier with TF-IDF vectorization to categorize text documents into topic classes.Pseudocode:Plaintext1. Tokenize and preprocess training text corpus.
2. Build vocabulary matrix with term frequency / TF-IDF representations.
3. Compute Prior P(Class) and Likelihood P(Word | Class) with Laplace smoothing:
     P(w | c) = (count(w, c) + 1) / (total_words_c + |V|).
4. For test document, assign class maximizing log(P(c)) + Sum(log(P(w | c))).
Sample Input:PlaintextCorpus:
"Machine learning algorithms automate data analysis" -> Tech
"Government announces new fiscal tax policy" -> Politics
"Deep neural networks achieve high accuracy" -> Tech
Sample Output / Result:Plaintext=== NAÏVE BAYES DOCUMENT CLASSIFICATION ===
Test Document: "Neural networks process data efficiently"
Predicted Class: Tech
Model Accuracy: 100.00%
SHARMILA S (192525011)
Exp 08: Bayesian Network for Medical DiagnosisAim:To construct a Bayesian Directed Acyclic Graph (DAG) for probabilistic medical diagnostic inference under uncertainty.Pseudocode:Plaintext1. Define network structure with Directed Nodes (e.g., Smoking -> HeartDisease, Age -> HeartDisease).
2. Define Conditional Probability Tables (CPTs) for all variables.
3. Apply Variable Elimination algorithm to compute P(Disease | Evidence).
Sample Input:PlaintextQuery: P(HeartDisease = Yes | Smoking = Yes, Cholesterol = High)
Sample Output / Result:Plaintext=== BAYESIAN NETWORK INFERENCE ===
Posterior Probability Distribution:
P(HeartDisease = Yes) : 0.7420
P(HeartDisease = No)  : 0.2580
Inferred Diagnostic: High Risk
SHARMILA S (192525011)
Exp 09: K-Means Clustering AlgorithmAim:To implement the K-Means unsupervised clustering algorithm to partition unlabelled data into distinct clusters.Pseudocode:Plaintext1. Select K initial centroids randomly from the dataset.
2. Repeat until centroids converge:
     Assign each data point to the nearest centroid using Euclidean distance.
     Update centroids by computing the mean vector of all points in each cluster.
Sample Input:Plaintext2D Dataset: [[1.0, 1.5], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]]
K = 2
Sample Output / Result:Plaintext=== K-MEANS CLUSTERING RESULT ===
Cluster 0 Centroid: [1.16, 1.30]
Cluster 1 Centroid: [7.33, 9.00]
Silhouette Score: 0.7924
SHARMILA S (192525011)
Exp 10: K-Nearest Neighbors (KNN) on Benchmark DatasetAim:To implement K-Nearest Neighbors classification on a benchmark dataset and evaluate the effect of neighborhood parameter $K$.Pseudocode:Plaintext1. Load dataset and split into 70% Train, 30% Test sets.
2. For each query instance in Test set:
     Compute Euclidean distances to all training points.
     Sort distances and select top K nearest neighbors.
     Assign majority vote class among the K neighbors.
3. Compute test classification accuracy and confusion matrix.
Sample Input:PlaintextBenchmark Dataset: Breast Cancer Diagnostic / Iris
Neighborhood Parameter: K = 5
Sample Output / Result:Plaintext=== KNN CLASSIFIER RESULT ===
Optimal K-Value: 5
Test Accuracy: 96.67%
Confusion Matrix:
[[15  0]
 [ 1 14]]
SHARMILA S (192525011)
Exp 11: Credit Score Classification using Random ForestAim:To implement a Random Forest ensemble classifier to predict personal credit scores based on income and financial variables.Pseudocode:Plaintext1. Load dataset with features: Income, Credit_Score, Years_Employed.
2. Partition dataset into 75% training and 25% testing sets.
3. Normalize features using StandardScaler.
4. Train RandomForestClassifier(n_estimators=100).
5. Predict test samples and evaluate accuracy score.
Sample Input:PlaintextFeatures: Income = $45,000, Credit_Score = 710
Sample Output / Result:Plaintext=== CREDIT SCORE CLASSIFICATION RESULT ===
Model Accuracy: 100.00%
Prediction for Income $45,000 & Credit Score 710: Good
SHARMILA S (192525011)
Exp 12: Iris Flower Classification using KNNAim:To classify Iris flower species using the K-Nearest Neighbors (KNN) algorithm with Euclidean distance metric.Pseudocode:Plaintext1. Load standard Iris dataset (Sepal Length, Sepal Width, Petal Length, Petal Width).
2. Split data into 70% training and 30% testing subsets.
3. Instantiate and fit KNeighborsClassifier(n_neighbors=3).
4. Evaluate test accuracy and compute Confusion Matrix.
5. Render 2D feature scatter plot with distinct species colors.
Sample Input:PlaintextSample Features: [Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2]
Sample Output / Result:Plaintext=== IRIS CLASSIFICATION USING KNN RESULT ===
Accuracy Score: 100.00%
Predicted Species for sample [5.1, 3.5, 1.4, 0.2]: setosa
SHARMILA S (192525011)
Exp 13: Car Price Prediction using Linear RegressionAim:To implement Linear Regression to forecast automobile market prices (in ₹ Lakhs) based on model manufacturing years (2020–2026).Pseudocode:Plaintext1. Define dataset mapping manufacturing Year to Price in Lakhs (INR).
2. Fit LinearRegression() model on X=['Year'] and y=['Price_Lakhs'].
3. Calculate Mean Squared Error (MSE) and R2 goodness-of-fit score.
4. Forecast price for target year (e.g., Year 2027).
5. Plot scatter data points along with linear regression trendline.
Sample Input:PlaintextYear: [2020, 2021, 2022, 2023, 2024, 2025, 2026]
Price_Lakhs: [6.50, 7.20, 8.10, 9.00, 9.80, 10.75, 11.50]
Sample Output / Result:Plaintext=== CAR PRICE PREDICTION RESULT (2020 - 2026) ===
Mean Squared Error: 0.0031
R2 Score: 0.9989
Predicted Car Price for Year 2027: ₹12.41 Lakhs
SHARMILA S (192525011)
Exp 14: House Price Prediction using Linear RegressionAim:To implement Linear Regression to estimate residential property valuation based on built-up square footage using standard Indian currency format.Pseudocode:Plaintext1. Create real estate dataset: SquareFeet vs Price_Lakhs.
2. Train Linear Regression model on training features.
3. Compute regression slope, intercept, MSE, and R2 score.
4. Forecast price for 2000 SqFt property in ₹ Lakhs, Crores, and exact Rupees.
5. Render scatter plot with regression fit line.
Sample Input:PlaintextQuery Property Area: 2000 Square Feet
Sample Output / Result:Plaintext=== HOUSE PRICE PREDICTION RESULT ===
Mean Squared Error: 0.4617
R2 Score: 0.9995
Predicted House Price for 2000 SqFt: ₹104.35 Lakhs (₹1.04 Crores)
In Exact Rupees: ₹1,04,35,000
SHARMILA S (192525011)
Exp 15: Iris Flower Classification using Naïve BayesAim:To classify Iris flower species using Gaussian Naïve Bayes classifier based on continuous petal and sepal attributes.Pseudocode:Plaintext1. Load Iris features into Pandas DataFrame with explicit column headers.
2. Partition data into training and test sets (70:30).
3. Train GaussianNB() model.
4. Evaluate test accuracy and print Classification Report (Precision, Recall, F1-Score).
5. Generate Petal Length vs Petal Width scatter visualization.
Sample Input:PlaintextSample Features: [6.0, 3.0, 4.8, 1.8]
Sample Output / Result:Plaintext=== IRIS CLASSIFICATION USING NAIVE BAYES RESULT ===
Accuracy Score: 0.9778

Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        19
  versicolor       1.00      0.92      0.96        13
   virginica       0.93      1.00      0.96        13

    accuracy                           0.98        45
Predicted Species for sample [6.0, 3.0, 4.8, 1.8]: virginica
SHARMILA S (192525011)
Exp 16: Performance Comparison of Classification AlgorithmsAim:To evaluate and compare the classification accuracy of Logistic Regression, K-Nearest Neighbors, Gaussian Naïve Bayes, and Decision Tree on a benchmark dataset.Pseudocode:Plaintext1. Load Iris dataset and standardize features using StandardScaler.
2. Initialize classifier dictionary: LogisticRegression, KNN, Naïve Bayes, Decision Tree.
3. Iterate through models: fit on X_train_scaled, predict on X_test_scaled, record accuracy.
4. Render comparison bar chart annotated with percentage labels.
Sample Input:PlaintextFeature Data: 4 attributes across 150 instances (Iris Dataset)
Sample Output / Result:Plaintext=== CLASSIFICATION ALGORITHMS PERFORMANCE COMPARISON ===
Algorithm          | Accuracy (%)
----------------------------------
Logistic Reg       | 100.00%
KNN                | 100.00%
Naive Bayes        | 97.78%
Decision Tree      | 100.00%
SHARMILA S (192525011)
Exp 17: Mobile Price Prediction using Random ForestAim:To implement a Random Forest Classifier to categorize smartphones into Budget, Mid-Range, and Flagship tiers based on RAM and Battery specifications.Pseudocode:Plaintext1. Create dataset mapping RAM (GB) and Battery (mAh) to Price_Range.
2. Perform train-test split (70:30).
3. Fit RandomForestClassifier(n_estimators=50).
4. Predict price tier for custom configuration (e.g., 6GB RAM, 4500mAh).
5. Generate scatter plot showing hardware clusters with legends.
Sample Input:PlaintextRAM: 6 GB, Battery Capacity: 4500 mAh
Sample Output / Result:Plaintext=== MOBILE PRICE PREDICTION RESULT ===
Model Accuracy: 1.0
Predicted Bracket for 6GB RAM & 4500mAh Battery: Mid-Range
SHARMILA S (192525011)
Exp 18: Single-Layer Perceptron for Iris ClassificationAim:To implement a single-layer linear Perceptron algorithm to classify Iris flower species and evaluate linear decision boundaries.Pseudocode:Plaintext1. Load Iris dataset and perform feature standard scaling.
2. Initialize Perceptron(max_iter=1000, random_state=42).
3. Fit model on training partition.
4. Compute accuracy score and generate Classification Report.
5. Plot feature map of Sepal Length vs Petal Length.
Sample Input:PlaintextIris Features: Scaled sepal and petal dimensions
Sample Output / Result:Plaintext=== PERCEPTRON IRIS CLASSIFICATION RESULT ===
Accuracy Score: 0.8889

Classification Report:
              precision    recall  f1-score   support
      setosa       0.95      1.00      0.97        19
  versicolor       1.00      0.62      0.76        13
   virginica       0.76      1.00      0.87        13

    accuracy                           0.89        45
SHARMILA S (192525011)
Exp 19: Bank Loan Eligibility Prediction using Naïve BayesAim:To implement a Gaussian Naïve Bayes model to predict bank loan approval status based on applicants' annual income and requested loan amounts.Pseudocode:Plaintext1. Build applicant dataframe: Annual_Income_Lakhs, Loan_Amount_Lakhs, Loan_Status.
2. Perform stratified train-test split (70:30).
3. Train GaussianNB() classifier.
4. Calculate model accuracy and generate 2x2 confusion matrix.
5. Plot applicant loan eligibility distribution.
Sample Input:PlaintextApplicant Profile: Annual Income = ₹5.5 Lakhs, Loan Amount = ₹1.5 Lakhs
Sample Output / Result:Plaintext=== BANK LOAN PREDICTION (NAIVE BAYES) ===
Model Accuracy: 1.0

Confusion Matrix [Approved, Rejected]:
 [[2 0]
  [0 1]]
Loan Status for Income ₹5.5 Lakhs & Loan ₹1.5 Lakhs: Approved
SHARMILA S (192525011)
Exp 20: Future Sales Forecasting using Linear RegressionAim:To implement a Linear Regression model to forecast product sales volume based on marketing ad spend in ₹ Lakhs.Pseudocode:Plaintext1. Construct dataset relating Ad_Spend_Lakhs to Sales_Units.
2. Fit LinearRegression() model on features.
3. Compute regression performance metrics: Mean Squared Error (MSE) and R2 score.
4. Predict future sales volume for an ad budget of ₹11.0 Lakhs.
5. Plot data points alongside linear regression trendline.
Sample Input:PlaintextMarketing Ad Budget: ₹11.0 Lakhs
Sample Output / Result:Plaintext=== FUTURE SALES PREDICTION RESULT ===
Mean Squared Error: 63.9629
R2 Score: 0.9995
Predicted Sales for Ad Spend ₹11.0 Lakhs: 972 Units
SHARMILA S (192525011)
EXECUTION INSTRUCTIONSTo execute any experiment individually in your terminal, run:Bashpython exp<number>.py
Example:Bashpython exp16.py
