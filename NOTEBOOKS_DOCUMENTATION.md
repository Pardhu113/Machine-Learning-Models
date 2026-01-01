# Jupyter Notebooks Documentation

This document provides detailed information about the Jupyter notebooks included in this Machine Learning Models repository.

## Overview

Both notebooks demonstrate practical implementations of supervised learning algorithms with real-world datasets, complete with data exploration, preprocessing, model training, and comprehensive evaluation.

---

## 1. Linear Regression with Insurance Data

**File**: `Linear_Regression.ipynb`

### Problem Statement
Predict insurance charges based on customer health and demographic features using Linear Regression.

### Dataset Information
- **Size**: ~11,000 records
- **Features**: 13 independent variables
- **Target**: Charges (continuous variable)
- **Features Include**:
  - Age, Sex, BMI (Body Mass Index)
  - Children, Smoker Status
  - Claim Amount, Hospital Expenditure
  - Past Consultations, Number of Steps
  - Annual Salary, Region

### Methodology

#### 1. Data Exploration & Cleaning
```
- Loaded dataset using pandas
- Checked for missing values
- Identified duplicates (None found)
- Analyzed data types and distributions
```

#### 2. Data Preprocessing
- **Missing Values**: Dropped rows with null values
- **Label Encoding**: Applied LabelEncoder to categorical features (sex, smoker, region)
- **Feature Engineering**: Separated features (X) and target (y)

#### 3. Model Development
- **Train-Test Split**: 80-20 split
  - Training set: 80% (~8,800 samples)
  - Testing set: 20% (~2,200 samples)
- **Algorithm**: Linear Regression from scikit-learn
- **Hyperparameters**:  
  - fit_intercept = True
  - n_jobs = None

#### 4. Model Evaluation
- **Primary Metric**: R² Score (Coefficient of Determination)
- **Formula**: R² = 1 - (SS_res / SS_tot)

### Results

**Performance Metric**: R² Score = **0.9937 (99.37%)**

**Interpretation**:
- The model explains 99.37% of the variance in insurance charges
- Excellent predictive performance
- Minimal error between predicted and actual values

### Key Insights
- Linear relationships exist between features and charges
- Feature scaling/normalization improved convergence
- Model suitable for charge prediction in insurance industry

---

## 2. Logistic Regression for Heart Disease Classification

**File**: `Logistic_regression.ipynb`

### Problem Statement
Predict the presence or absence of heart disease using patient health metrics (Binary Classification).

### Dataset Information
- **Size**: 303 patient records
- **Features**: 13 medical and demographic variables
- **Target**: Heart disease presence (0 = No, 1 = Yes)
- **Features Include**:
  - Age, Sex, CP (Chest Pain Type)
  - Trestbps (Resting Blood Pressure), Chol (Cholesterol)
  - FBS (Fasting Blood Sugar), Restecg (Resting ECG)
  - Thalach (Max Heart Rate), Exang (Exercise Induced Angina)
  - Oldpeak (ST depression), Slope, CA (Coronary Artery count)
  - Thal (Thalassemia type)

### Methodology

#### 1. Data Exploration
- Loaded and inspected dataset structure
- Checked for missing values (None found)
- Identified data types
- Analyzed target variable distribution

#### 2. Data Preprocessing
- **Missing Value Handling**: No missing values present
- **Duplicate Check**: No duplicates found
- **Feature-Target Separation**: X and y split

#### 3. Model Development
- **Train-Test Split**: 80-20 split
  - Training samples: 242
  - Testing samples: 61
- **Algorithm**: Logistic Regression from scikit-learn
- **Solver**: LBFGS (Limited-memory BFGS)
- **Hyperparameters**:
  - max_iter = 1000
  - random_state = 42

#### 4. Model Evaluation
Multiple evaluation metrics computed:

**Classification Metrics**:
1. **Accuracy**: Overall correctness of predictions
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: TP, TN, FP, FN breakdown

### Results

**Performance Metric**: Accuracy = **88.52%**

**Classification Report**:
- Correctly classified 88.52% of test samples
- High sensitivity and specificity
- Good balance between precision and recall

**Confusion Matrix Analysis**:
- True Positives (correctly identified disease)
- True Negatives (correctly identified no disease)
- False Positives & False Negatives minimized

### Key Insights
- Logistic regression effective for binary medical classification
- Model useful for preliminary heart disease screening
- Feature importance varies across different variables
- Potential for ensemble methods to improve performance

---

## Technical Stack

### Libraries Used
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning algorithms
- **Jupyter**: Interactive notebook environment

### Python Version
- Python 3.8 or higher

---

## Running the Notebooks

### Prerequisites
```bash
pip install -r requirements.txt
```

### Execution Steps
1. Install Jupyter: `pip install jupyter`
2. Navigate to repo directory
3. Start Jupyter: `jupyter notebook`
4. Open desired notebook file
5. Run cells sequentially (Shift + Enter)

---

## Potential Enhancements

### Linear Regression
- Polynomial feature creation
- Ridge/Lasso regularization
- Cross-validation for robustness
- Feature importance analysis

### Logistic Regression
- Hyperparameter tuning (C, penalty)
- Class weighting for imbalanced data
- ROC-AUC analysis
- Ensemble methods (Random Forest, Gradient Boosting)

---

## Conclusion

Both projects demonstrate successful implementation of supervised learning algorithms with proper data handling, model training, and evaluation. The notebooks serve as practical references for similar ML tasks in regression and classification domains.
