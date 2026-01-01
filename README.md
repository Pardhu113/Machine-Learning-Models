# Machine Learning Models

A comprehensive collection of machine learning models and data science projects implementing regression, classification, and advanced ML algorithms with complete documentation and analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results & Performance](#results--performance)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Contact](#contact)

## Overview

This repository contains practical implementations of machine learning algorithms using real-world datasets. Each project includes comprehensive data analysis, model development, evaluation metrics, and visualization of results.

## 📁 Projects

### 1. **Linear Regression Analysis**
- **File**: `Linear_Regression.ipynb`
- **Dataset**: Insurance/Medical Data
- **Objective**: Predict charges based on various health and demographic features
- **Key Features**:
  - Data preprocessing and exploratory data analysis (EDA)
  - Label encoding for categorical variables
  - Feature scaling and normalization
  - Linear regression model implementation
  - Model performance evaluation with R² score
- **Performance**: R² Score: 0.9937 (99.37% accuracy)

### 2. **Logistic Regression - Binary Classification**
- **File**: `Logistic_regression.ipynb`
- **Dataset**: Heart Disease Classification Dataset
- **Objective**: Predict presence/absence of heart disease using patient health metrics
- **Key Features**:
  - Binary classification problem
  - Data exploration and missing value handling
  - Logistic regression model with optimized parameters
  - Confusion matrix analysis
  - Classification metrics (accuracy, precision, recall, F1-score)
- **Performance**: Accuracy: 88.52%

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Pardhu113/Machine-Learning-Models.git
cd Machine-Learning-Models

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 📦 Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 📂 Project Structure

```
Machine-Learning-Models/
├── Linear_Regression.ipynb
├── Logistic_regression.ipynb
├── README.md
├── LICENSE
└── requirements.txt
```

## 💻 Usage

### Running Jupyter Notebooks

```bash
# Start Jupyter notebook
jupyter notebook

# Open desired notebook
# - Linear_Regression.ipynb
# - Logistic_regression.ipynb
```

### Step-by-step Execution

1. **Load & Explore Data**: Understand data structure and distributions
2. **Data Preprocessing**: Handle missing values, encode categorical variables
3. **Feature Engineering**: Select relevant features, scale data
4. **Model Training**: Train ML algorithms on training dataset
5. **Model Evaluation**: Evaluate performance using appropriate metrics
6. **Visualization**: Analyze results with plots and statistical summaries

## 📊 Results & Performance

### Linear Regression
- **R² Score**: 0.9937 (99.37%)
- **Use Case**: Continuous value prediction (medical charges)

### Logistic Regression
- **Accuracy**: 88.52%
- **Use Case**: Binary classification (disease presence prediction)

## 🔧 Technologies Used

- **Python**: Programming language
- **Scikit-learn**: ML algorithms and tools
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter**: Interactive computing environment

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Author**: Pardhu113  
**GitHub**: [@Pardhu113](https://github.com/Pardhu113)  
**Email**: Available on GitHub profile

---

### 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

### ⭐ Show Your Support

Give a ⭐ if you found this repository helpful!
