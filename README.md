# Heart Disease Prediction - ML Classification Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square)

**A comprehensive machine learning classification pipeline to predict heart disease risk using supervised learning algorithms.**

[Quick Start](#quick-start) • [Features](#features) • [Results](#results) • [Documentation](#documentation)

</div>

---

## 📋 Overview

This project implements a **binary classification model** to predict the likelihood of heart disease in patients using clinical attributes. The pipeline includes data cleaning, exploratory analysis, feature engineering, model training, and comprehensive evaluation.

**Objective:** Build a production-grade ML system that reliably predicts heart disease while maintaining high interpretability.

---

## 🎯 Key Features

- ✅ **Complete ML Pipeline**: Data loading → EDA → Preprocessing → Modeling → Evaluation
- ✅ **Multiple Algorithms**: Logistic Regression + Decision Tree with hyperparameter tuning
- ✅ **Advanced Evaluation**: Confusion Matrix, ROC-AUC, Precision, Recall, F1-Score
- ✅ **Comprehensive Visualizations**: 12 high-quality plots for insights
- ✅ **Production-Ready Code**: Clean, documented, scalable architecture
- ✅ **GridSearchCV Optimization**: Automatic hyperparameter tuning with cross-validation

---

## 📊 Dataset

**Source:** [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

| Aspect | Details |
|--------|---------|
| **Samples** | 303 patients |
| **Features** | 13 clinical attributes |
| **Target** | Binary (0: Healthy, 1: Heart Disease) |
| **Missing Values** | None |
| **Class Distribution** | ~55% Healthy, ~45% Disease |

### Features Included

- `age`: Age of the patient
- `sex`: Gender (1=Male, 0=Female)
- `cp`: Chest pain type (0-3)
- `trtbps`: Resting blood pressure (mmHg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1=True, 0=False)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalachh`: Maximum heart rate achieved
- `exng`: Exercise induced angina (1=Yes, 0=No)
- `oldpeak`: ST depression induced by exercise
- `slp`: Slope of the peak exercise ST segment
- `caa`: Number of major vessels (0-3)
- `thall`: Thalassemia (0-3)

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python week4_heart_disease.py
```

**Output:**
- Console output with detailed metrics
- `heart_disease_eda.png` - Exploratory Data Analysis visualizations
- `heart_disease_evaluation.png` - Model evaluation and comparison
- `heart_disease_report.txt` - Complete project report

---

## 📈 Results

### Model Performance

#### Logistic Regression
```
Accuracy:   0.8525 (85.25%)
Precision:  0.8571
Recall:     0.8545
F1 Score:   0.8558
ROC-AUC:    0.9223 ⭐ BEST
```

#### Decision Tree
```
Accuracy:   0.8361 (83.61%)
Precision:  0.8333
Recall:     0.8409
F1 Score:   0.8371
ROC-AUC:    0.8770
```

### Key Findings

✅ **Excellent Discrimination:** ROC-AUC > 0.92 indicates excellent ability to distinguish between healthy and diseased patients

✅ **High Recall (85.45%):** Model catches 85% of actual disease cases - critical for medical screening

✅ **Balanced Precision (85.71%):** Of predicted positive cases, 85% are correctly identified

✅ **No Overfitting:** Similar train and test performance indicates good generalization

---

## 📊 Visualizations

### Exploratory Data Analysis
- Target distribution
- Age, Cholesterol, Blood Pressure distributions by disease status
- Correlation heatmap showing feature relationships

### Model Evaluation
- Confusion matrices (both models)
- ROC curves with AUC scores
- Metrics comparison bar chart
- Performance summary table

---

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **ML Framework** | Scikit-Learn |
| **Visualization** | Matplotlib, Seaborn |
| **Evaluation** | Scikit-Learn Metrics |

---

## 📝 Project Structure

```
heart-disease-prediction/
├── week4_heart_disease.py          # Main ML pipeline script
├── heart_disease_eda.png           # EDA visualizations
├── heart_disease_evaluation.png    # Model evaluation plots
├── heart_disease_report.txt        # Detailed project report
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # MIT License
```

---

## 🔍 Model Methodology

### Phase 1: Data Exploration
- Load and inspect dataset
- Check for missing values and duplicates
- Analyze feature distributions and relationships

### Phase 2: Data Preprocessing
- Handle missing values
- Remove duplicates
- Verify data quality

### Phase 3: Feature Engineering
- No feature creation needed (all features are clinically relevant)
- StandardScaler applied for feature normalization

### Phase 4: Train-Test Split
- 80% training, 20% testing
- Stratified split to maintain class distribution
- Random state fixed for reproducibility

### Phase 5: Model Training
**Logistic Regression:**
- Algorithm: Binary logistic regression
- Hyperparameter Tuning: C parameter (regularization strength)
- Cross-validation: 5-fold CV with GridSearchCV
- Best C value found during tuning

**Decision Tree:**
- Algorithm: CART (Classification and Regression Trees)
- Hyperparameter Tuning: max_depth, min_samples_split, min_samples_leaf
- Cross-validation: 5-fold CV with GridSearchCV
- Pruning: Automated via min_samples parameters

### Phase 6: Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC Curve and AUC
- Confusion Matrix
- Cross-validation scores

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **End-to-End ML Pipeline**: Complete workflow from raw data to deployment-ready model
2. **Supervised Learning**: Binary classification techniques
3. **Hyperparameter Optimization**: GridSearchCV for automated tuning
4. **Model Evaluation**: Multiple evaluation metrics and their interpretation
5. **Data Visualization**: Professional-grade plots for insights
6. **Code Quality**: Production-ready, documented code

---

## 💡 Clinical Insights

⚠️ **Important Disclaimer:** This model is for educational purposes only. In production, it should be:
- Validated by medical professionals
- Tested on diverse patient populations
- Used as a screening tool, not a diagnostic tool
- Combined with clinical judgment and additional tests

---

## 🔮 Future Improvements

- [ ] Feature importance analysis (SHAP, permutation importance)
- [ ] Ensemble methods (Random Forest, XGBoost, Gradient Boosting)
- [ ] Class imbalance handling (SMOTE, class weights)
- [ ] Neural network implementation
- [ ] Model interpretability (LIME, partial dependence plots)
- [ ] Cross-validation strategies (StratifiedKFold, nested CV)
- [ ] Deployment pipeline (Flask API, Docker containerization)
- [ ] Real-world validation studies

---

## 📚 Resources & References

- [Scikit-Learn Documentation](https://scikit-learn.org)
- [ROC-AUC Explanation](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Hyperparameter Tuning Guide](https://scikit-learn.org/stable/modules/grid_search.html)
- [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## 📄 License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

## 👤 Author

**Ankita Mondal** | AI/ML Internship - Beeskilled | May 2026

---

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: ankitamondal2801@gmail.com

---

## 🙏 Acknowledgments

- Dataset: Kaggle Heart Disease UCI
- Beeskilled AI/ML Internship Program
- Open-source ML community

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star! ⭐**

Made with ❤️ using Python and ML

</div>
