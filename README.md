# Sonar Rock vs Mine Detection using Logistic Regression

A machine learning project that uses sonar data to classify underwater objects as either rocks (R) or mines (M) using Logistic Regression.

## 🎯 Project Overview

This project implements a binary classification system to distinguish between rocks and mines using sonar signal data. The model analyzes 60 different sonar signal features to make accurate predictions.

## 📊 Dataset Information

- **Total Samples**: 208 sonar readings
- **Features**: 60 numerical features (sonar signal frequencies)
- **Target Classes**: 
  - R (Rock): 97 samples
  - M (Mine): 111 samples
- **Data Type**: Continuous numerical values representing sonar signal returns

## 🔄 Project Workflow

https://github.com/somethingshawnx/Sonar_Detection/blob/ddb23007fd021759a301b56120e0206416d3ce51/flowchart.png

```
Sonar Data → Data Preprocessing → Train-Test Split → Logistic Regression Model
                                                              ↓
                    Trained Logistic Regression Model
                              ↙              ↘
                        New Data          Prediction
                                         (Rock or Mine)
```

### Workflow Steps:

1. **Data Collection**: Load sonar dataset with 60 features + 1 target column
2. **Data Preprocessing**: 
   - Analyze data distribution
   - Separate features (X) and labels (Y)
3. **Train-Test Split**: 90% training, 10% testing with stratification
4. **Model Training**: Logistic Regression implementation
5. **Model Evaluation**: Calculate accuracy scores
6. **Prediction System**: Real-time classification of new sonar data

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning library
  - `train_test_split` - Data splitting
  - `LogisticRegression` - Classification algorithm
  - `accuracy_score` - Model evaluation

## 📁 Project Structure

```
sonar-detection/
│
├── sonar data.csv          # Dataset file
├── sonarproject.ipynb      # Main Jupyter notebook
└── README.md              # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn jupyter
```

### Running the Project

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sonar-rock-mine-detection.git
cd sonar-rock-mine-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Open Jupyter Notebook**
```bash
jupyter notebook sonarproject.ipynb
```

4. **Run all cells** to execute the complete pipeline

## 📈 Model Performance

- **Training Accuracy**: 83.42%
- **Testing Accuracy**: 76.19%

The model shows reasonable performance with slight overfitting, which is common for small datasets.

## 🔍 Key Features

### Data Analysis
- Exploratory data analysis of sonar features
- Class distribution visualization (M: 111, R: 97)
- Statistical summary of feature values

### Model Implementation
- **Algorithm**: Logistic Regression
- **Train-Test Split**: 90-10 with stratification
- **Random State**: 1 (for reproducibility)

### Prediction System
The model can classify new sonar readings:
```python
# Example prediction
input_data = (0.0228,0.0106,0.0130,0.0842,...) # 60 features
prediction = model.predict(input_data_reshaped)
# Output: 'R' for Rock or 'M' for Mine
```

## 📊 Sample Results

```
Input: New sonar signal data (60 features)
Output: "The object is Mine" or "The object is Rock"
```

## 🔧 Code Highlights

### Data Preprocessing
```python
# Separate features and target
x = df.drop(columns=60, axis=1)  # Features (60 columns)
y = df[60]                       # Target (Rock/Mine)
```

### Model Training
```python
# Train-test split with stratification
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, stratify=y, random_state=1
)

# Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)
```

### Prediction System
```python
# Reshape input for prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
```

## 🎯 Future Enhancements

- [ ] Implement cross-validation for better model evaluation
- [ ] Try other algorithms (SVM, Random Forest, Neural Networks)
- [ ] Add feature selection techniques
- [ ] Implement hyperparameter tuning
- [ ] Create a web interface for real-time predictions
- [ ] Add data visualization for better insights

## 📝 Lessons Learned

1. **Data Quality**: The sonar dataset provides good separation between classes
2. **Model Selection**: Logistic Regression works well for binary classification
3. **Overfitting**: Small dataset size leads to some overfitting
4. **Feature Importance**: All 60 features contribute to the classification



