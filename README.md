# Bridge Damage Prediction
Code  Contribution for Bridge Damage Prediction Project

## Notebook Summary

This notebook performs a comprehensive analysis to predict bridge conditions ('Good', 'Fair', 'Poor') using various machine learning models. The workflow includes data loading, preprocessing, model training, evaluation, and comparison.

### 1. Data Loading and Preprocessing

* The bridge_data_final-1.csv file was loaded into a Spark DataFrame and null values were dropped.
* Only bridges with 'Good', 'Fair', or 'Poor' conditions were retained.
* Categorical features Bridge_Condition and Main_Span_Material were converted to numerical representations using StringIndexer, creating label and Main_Span_Material_Index columns, respectively.
* Bridge_Age, Average_Daily_Traffic, and Deck_Area_sqft were cast to DoubleType.
* A feature vector was assembled using Bridge_Age, Average_Daily_Traffic, Deck_Area_sqft, and Main_Span_Material_Index.
* The data was split into training (80%) and testing (20%) sets.

### 2. Model Training and Evaluation

Four classification models were trained and evaluated:

**a. Random Forest Classifier**
* Accuracy: 0.71
* F1-Score: 0.69

**Key Finding:** 
* Feature importances showed Bridge_Age as the most significant predictor (0.7953), followed by Main_Span_Material_Index (0.1561).
* State-wise Accuracy: Virginia (0.768), New York (0.703), Texas (0.701).
* Material-wise Accuracy: Masonry (0.867), Other Material (0.857).

**b. Naive Bayes Classifier**
* Accuracy: 0.24
* F1-Score: 0.27

**Observation:**

This model performed significantly worse than Random Forest and Decision Tree, indicated by low precision, recall, and F1-scores for all classes, particularly 'Good' and 'Poor'. The confusion matrix highlighted issues with misclassification, often predicting 'Fair' or 'Poor' when the true label was 'Good'.

State-wise Accuracy: New York (0.299), Virginia (0.294), Texas (0.213).

**c. Decision Tree Classifier**
* Accuracy: 0.72
* F1-Score: 0.70

**Key Finding:**
* Similar to Random Forest, Bridge_Age was the most important feature (0.8594), followed by Main_Span_Material_Index (0.1156).
* State-wise Accuracy: Virginia (0.768), New York (0.706), Texas (0.705).
* Material-wise Accuracy: Masonry (0.867), Other Material (0.857).

**d. K-Nearest Neighbors (KNN) Classifier**
* Preprocessing: Features were scaled using StandardScaler to prevent larger values from dominating the distance calculations.
* Accuracy: 0.69
* F1-Score: 0.50 (macro average)

**Observation:**

KNN showed moderate performance, with a relatively low F1-score for the 'Poor' class (0.09).

State-wise Accuracy: Virginia (0.700), Texas (0.694), New York (0.683).

### 3. Model Comparison

* Accuracy Comparison: Decision Tree (0.72) and Random Forest (0.71) performed similarly and were the best models, followed by KNN (0.69) and Naive Bayes (0.24).
* F1-Score Comparison: Decision Tree (0.70) and Random Forest (0.69) also had the highest F1-scores, significantly outperforming KNN (0.50) and Naive Bayes (0.27).
  
### 4. Correlation Analysis

A correlation matrix of numerical features (Bridge_Age, Deck_Area_sqft, Average_Daily_Traffic) showed:
Bridge_Age has a very weak negative correlation with Deck_Area_sqft (-0.10) and Average_Daily_Traffic (-0.05).
Deck_Area_sqft and Average_Daily_Traffic have a moderate positive correlation (0.26).

# Conclusion

Based on accuracy and F1-score, Random Forest and Decision Tree models are the most effective for predicting bridge conditions in this dataset. 
Bridge_Age consistently emerges as the most influential feature across these models. 
Naive Bayes performed poorly, likely due to its strong assumption of feature independence not holding true for this dataset. KNN showed acceptable but not leading performance. Future work could involve hyperparameter tuning for the top-performing models or exploring more advanced ensemble methods.


