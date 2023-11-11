![titanic](https://github.com/AnaPatSilva/Titanic_I-Machine-Learning-Python/blob/main/titanic_custom-fc6a03aedd8e562d780ecf9b9a8a947d4dcbf163-s1100-c50.jpg)
# Titanic_I
Do you know Kaggle? Do you know their competitions? Do you know the Titanic competition? This is my first submission!


## Intro - Who I am
After finishing my postgraduate degree in Analytics & Data Science I decided that I should continue practicing what I had learned, because I don't want to forget what I had learned, because practice makes perfect and because I like it! :-D
So I went to Kaggle and choose the **Titanic - Machine Learning from Disaster** Competition to start my practice (https://www.kaggle.com/competitions/titanic/overview).


## Titanic - Machine Learning from Disaster
The sinking of the Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).


## First Attempt (One feature: “Sex”)
[Code (Python): First Attempt](https://github.com/AnaPatSilva/Titanic_I-Machine-Learning/blob/main/Code%20(Python)/Titanic%201f.py)

To begin with, I decided to create a machine learning model, using Random Forest, but with only one feature (Sex).


**Steps:**
1. **Library Installation:** The necessary libraries, such as pandas and scikit-learn, are installed.
2. **Dataset Import:** The Titanic datasets (gender_submission.csv, test.csv, and train.csv) are loaded.
3. **Data Analysis:** Data Profiling section intended to generate reports ([Profile Train.pdf](https://github.com/AnaPatSilva/Titanic-I/blob/main/Data%20Profiling/profile_train.pdf))
4. **Data Split and Feature (Sex) and Target/Label (Survived) Definition:** The data is prepared with "Sex" as the feature and "Survived" as the target.
5. **Training the Initial Random Forest Model:** An initial Random Forest model is trained with pre-defined parameters.
6. **Measuring the Initial Model's Accuracy:** The model's accuracy is calculated, but it is not compared to the test set.
7. **Cross-Validation:** Cross-validation is used to evaluate model accuracy and prevent overfitting.
8. **Tuning the Adjusted Random Forest Model:** Model hyperparameters are tuned using both a random search and a grid search.
9. **Training the Adjusted Random Forest Model:** Two Random Forest models are trained with the best hyperparameters found in the Random Search and Grid Search.
10. **Measuring Model Accuracy after Tuning:** A ROC curve is plotted to assess the model's performance.
11. **Confusion Matrix (commented out):** The confusion matrix is mentioned but cannot be calculated due to a lack of test information.
12. **Summary of Results:** The average scores of Cross Folding, Repeated Cross Folding, Random Search, and Grid Search are displayed.
13. **Model Output:** The result of the adjusted model is saved in a CSV file ([RF_GS (1f).csv](https://github.com/AnaPatSilva/Titanic-I/blob/main/Outputs/RF_GS%20(1f).csv)).

But the result didn't satisfy me because I think that only one feature won’t be sufficient to make a good prediction.


## Second Attempt (Three features: “Sex”, “Pclass”, “Fare”)
[Code (Python): Second Attempt](https://github.com/AnaPatSilva/Titanic-I/blob/main/Code%20(Python)/Titanic%203f.py)

This second attempt aims to enhance the Titanic dataset analysis by incorporating additional features (Sex, Pclass, Fare).

**Steps:**
1. **Library Installation:** The necessary libraries, such as pandas and scikit-learn, are installed.
2. **Dataset Import:** The Titanic datasets (gender_submission.csv, test.csv, and train.csv) are loaded.
3. **Data Analysis:** Data Profiling section intended to generate reports ([Profile Train.pdf](https://github.com/AnaPatSilva/Titanic-I/blob/main/Data%20Profiling/profile_train.pdf))
4. **Feature Selection:** Additional features - Sex, Pclass, and Fare - are chosen.
5. **Data Preprocessing:** The code checks for missing values in Pclass, Fare, and Sex in the training dataset. Any rows with missing Fare values are dropped, and a new dataset is created (test_na) without the dropped rows.
6. **Data Split and Feature (Sex, Pclass, Fare) and Target/Label (Survived) Definition:** The selected features are used for prediction, and one-hot encoding is applied to categorical variables.
7. **Training the Initial Random Forest Model:** An initial Random Forest model is trained with predefined parameters.
8. **Measuring the Initial Model's Accuracy:** Model accuracy is calculated but not compared to a test dataset.
9. **Cross-Validation to Avoid Overfitting:** Cross-validation with KFold and RepeatedKFold is used to assess the model's performance and prevent overfitting.
10. **Tuning the Initial Random Forest Model:** Random Search and Grid Search are performed to optimize the model's hyperparameters.
11. **Training the Adjusted Random Forest Models:** Two models are trained using the best hyperparameters obtained from Random Search and Grid Search.
12. **Measuring Model Accuracy after Tuning:** A ROC curve is plotted to evaluate the model's performance.
13. **Summary of Results:** The average scores from Cross Folding, Repeated Cross Folding, Random Search, and Grid Search are displayed.
14. **Model Output:** The results from the tuned models are saved in CSV files ([RF_GS (3f).csv](https://github.com/AnaPatSilva/Titanic-I/blob/main/Outputs/RF_GS%20(3f).csv)).

I tried to submit this model, but it wasn't accepted because I need to have 418 predictions and I only had 417. This happened because I decided to drop the lines with no values.


## Third Attempt (Three features: “Sex”, “Pclass”, “Fare”)
[Code (Python): Third Attempt](https://github.com/AnaPatSilva/Titanic-I/blob/main/Code%20(Python)/Titanic%203f%20(1).py)

On this third attempt I used the same features, but instead of drop the lines with no values, I made a Multiple Linear Regression model to determine the missing values.
In addition, I used the train dataset to made the split between train and test.

**Steps:**
1. **Library Installation:** The code begins by installing necessary libraries like pandas and scikit-learn.
2. **Dataset Import:** It loads the Titanic datasets (gender_submission.csv, test.csv, and train.csv).
3. **Data Analysis:** Data Profiling section intended to generate reports ([Profile Train.pdf](https://github.com/AnaPatSilva/Titanic-I/blob/main/Data%20Profiling/profile_train.pdf))
4. **Feature Selection:** The code selects three features for prediction: Sex, Pclass, and Fare.
5. **Data Preprocessing:** Missing values in the 'Pclass' and 'Fare' columns are checked in the training dataset.
6. **Data Split and Definition of Features and Target:** The dataset is split into training and testing sets (70% training, 30% testing), and one-hot encoding is applied to categorical variables.
7. **Training the Initial Random Forest Model:** An initial Random Forest model is trained with predefined parameters.
8. **Measuring the Initial Model's Accuracy:** The accuracy of the initial model is calculated using the testing dataset.
9. **Cross-Validation to Avoid Overfitting:** Cross-validation with KFold and RepeatedKFold is used to assess the model's performance and avoid overfitting.
10. **Tuning the Initial Random Forest Model:** Random Search and Grid Search are performed to optimize the model's hyperparameters.
11. **Training the Adjusted Random Forest Models:** Two models are trained using the best hyperparameters obtained from Random Search and Grid Search.
12. **Measuring Model Accuracy after Tuning:** The accuracy of both tuned models is calculated using the testing dataset, and a ROC curve is plotted for one of them.
13. **Confusion Matrix:** The code calculates and displays a confusion matrix to assess model performance further.
14. **Summary of Results:** The code provides a summary of the initial model's accuracy, cross-validation results, tuning results, and accuracy of the tuned models.
15. **Applying the Best Model to the Test Dataset:** The best model (Grid Search) is applied to the test dataset to make predictions.
16. **Replace the missing value:** Replace the missing values for the value given by the Linear Regression model.
17. **Model Output:** The final predictions are saved in a CSV file ([RF_GS(3f1).csv](https://github.com/AnaPatSilva/Titanic-I/blob/main/Outputs/RF_GS%20(3f1).csv)).


## Linear Regression (impute missing values)
[Code (Python): Linear Regression](https://github.com/AnaPatSilva/Titanic-I/blob/main/Code%20(Python)/Titanic%20Fare_NA.py)

For my third attempt I had to impute the missing values, so I decided to make a Linear Regression to impute those values.

**Steps:**
1. **Library Installation:** The code starts by installing necessary libraries, such as scikit-learn and seaborn.
2. **Dataset Import:** It loads the Titanic test dataset.
3. **Data Analysis:** Data Profiling section intended to generate reports ([Profile Train.pdf](https://github.com/AnaPatSilva/Titanic-I/blob/main/Data%20Profiling/profile_train.pdf))
4. **Feature Selection and One-Hot Encoding:** The code selects specific columns from the test dataset and applies one-hot encoding to the 'Sex' column.
5. **Correlation Analysis:** It calculates the correlation matrix and plots a heatmap to visualize the correlations between features. The code identifies features that are highly correlated with 'Fare' and lists the top correlated features.
6. **Handling Missing Values:** It identifies and prints the rows with missing values, which are only found in the 'Cabin' column.
7. **Feature Selection and Analysis:** The code selects the features 'Pclass', 'Age', and 'Parch' for further analysis. It provides the distribution of these features in the dataset and checks for missing values in each of them.
8. **Handling Missing Values in 'Age':** The 'Age' feature has many missing values. It considers replacing it with another feature, and after analysis, it selects 'Sex' as the replacement.
9. **Data Split and Definition of Features and Target:** It splits the data into two parts, one with complete values and one with missing values in the 'Fare' feature. Then, it defines the features and target for regression.
10. **Linear Regression for Imputation:** The code creates a linear regression model and fits it to the data with complete 'Fare' values. It uses this model to predict missing 'Fare' values based on the 'Pclass' feature.
11. **Imputation of Missing Values:** Missing 'Fare' values are imputed using the predictions from the linear regression model.
12. **Combining Imputed Data:** The code combines the data with imputed 'Fare' values with the data that had complete 'Fare' values.
13. **Visualizing Imputed 'Fare' Values:** It prints the 'Fare' values after imputation to verify the reasonableness of the imputed values.


## Conclusions
After all these experiments, I submitted the code from the third attempt (Random Forest), with the data obtained by the Linear Regression model to impute the null values.
My score was **0.77511** of 1.
It wasn’t bad but I will try to do better!
