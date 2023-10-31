![titanic_custom-fc6a03aedd8e562d780ecf9b9a8a947d4dcbf163-s1100-c50](https://github.com/AnaPatSilva/Titanic-I/assets/92860743/659a3e7c-7481-4e62-af33-922b9b4b0b0e)
# Titanic-I
Do you know Kaggle? Do you know their competitions? Do you know the Titanic competition? This is my first submission!


## Intro - Who I am
After finishing my postgraduate degree in Analytics & Data Science I decided that I should continue practicing what I had learned, because I don't want to forget what I had learned, because practice makes perfect and because I like it! :-D
So I went to Kaggle and choose the **Titanic - Machine Learning from Disaster** Competition to start my practice (https://www.kaggle.com/competitions/titanic/overview).

## Titanic - Machine Learning from Disaster
The sinking of the Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

## First Attempt
To begin, I decided to create a machine learning model, using Random Forest, but with only one feature (Sex).

Steps:
1. Library Installation: The necessary libraries, such as pandas and scikit-learn, are installed.
2. Dataset Import: The Titanic datasets (gender_submission.csv, test.csv, and train.csv) are loaded.
3. Data Analysis: Data Profiling section intended to generate reports

Data Split and Feature (Sex) and Target/Label (Survived) Definition: The data is prepared with "Sex" as the feature and "Survived" as the target.

Training the Initial Random Forest Model: An initial Random Forest model is trained with pre-defined parameters.

Measuring the Initial Model's Accuracy: The model's accuracy is calculated, but it is not compared to the test set.

Cross-Validation: Cross-validation is used to evaluate model accuracy and prevent overfitting.

Tuning the Adjusted Random Forest Model: Model hyperparameters are tuned using both a random search and a grid search.

Training the Adjusted Random Forest Model: Two Random Forest models are trained with the best hyperparameters found in the Random Search and Grid Search.

Measuring Model Accuracy after Tuning: A ROC curve is plotted to assess the model's performance.

Confusion Matrix (commented out): The confusion matrix is mentioned but cannot be calculated due to a lack of test information.

Summary of Results: The average scores of Cross Folding, Repeated Cross Folding, Random Search, and Grid Search are displayed.

Model Output: The result of the adjusted model is saved in a CSV file.

This code provides a basic framework for predicting the deceased in the Titanic shipwreck based on passengers' gender. Keep in mind that the analysis and model evaluation can be extended to improve prediction accuracy.
