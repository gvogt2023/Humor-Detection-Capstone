# Humor-Detection-Capstone
Humor-Detection-Capstone for Flatiron School



## Business and Data Understanding:

Stakeholders:
- The stakeholders of the humor detection project may include social media platforms, advertisers, or content creators who want to gauge the effectiveness of their content in eliciting humor.
- For social media platforms, being able to detect humorous content can improve the user experience by prioritizing and promoting content that generates positive emotions.
- Advertisers may want to measure the impact of humorous ads on brand recall and engagement.
- Content creators may want to evaluate the effectiveness of their humor and adjust it accordingly to maximize its impact.

Business Objectives:
- The objective of the humor detection project is to develop a machine learning model that can accurately detect humor in textual content.
- The model should be trained on a large dataset of labeled humorous and non-humorous content.
- The resulting model can then be used to classify new content as either humorous or non-humorous, allowing stakeholders to measure the effectiveness of their content in eliciting humor.

## Data understanding:
- In a [2004 paper](https://arxiv.org/abs/2004.12765), the authors Issa Annamoradnejad, Gohar Zoghi provided on a dataset for humor detection consisting of 200,000 formal short texts. Each entry has two columns: one for the raw text of the joke and the other for target binary variable 'humor' labled true or false.
- Example: "What is a pokemon master's favorite kind of pasta? wartortellini!!" is 'true' for 'humor'
- The non-humorous texts are short headlines from online news sources
- Example: "Kim kardashian baby name: reality star discusses the 'k' name possibility" is 'false' for 'humor'


![Correlation Matrix](imgs/correlation_matrix.png)
High negative correlations:
- Income
- Education
- PhysActivity

Highest positive correlations:
- GenHlth
- HighBP
- DiffWalk
- BMI

![Target Distribution](imgs/diabetes_target_distribution.png)

Yes, there is a class imbalance. Class 0 has a significantly larger number of samples compared to classes 1 and 2. We will undersample the major and also utilize SMOTE later.

![Histogram All](imgs/histogram_all_features.png) 

![BMI Distribution](imgs/BMI_distribution.png)

BMI is skewed to the for diabetic patients.

![Age Distribution](imgs/age_distribution.png) 

Age is also skewed to the for diabetic patients.

Due to the size of the dataset and time for processing, we are undersampling the majority class 0 and keeping all rows for the minority class. Randomly selecting the matching length of minority class from majority dataframe.

## Universal Sentence Encoder

```


## Topic Modeling

Given a user's input, encode it into a vector representation using the same Universal Sentence Encoder model.
user_input = input("Enter a sentence: ")

## Classification Modeling

During this iteration process for the multiclass problem of predicting for 0 (non-diabetic), 1 (prediabetic) or 2 (diabetic), I processed the following models:
- Basic logistic regression will be our basline model. We will score for macro recall because of multiclass classification and wanting to account for positive instances of our target for our minority class '1'.
- Add a parameter grid search for logistic regression with SMOTE for minority and standard scaler to improve the model.
- Utilize KBest features for tuned logistic regression model.
- RandomForest classifier with randomsearch, standard scaler and SMOTE in pipeline.
- Utilize a bagging model with DecisionTree as my base estimator and grid search for optimal params.
- Utilize KNN model with SelectKBest features and grid search.
- Utilize AdaBoostClassifier with standard scaler and gridsearch
- Utilize GradientBoost classifier with SMOTE, standard scaler for numeric columns and gridsearch.
- Stacking model using xgb, bagging, randomforest and adaboost pipelines. Utilize parameters that were optimized during earlier models.

Then when I shifted to the binary class problem of predicting for 0 (non-diabetic) or 1 (prediabetic or diabetic), I processed the same iterations of models:
- Basic logistic regression will be our basline model. 
- Add a parameter grid search for logistic regression with SMOTE for minority and standard scaler to improve the model.
- Utilize KBest features for tuned logistic regression model.
- RandomForest classifier with randomsearch and standard scaler in pipeline.
- Utilize a bagging model with DecisionTree as my base estimator and grid search for optimal params.
- Utilize KNN model with SelectKBest features and grid search.
- Utilize AdaBoostClassifier with standard scaler and gridsearch
- Utilize GradientBoost classifier with standard scaler for numeric columns and gridsearch.
- Stacking model using xgb, bagging, randomforest and adaboost pipelines. Utilize parameters that were optimized during earlier models.

## Evaluation

Multiclass results:

![Multiclass Results](imgs/multiclass_results.png)

Based on the classification reports, it appears that the basic logistic regression, bagging, and knn models all performed similarly with an accuracy score of around 70%. However, these models fail to predict for the minority class '1' at all while tuned logistic regression, logistic kbest, random forest, and gradient boost models all had lower accuracy scores but did successfully predict for value '1' of target.

Overall, it seems that these techniques were not very successful in the multiclass approach. Bagging and knn models may be the best choices for this particular dataset due to highest accuracy scores and precision and recall scores above 0.7 for all three classes, however, it's worth noting the next notebook for Binary approach improves on this multiclass approach.

![Feature Importances](imgs/feature_importances_final_model.png) age_distribution.png

BMI, age and general health are the most important features of our multiclass model. 

Binary results:

![Binary Results](imgs/binary_final_results.png) 

All of these models seem to have similar performance with an overall accuracy of around 75%. This is a strong improvement over the multiclass models from the first notebook. The precision and recall scores for the two classes (0 and 1) are also relatively similar across the models. However, there are slight variations in the macro and weighted F1 scores.

The logistic regression model with and without selectKBest feature selection has the lowest accuracy and F1 scores. This could indicate that the model is underfitting the data and may not be capturing the complexity of the relationships between the features and the target variable. The other models, including RandomForest, Bagging, KNN, XGBoost, AdaBoost, and Stacking models seem to be performing similarly well, with only slight variations in their precision, recall, and F1 scores.

I will select AdaBoost as our final model in proposal.

![Binary Feature Importance](imgs/feature_importances_binary.png) 

BMI, age, and general health remain the most important factors in this classification model. 

## Conclusion and Recommendations

Takeaways:
- Predicting diabetes with phone screening is not very reliable
- Binary model performs better than multi-class and still applies to our business case
- Model can still be useful for a generalization of risk
- Further improve model by tuning hyper-parameters, try more models, polynomial feature engineering etc.
- Look in the future to add additional biometric data to strengthen predictions
- Prioritize feedback on BMI, age and general health on future surveys as these have the strongest
