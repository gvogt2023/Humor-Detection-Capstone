# Humor-Detection-Capstone
Humor-Detection-Capstone for Flatiron School



## Business Understanding:

Stakeholders:
- The stakeholders of the humor detection project may include social media platforms, advertisers, or content creators who want to gauge the effectiveness of their content in eliciting humor.
- For social media platforms, being able to detect humorous content can improve the user experience by prioritizing and promoting content that generates positive emotions.
- Advertisers may want to measure the impact of humorous ads on brand recall and engagement.
- Content creators may want to evaluate the effectiveness of their humor and adjust it accordingly to maximize its impact.

Business Objectives:
- The objective of the humor detection project is to develop a machine learning model that can accurately detect humor in textual content.
- The model should be trained on a large dataset of labeled humorous and non-humorous content.
- The resulting model can then be used to classify new content as either humorous or non-humorous, allowing stakeholders to measure the effectiveness of their content in eliciting humor.

## Data understanding
- In a [2004 paper](https://arxiv.org/abs/2004.12765), the authors Issa Annamoradnejad, Gohar Zoghi provided on a dataset for humor detection consisting of 200,000 formal short texts. 
- This data is contained in the file 'jokes_dataset.csv' in the data folder.
- Each entry has two columns: one for the raw text of the joke and the other for target binary variable 'humor' labled true or false.  Example: "What is a pokemon master's favorite kind of pasta? wartortellini!!" is 'true' for 'humor'.
- The non-humorous texts are short headlines from online news sources. Example: "Kim kardashian baby name: reality star discusses the 'k' name possibility" is 'false' for 'humor'.

## Exploratory Data Analysis

The dataset is balanced in regards to its target variable 'humor.'

![Target Distribution](imgs/target_distribution.png)

Following removal of stopwords and punctuatino, the length of the values in 'text' column follow a normal distribution and have a mean of 47.84 characters with a minimum of 4 and maximum of 95 characters.

![Joke Length Distribution](imgs/jokes_length_distribution.png)

The distribution is similar for jokes vs. non-jokes.

![Joke Length Distribution by Type](imgs/jokelength_distribution_type.png)

A word cloud was generated to visualize the most common words in a dataset of jokes and non-jokes.

![Wordcloud Jokes](imgs/wordcloud_jokes.png)
![Wordcloud Non-Jokes](imgs/wouldcloud_nonjokes.png)

# Universal Sentence Encoder

I used a pre-trained Universal Sentence Encoder model from TensorFlow Hub. The encoded vectors are stored in a list called joke_vectors_list. To encode the jokes in batches, a for loop is used that iterates over the jokes in steps of the batches required due to the size of the dataset. The model object is called on each batch of jokes, and the resulting encoded vectors are appended to the joke_vectors_list list.

Next, cosine similarity is calculated between all pairs of joke vectors using the cosine_similarity function from sklearn.metrics.pairwise. Since the joke vectors array may be too large to calculate cosine similarity between all pairs of vectors at once, the calculation is done in batches. The final similarity_matrix contains cosine similarity values for all pairs of joke vectors.

A user will be prompted to enter a string to encode and assess cosine similarity to the list of jokes and return the most similar rows.

Given a user's input, encode it into a vector representation using the same Universal Sentence Encoder model.

```
user_input = input("Enter a sentence: ")

```
Get the number of jokes to recommend from the user

```
num_jokes = int(input("How many jokes would you like to see? "))
```

Print the top-n jokes

```
print(f"Top {num_jokes} jokes:")
for i, joke in enumerate(top_jokes):
    print(f"{i+1}. {joke.text}")
```

I enter the phrase "Dr. Seuss cat in the hat" and enter "5" for number of results to show.

```
Top 5 jokes:
1. What did dr. seuss call the book he wrote about star wars? the cat in the at-at
2. What was schrodinger's favorite childhood book? the cat in the box by dr. seuss
3. What is dr. seuss' favorite play? green eggs and hamlet
4. Did you read dr seuss as a kid because green eggs and damn
5. What do you call a magician in a dr. seuss book? who-dini
```

## Topic modeling

LDA is used as a topic modeling technique to discover latent topics in a collection of documents. It models each topic as a probability distribution over the words in the vocabulary to explain the topic-word distributions.

For topic modeling with LDA, lemmatization, tokenization and vectorization are used to prepare the text data for input into the model. TfidfVectorizer is used to tokenize the lemmatized text data, and an LDA model is created and trained on the tokenized data. I analyze the topic structure of the jokes in the dataset and print the top words for each topic using get feature names.

```
feature_names = vectorizer.get_feature_names_out()

for i in range(num_topics):
    topic_words = ' '.join([feature_names[idx] for idx in np.argsort(lda_model.components_[i])[:-11:-1]])
    print(f"Topic {i}:\n{topic_words}\n")

```
```
Topic 0:
  joke
  mexican
  whats
  make
  say
  trump
  change
  im
  like
  want

Topic 1:
  like
  joke
  chicken
  whats
  say
  im
  know
  people
  tell
  woman

Topic 2:
  walk
  bar
  say
  man
  im
  whats
  people
  like
  guy
  got

Topic 3:
  like
  say
  whats
  woman
  black
  know
  im
  people
  make
  coffee

Topic 4:
  im
  like
  people
  knock
  know
  hear
  girl
  time
  whats
  say
```

The function "get_top_tokens" is defined to extract the most important words for each topic, along with their corresponding weights, for the 5 defined topics. The feature names for the vectorizer used to process the input documents are obtained using the "get_feature_names_out" method. Finally, a bar plot is generated for each topic, displaying the top tokens and their weights.

![Topic 1 Weights](imgs/topic_1_token_weights.png)
![Topic 2 Weights](imgs/topic_2_token_weights.png)
![Topic 3 Weights](imgs/topic_3_token_weights.png)
![Topic 4 Weights](imgs/topic_4_token_weights.png)
![Topic 5 Weights](imgs/topic_5_token_weights.png)

Then I applied t-SNE to visualize the topic distributions of the documents in a 2D plot.
I used the LDA model to transform the tokenized data into topic distributions for each document and a TSNE object to apply the t-SNE algorithm to the topic distributions of the documents. This is needed to reduce dimensionality for visual inspection.

I created a a scatter plot of the t-SNE output using different colors for each topic. Each point on the plot represents a document, and the color of the point indicates its assigned topic. The legend on the plot shows which color corresponds to each topic.

IMAGE HERE

Documents that are closer together on the plot are more similar in terms of their topic distributions, while documents that are farther apart are less similar. 

I created a similar plot with binary labels in the y array, which indicate whether a joke is humorous or not representing their similarity based on their TF-IDF representations.

![t-SNE Visualization by Humor Type](imgs/tsne_visualization.png)

#

## Classification Modeling

- Simple logistic regression model using TFIDF vectorization
- Logistic regression with lemmatization to test if lemmatization will help accuracy
- Support Vector Machines (SVM) model
Random Forest with gridsearch
K Nearest Neighbors with gridsearch
XGBoost classification model
Basic neural network
Neural network with regularization
Neural network with dropout
Convolutional Neural Network (CNN)
Recurrent Neural Network (RNN)
Naive Bayes

## Results

1. Model #1: Simple logistic regression model using TFIDF vectorization.

Training Accuracy: 0.92921875
Test Accuracy: 0.9108

2. Model #2: Logistic regression with lemmatization to test if lemmatization will help accuracy.
Training Accuracy: 0.92335625
Test Accuracy: 0.905625

3. Model #3: Support Vector Machines (SVM) model.
Train Accuracy: 0.94555
Test Accuracy: 0.9144

4. Model #4: RandomForest with gridsearch
Training Accuracy: 0.897475
Test Accuracy: 0.85495

5. Model #5: K nearest neighbors with gridsearch
Training Accuracy: 1.0
Test Accuracy: 0.824

6. Model #6: XGBoost classification model
Training Accuracy: 0.789875
Test Accuracy: 0.7675

7. Model #7: Basic neural network 
Train Accuracy: 0.9999374747276306
Test Accuracy: 0.8657500147819519

8. Model #8: Neural network with regularization
Train Accuracy: 0.8952500224113464
Test Accuracy: 0.843999981880188

9. Model #9: Neural network with dropout
Train Accuracy: 0.999875009059906
Test Accuracy: 0.862500011920929

10. Model #10: CNN 
Train Accuracy: 0.9991250038146973
Test Accuracy: 0.8542500138282776

11. Model #11: RNN
Train Accuracy: 0.9983749985694885
Test Accuracy: 0.8497499823570251

12. Model #12: Naive Bayes
Train Accuracy: 0.947
Test Accuracy: 0.8835


## Evaluation

In this project, we aimed to develop a model to classify whether a given joke is humorous or not. We used a dataset of 20,000 jokes for training and testing various machine learning models.

We started with a simple logistic regression model using TFIDF vectorization, which achieved a training accuracy of 0.92921875 and a test accuracy of 0.9108. We then tried lemmatization with logistic regression, which achieved a training accuracy of 0.92335625 and a test accuracy of 0.905625.

We also experimented with Support Vector Machines (SVM), which achieved a training accuracy of 0.94555 and a test accuracy of 0.9144. Random Forest with grid search, K nearest neighbors with grid search, XGBoost classification model, basic neural network, neural network with regularization, neural network with dropout, CNN, and RNN models were also evaluated.

Naive Bayes achieved the best test accuracy of 0.8835 among all the models we tested. The training accuracy was 0.947, indicating that the model may be slightly overfitting on the training data. The SVM model was the second-best performing model with a test accuracy of 0.9144.

Overall, the results demonstrate that machine learning models can be effective at classifying jokes as humorous or not. Future work could involve using larger datasets or exploring more complex models, such as deep learning models, to achieve even better performance.

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
