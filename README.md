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

LDA and NMF are two tools used as a topic modeling technique to discover latent topics in a collection of documents. They model each topic as a probability distribution over the words in the vocabulary to explain the topic-word distributions.

For topic modeling with LDA and NMF, lemmatization, tokenization and vectorization are used to prepare the text data for input into the model. TfidfVectorizer is used to tokenize the lemmatized text data, and an LDA model is created and trained on the tokenized data. I analyze the topic structure of the jokes in the dataset and print the top words for each topic using get feature names. NMF provided the strongest results and clearest topic distributions so this is the topic modeling technique I chose. 

```
# Print the top 10 words for each topic
for i, words in enumerate(top_words):
    print(f"Topic {i}:")
    print(", ".join(words))

```
```
Topic 0:
one, make, man, day, time, it, never, two, take, women
Topic 1:
trump, donald, trumps, clinton, hillary, says, president, gop, us, obama
Topic 2:
get, cross, road, chicken, side, it, married, christmas, cant, pregnant
Topic 3:
people, black, world, white, cant, hate, types, 10, think, racist
Topic 4:
new, photos, york, years, video, week, year, fashion, best, 10
```

The function "get_top_tokens" is defined to extract the most important words for each topic, along with their corresponding weights, for the 5 defined topics. The feature names for the vectorizer used to process the input documents are obtained using the "get_feature_names_out" method. Finally, a bar plot is generated for each topic, displaying the top tokens and their weights.

![Topic 0 Weights](imgs/topic_weights_0.png)
![Topic 1 Weights](imgs/topic_weights_1.png)
![Topic 2 Weights](imgs/topic_weights_2.png)
![Topic 3 Weights](imgs/topic_weights_3.png)
![Topic 4 Weights](imgs/topic_weights_4.png)

Then I applied t-SNE to visualize the topic distributions of the documents in a 2D plot.
I used the LDA model to transform the tokenized data into topic distributions for each document and a TSNE object to apply the t-SNE algorithm to the topic distributions of the documents. This is needed to reduce dimensionality for visual inspection.

I created a a scatter plot of the t-SNE output using different colors for each topic. Each point on the plot represents a document, and the color of the point indicates its assigned topic. The legend on the plot shows which color corresponds to each topic.

![t-SNE Visualization by Topic](imgs/tsne_plot_topics.png)

Documents that are closer together on the plot are more similar in terms of their topic distributions, while documents that are farther apart are less similar. 

I created a similar plot with binary labels in the y array, which indicate whether a joke is humorous or not representing their similarity based on their TF-IDF representations.

![t-SNE Visualization by Humor Type](imgs/tsne_visualization.png)



#

## Classification Modeling

For this binary classification problem of selecting whether humor = 'true' or 'false' based on the clean text data, I analyzed 12 different models.

1. Simple logistic regression model using TFIDF vectorization
2. Logistic regression with lemmatization 
3. Logistic regression with part-of-speech tagging
4. Support Vector Machines (SVM) model
5. Random Forest with gridsearch
6. K Nearest Neighbors with gridsearch
7. XGBoost classification model
8. Basic neural network
9. Neural network with regularization
10. Neural network with dropout
11. Convolutional Neural Network (CNN)
12. Recurrent Neural Network (RNN)
13. Naive Bayes

## Results

Accuracy is the metric of greatest importance to our analysis and selection of a final model.

| Model                                                  | Training Accuracy | Test Accuracy |
|--------------------------------------------------------|------------------|---------------|
| Simple logistic regression model                        | 0.938            | 0.926         |
| Support Vector Machines (SVM) model                     | 0.953            | 0.929         |
| Random Forest with gridsearch                           | 1.000            | 0.918         |
| K Nearest Neighbors with gridsearch                     | 0.897            | 0.855         |
| XGBoost classification model                            | 1.000            | 0.883         |
| Basic neural network                                    | 0.938            | 0.890         |
| Neural network with regularization                       | 0.999            | 0.908         |
| Neural network with dropout                            | 0.999            | 0.903         |
| Convolutional Neural Network (CNN)                      | 0.997            | 0.894         |
| Recurrent Neural Network (RNN)                          | 0.942            | 0.894         |
| Naive Bayes                                             | 0.946            | 0.888        |

12 different classification models were tested using TF-IDF vectorization and selected highest performing train accuracy. SVM had the highest test accuracy of 0.933, followed by random forest with .929. Models like neural networks and XGBoost did not perform well for this problem. Removing punctuation and/or Lemmatization decreased model accuracy. Removing stop-words did not increase model accuracy.


## Evaluation

In this project, we aimed to develop a model to classify whether a given text is humorous or not. We used a dataset of 20,000 jokes for training and testing various machine learning models.

We started with a simple logistic regression model using TFIDF vectorization, which achieved a test accuracy of 0.926. We also experimented with Support Vector Machines (SVM), which achieved a training accuracy of 0.94555 and a test accuracy of 0.9144. Random Forest with grid search, K nearest neighbors with grid search, XGBoost classification model, basic neural network, neural network with regularization, neural network with dropout, CNN, and RNN models were also evaluated.

SVM model was the best performing model with a test accuracy of 0.93. 

Classification report:

| precision | recall | f1-score | support |
|-----------|--------|----------|---------|
|    0.93   |  0.93  |   0.93   |  20001  |
|    0.93   |  0.93  |   0.93   |  19999  |
|-----------|--------|----------|---------|
|    0.93   |  0.93  |   0.93   |  40000  |


Confusion matrix for final SVM model:

![Confusion Matrix SVM](imgs/confusion_matrix_final_model.png)

ROC curve for final SVM model:

![ROC Curve SVM](imgs/ROC_curve_final_model.png)

Overall, the results demonstrate that machine learning models can be effective at classifying texts as humorous or not. Future work could involve using larger datasets or exploring more complex models, such as deep learning models, to achieve even better performance.


## Conclusion and Recommendations

After evaluating several classification models on the joke dataset, we can conclude that the SVM model had the highest test accuracy of 0.9144, closely followed by the simple logistic regression model using TFIDF vectorization with a test accuracy of 0.9108. However, some models like XGBoost classification and K Nearest Neighbors did not perform well, indicating that they may not be suitable for this problem.

Based on these results, we recommend using the SVM or logistic regression model with TFIDF vectorization for joke classification. Additionally, exploring other natural language processing techniques like stemming or part-of-speech tagging may improve model accuracy further. Collecting more data and expanding the dataset can also help improve the models' generalizability.

Finally, we recommend exploring topic modeling to cohort users and understand their preferences better. This approach can help tailor the joke recommendations and improve user engagement.
