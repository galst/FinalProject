# Final Project

#### Gal Bar 200462133, Gal Steimberg 201253572

## Phase 1 - Data Collection

In this phase of the project we will gather the data needed for the rest of the final project. We will collect Facebook posts from public figures using the Facebook API in R. 

The process of phase 1 will be the following:
1. Collect the 1000 latest Facebook posts from several popular figures
2. Manipulate the data to fit our needs
3. Explore the data and manipulate it
4. Save the dataset to the disk


#### Fetching the data from Facebook

Here we wrote an R script that uses 'Rfacebook' package in order to access the Facebook API.

First we will set up the configuration that will allow us to access Facebook: 

```r
library(Rfacebook)
library(tm)
library(RCurl)
library(SnowballC)
library(wordcloud)

app_id <- "219932136741834182"
app_secret <- "1e8ad2627d91239702754e9487af039a571"
fb_oauth <- fbOAuth(app_id, app_secret)
access_token <- "EAADICANevcYBALERIpywyS123q2x3YZBNBU9Wr2jR4ZA8Vsh5aQTWtngZAsIZAMi8zFRroGCEVkJBI9ab1ZBpLws3KYrxQhnVZB4DDm6yML9vZCDHAb4HJZACgZAjUca67Jyl3t7klRQfNRUMuYp8oldMeLKzPZAhucyCXscZD"
```
* The ids shown above are fake ids and not the real ids we used (in order to refrain from privacy issues)


Now that we set up our credenentials we can move on to retrieving the data. We picked five different public figures and used the API to get their latest posts:

```r
#***cristiano***
cristiano_page <- getPage(page="Cristiano", token=access_token, n=100)
message <- cristiano_page$message

write.csv(message, file="cristiano.csv")
#***run graphs script here***

#***trump***
trump_page <- getPage(page="DonaldTrump", token=access_token, n=100)
message <- trump_page$message

write.csv(message, file="trump.csv")
#***run graphs script here***

#***adele***
adele_page <- getPage(page="adele", token=access_token, n=100)
message <- adele_page$message

write.csv(message, file="adele.csv")
#***run graphs script here***

#***hilary***
hillary_page <- getPage(page="hillaryclinton", token=access_token, n=100)
message <- hillary_page$message

write.csv(message, file="hillary.csv")
#***run graphs script here***

#***bill gates***
gates_page <- getPage(page="BillGates", token=access_token, n=100)
message <- gates_page$message

write.csv(message, file="gates.csv")
```
#### Data Cleaning

Now that we have the raw data we can move on to cleaning it a bit
The cleaning that we performed met the following criteria:

- Remove numbers
- Remove english common stopwords
- Remove punctuations
- Eliminate extra white spaces
- Text stemming
- Minor error fixing

```r

docs <- Corpus(VectorSource(message))

toSpace <- content_transformer(function(x,pattern) gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")

#Remove numbers
docs <- tm_map(docs, removeNumbers)

#Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))

#Remove punctuations
docs <- tm_map(docs, removePunctuation)

#Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)

#Text stemming
docs <- tm_map(docs, stemDocument)

#error fix
docs <- tm_map(docs, function(x) iconv(enc2utf8(x), sub = "byte"))

```

#### Dataframe Extractio and Data Exploration

Now we would like to explore the data that we have a bit more.
In total we have posts for five different figures, and our final goal is to build a predictor for a new post for each of these figures. Let's try and take a look at the popular words for each figure to get a high level understanding of the common words for each:

First we will count the words for each figure and post and then create a word could from it

```r
#Word counts
tdm <- TermDocumentMatrix(docs)
m <- as.matrix(tdm)
v <- sort(rowSums(m), decreasing=TRUE)
d <- data.frame(word=names(v), freq=v)
head(d, 10)

#Draw word cloud
wordcloud(words=d$word, freq=d$freq, min.freq=1, max.words=200, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))
```

- Hillary:
![](img/hilary-cloud.PNG)


- Trump:
![](img/trump-cloud100.PNG)


- Adele
![](img/adele-cloud.PNG)


- Ronaldo
![](img/cristiano-cloud.PNG)


- Gates
![](img/gates-cloud.PNG)


Now that we have a sense we have a good sense of the most common words for each figure, lets translate the data into a graph and calculate some important features of it. We think that looking at the problem as a graph can help us visualize the data better and get to a better understanding of it. By calculating these feature we can have some sense on:

- The most important words
- Words appearing next to each
- Words usually coming together
- First impression of possible sentences
- Central words to each post and to each figure


First we need to decribe the data as an adjacency matrix:
```r
#Transform data into an adjacency matrix
#using this: https://rdatamining.wordpress.com/2012/05/17/an-example-of-social-network-analysis-with-r-using-package-igraph/

tdm <- TermDocumentMatrix(docs)
freq_terms <- findFreqTerms(tdm, lowfreq=10)
tdm <- tdm[freq_terms, ]
m <- as.matrix(tdm)
m[m >= 1] <- 1
tm <- m %*% t(m)

```

Now we can create graph for each figure using the following:
```r
#Create the graph
#build a graph from the above matrix
graph <- graph.adjacency(tm, weighted=T, mode="undirected")
#remove loops
graph <- simplify(graph)
#set labels and degrees of vertices
V(graph)$label <- V(graph)$name
V(graph)$degree <- degree(graph)
#design the graph
V(graph)$label.cex <- 2.2 * V(graph)$degree / max(V(graph)$degree)+ .2
V(graph)$label.color <- rgb(0, 0, .2, .8)
V(graph)$frame.color <- NA
egam <- (log(E(graph)$weight)+.4) / max(log(E(graph)$weight)+.4)
E(graph)$color <- rgb(.5, .5, 0, egam)
E(graph)$width <- egam

layout <- layout.fruchterman.reingold(graph)
plot(graph, layout=layout)
```

And here are the results:

- Hillary:
![](img/hilary-graph.PNG)


- Trump:
![](img/trump-graph.PNG)


- Adele
![](img/adele-graph.PNG)


- Ronaldo
![](img/cristiano-graph.PNG)


- Gates
![](img/gates-graph.PNG)


When taking a look at these graph it is interesting to see to how the there are common (central) words for all the figures on one hand, while on the other hand we see that there are some posts that have totaly different words with totaly different weights and importance.

Through these graphs we can understand the features of each figure and the words he uses together and their ordering as well.

Next we would like to calculate betweeness and closeness for each of these figures. Betweeness and closeness will help us understand better the importance of each word is in a standard post for each figure. So, when we can understand how much "impact" a specif word will have on our predictor.


```r
#betweenness
betweenness <- betweenness(graph, v=V(graph))
betweenness

#closeness
closeness <- closeness(graph, vids=V(graph))
closeness
```

The Edge Betweenness algorithm detects communities by progressively removing edges from the original network. The connected components of the remaining network are the communities. Instead of trying to construct a measure that tells us which edges are the most central to communities, the Edge Betweeness algorithm focuses on edges that are most likely "between" communities. 

We use it here to detect words close to each other or words which could possibly fit together in a post:

```r
#cluster edge betweenness
set.seed(1)
ceb <- cluster_edge_betweenness(graph)
ceb
plot(graph, vertex.color=membership(ceb))
```

- Hillary:
![](img/hilary-clusteredgebetweenness.PNG)


- Trump:
![](img/trump-clusteredgebetweenness.PNG)


- Adele
![](img/adele-clusteredgebetweenness.PNG)


- Ronaldo
![](img/cristiano-clusteredgebetweenness.PNG)


- Gates
![](img/gates-clusteredgebetweenness.PNG)


By observing these graph we can find some interesting results on our data. This information will help us in phase two by allowing us to traing a better classsifier.

## Phase 2 - Classifier Training

In this phase of the project we used several classifying algorithms and trained them with 80% of the data in order to predict the 20% left.

#### Importing data collected in phase 1

Here we are reading all the data that we collected in the previous phase from a csv file seperated into 3 columns defining id, content (text sequence) and figure (type of text sequence)

```r
library(tm)

#First install and load packages needed for text mining.

posts = read.csv("<path>/allPosts.csv", stringsAsFactors = F,row.names = 1)
```

#### Normalizing text

Now we will transform the data set into a corpus, normalize the text using a series of pre-processing steps:

- witch to lower case
- Remove numbers
- Remove punctuation marks and stopwords
- Remove extra white spaces

```r-using-package-igraph/#To use the tm package we first transfrom the dataset to a corpus:
post_corpus = Corpus(VectorSource(posts$content))

post_corpus

#Next we normalize the texts in the post using a series of pre-processing steps
#Switch to lower case 
#Remove numbers
#Remove punctuation marks and stopwords
#Remove extra whitespaces
post_corpus = tm_map(post_corpus, content_transformer(tolower))
post_corpus = tm_map(post_corpus, removeNumbers)
post_corpus = tm_map(post_corpus, removePunctuation)
post_corpus = tm_map(post_corpus, removeWords, c("the", "and", stopwords("english")))
post_corpus = tm_map(post_corpus, stripWhitespace)
```

#### Analyzing the textual data

In order to analyze the textual data we use DTM representation. after creating a dtm from the corpus we will do a few manipulations and then split to training (80%) and testing (20%) sets.

```
#To analyze the textual data, we use a Document-Term Matrix (DTM) representation: 
post_dtm <- DocumentTermMatrix(post_corpus)
post_dtm

inspect(post_dtm[50:55, 50:55])

#To reduce the dimension of the DTM, we can remove the less frequent terms such that the sparsity is less than
#0.99
post_dtm = removeSparseTerms(post_dtm, 0.99)
post_dtm

post_dtm_tfidf <- DocumentTermMatrix(post_corpus, control = list(weighting = weightTfIdf))
post_dtm_tfidf = removeSparseTerms(post_dtm_tfidf, 0.95)

post_dtm_tfidf

#Let’s remove the actual texual content for statistical model building
posts$content = NULL
#Now we can combine the tf-idf matrix with the sentiment figure according to the sentiment lists.
posts = cbind(posts, as.matrix(post_dtm_tfidf))
posts$figure = as.factor(posts$figure)

#Split to testing and training set
id_train <- sample(nrow(posts),nrow(posts)*0.80)
posts.train = posts[id_train,]
posts.test = posts[-id_train,]
```

#### Evaluating performance

First we import the relevant libraries

```r
library(rpart)
library(rpart.plot)
library(e1071)
```

We ran a few different classifying algorithms. Each once seperated into 2 parts. first part, building the classifier using the training set. second part, predicting the type of the textual sequences from the testing set.

- SVM
Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on on which side of the gap they fall.

```r
#svm
posts.svm = svm(figure~ ., data = posts.train);
pred.svm = predict(posts.svm, posts.test)
```

```r 
table(posts.test$figure,pred.svm,dnn=c("Obs","Pred"))
```
![](img/svm-table.PNG)

```r 
plot(pred.svm)
```
![](img/svm-plot.PNG)

```r 
mean(ifelse(posts.test$figure != pred.svm, 1, 0))
```
![](img/svm-mean.PNG)

- Random forest
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
```r
library(randomForest)
#random forest
posts.rf = randomForest(figure~ ., data = posts.train, importance=TRUE)
plot(posts.rf)
pred.rf= predict(posts.rf,posts.test)
```

```r 
table(posts.test$figure,pred.svm,dnn=c("Obs","Pred"))
```
![](img/randomForest-table.PNG)

```r 
plot(pred.rf)
```
![](img/randomForest-plot.PNG)

```r 
mean(ifelse(posts.test$figure != pred.rf, 1, 0))
```
![](img/randomForest-mean.PNG)

Loading more libraries
```r

# Load libraries
library(mlbench)
library(caret)
library(caretEnsemble)
```
#### Example of Boosting Algorithms

Boosting is a machine learning ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and a family of machine learning algorithms which convert weak learners to strong ones. Boosting is based on the question posed by Kearns and Valiant (1988, 1989) Can a set of weak learners create a single strong learner? A weak learner is defined to be a classifier which is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification.

```r 
# Example of Boosting Algorithms
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
```

- C5.0

```r
# C5.0
set.seed(seed)
fit.c50 <- train(figure~ ., posts.train, method="C5.0", metric=metric, trControl=control)
plot(fit.c50)
```

![](img/c50-plot.PNG)

- Stochastic Gradient Boosting

```r
# Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(figure~ ., posts.train, method="gbm", metric=metric, trControl=control, verbose=FALSE)
plot(fit.gbm)
```

![](img/sgb-plot.PNG)

- Boosting Algorithms Summary

```r
# summarize results
boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
summary(boosting_results)
```

![](img/boosting-summary.PNG)

```r
dotplot(boosting_results)
```

![](img/boosting-summary-plot.PNG)

#### Example of Bagging Algorithms

Bootstrap aggregating, also called bagging, is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach.

```r 
# Example of Bagging algorithms
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
```

- Bagged CART

```r
# Bagged CART
set.seed(seed)
fit.treebag <- train(figure~ ., posts.train, method="treebag", metric=metric, trControl=control)
```

- Random Forest

```r
# Random Forest
set.seed(seed)
fit.rf <- train(figure~ ., posts.train, method="rf", metric=metric, trControl=control)
```

- Bagging Algorithms Summary

```r
# summarize results
bagging_results <- resamples(list(treebag=fit.treebag, rf=fit.rf))
summary(bagging_results)
```

![](img/bagging-summary.PNG)

```r
dotplot(bagging_results)
```

![](img/bagging-summary-plot.PNG)

### Comparing the results
//TODO: compare the results



