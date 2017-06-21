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


<img src="https://github.com/galst/FinalProject/tree/master/img/hilary-cloud.PNG" width="200">

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
![](img/trump-clusteredgebetweenness.PNG.PNG)


- Adele
![](img/adele-clusteredgebetweenness.PNG)


- Ronaldo
![](img/cristiano-clusteredgebetweenness.PNG)


- Gates
![](img/gates-clusteredgebetweenness.PNG)
