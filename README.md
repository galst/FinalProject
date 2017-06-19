# Assignment 4 - Machine Learning With Python

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
