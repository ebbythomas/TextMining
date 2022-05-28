# Natural Language processing using R:
# Also called Text Mining in most of its sense.
# The way through which Machines respond to Humans. eg: Giving customised flight routes to a customer.
# Gives Machines the capability to understand human communication.
# Example of applications of NLP is Sentiment Analysis of customers, Chatbots, Information extraction from Google, Google Translate, Advertisement matching based on history,
# Spell checker, Informaiton extraction, etc...

# Steps:
# Tokenisation: Breaking up the sentences into words
# Stemming: All the forms of words are converted to its root form. eg: lovable, loving, all are converted into "love". Normally cut off beginning/end of the word. Not 100% correct though.
# Lemmatisation: Here, all like words are combined into one. eg: Takes in Going, Gone, Went and outputs "GO", ie, the LEMMA. This would not happen in Stemming.
# Remove Stop words: Remove words such as "A", "an", "the", "was", "is", ......eg: Google keywords.
# Document Term Matrix: Match documents to an already exisitng  document with 1s and 0s - 1s where the term is present and 0 if term is not present.
# Now, we get a Corpus -- a collectio of words/documents.
# POS Tags: Parts Of Speech are added. 
# NER : Name Entity Recognition: Adds Tags to each word. eg: Ebby is treated as "name of person", Brisbane as a "place name" etc..
# Chunking: Grouping words into meaningful sentences.

# Ever since humans started interacting with Machines, NLPs got popular.





# install.packages("tm") # Text Mining
library(tm)

myDocs <- Corpus(DirSource("NLP/inputFolder")) # Gives the document in a crunched form - has two documents in this folder.

# If we need to inspect the whole Corpus
casualInspect <- inspect(myDocs)

# If we need to inspect a specific document just in case
writeLines(as.character(myDocs[[2]]))

# Start pre- processing
# Almost as the first step, before starting to Remove Punctuations,  we need to treat the punctuations such as commas hyphens as spaces.
# else Machine would read trade-off as "tradeoff" and Ebby is in:Brisbane as inBrisbane which does not make sense. So replace all punctuations with spaces.
convertToSpace <- content_transformer(function(x,pattern) {return (gsub(pattern, " ",x))}) # content_transformer loaded from tm.

myDocs <- tm_map(myDocs, convertToSpace, "-") # Will convert all "-" to spaces.
myDocs <- tm_map(myDocs, convertToSpace, ":")
myDocs <- tm_map(myDocs, convertToSpace, "'")
myDocs <- tm_map(myDocs, convertToSpace, "_")
myDocs <- tm_map(myDocs, convertToSpace, ",")

writeLines(as.character(myDocs[[2]])) # Checking after every step. Cannot assign to a variable. Just like a "PRINT" command

myDocs <- tm_map(myDocs, removePunctuation)

myDocs <- tm_map(myDocs, content_transformer(tolower))

myDocs <- tm_map(myDocs, removeNumbers)

myDocs <- tm_map(myDocs, removeWords, stopwords("english"))

myDocs <- tm_map(myDocs, stripWhitespace)

writeLines(as.character(myDocs[[2]]))

# Stemming is done via another package - SnowballC
# install.packages("SnowballC")
library(SnowballC)

myDocs <- tm_map(myDocs, stemDocument) # Moving, Moved, etc will be converted to MOVE
writeLines(as.character(myDocs[[2]])) # We can see that some first and last alphabets are cut down. Depending on the text, we may/may not use STEMMING.


myDocs <- tm_map(myDocs, content_transformer(gsub), pattern = "describ", replacement = "describe")
myDocs <- tm_map(myDocs, content_transformer(gsub), pattern = "purpos", replacement = "purpose")

# Create a Document Term Matrix
dtm <- DocumentTermMatrix(myDocs) # This gives the arrangement/ repeatability of words in these documents.
dim(dtm)

inspect(dtm[1:2,1001:1010]) # For testing

# Now the text in the Corpus has been converted to a matrix.

freq <- colSums(as.matrix(dtm))

ordered <- order(freq, decreasing = TRUE) # Ordering the words based on their occurence

freq[head(ordered)] # Highest occuring words
freq[tail(ordered)] # Least occuring words
# Note that the least occuring words could be the most descriptive about the document. So least frequent words are probably more important.
# Here we remove the most occuring words
dtmr <- DocumentTermMatrix(myDocs, control = list(wordLengths = c(4,20), bounds = list (global =  c(3,27)))) 
freq2 <- colSums(as.matrix(dtmr))
ordered2 <- order(freq2, decreasing = TRUE) # Ordering the words based on their occurence

freq2[head(ordered2)] # Highest occuring words
freq2[tail(ordered2)]

# Text mining is an iterative process. So we may need to do all teh above processes over and over again to completely get rid of any unwanted stuffs such as commas..

findFreqTerms(dtm, lowfreq = 40) # This collects words that have occured at least 20 times (or more)


dfHist <- data.frame(term = names(freq), occurrences = freq)
# Can filter most used texts from
mostUsedTexts <- subset(dfHist, freq>30)

library(ggplot2)
myHist <- ggplot(subset(dfHist, freq>20), aes(term, occurrences)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust =1))

# Also can improve visualisation using wordcloud package
# install.packages("wordcloud")
library(wordcloud)
set.seed(1999)
wordcloud(names(freq), freq, min.freq = 30)
wordcloud(names(freq), freq, min.freq = 30, colors= brewer.pal(8,"Dark2"))


################ SENTIMENT ANALYSIS FOR TWITTER TWEETS ABOUT APPLE--------------------
# Twitter data is stored in the inputfolder, with first column is the text (actual tweet) and rest columns indicating the time, etc..
# Getting input Data ----------
appleData <- read.csv("inputFolder/apple.csv")
# This has 1000 observations -- which means it has 1000 tweets. We require only the texts, ie, first ccolumn only..
typeof(appleData$text)
# Building Corpus------
library(tm)
library(dplyr)
myTweets <- iconv(appleData$text, to = "utf-8") # "utf-8-mac" if Mac.
myCorpus <- Corpus(VectorSource(myTweets))
inspect(myCorpus[1:4])

# Data Cleaning ------------
myCorpus <- tm_map(myCorpus, tolower)
inspect(myCorpus[1:4])

myCorpus <- tm_map(myCorpus, removePunctuation)
myCorpus <- tm_map(myCorpus, removeNumbers)
stopwords("english") # Just to see the stopwords in english that are gonna be removed. 
myCorpus <- tm_map(myCorpus, removeWords, stopwords("english")) 

urlRemovedCorpus <- tm_map(myCorpus, content_transformer(gsub), pattern = "http[[:alnum:]]*", replacement = "") # * Makes sure that the whole string after http is removed.
inspect(urlRemovedCorpus[1:4])

cleanCorpus <- tm_map(urlRemovedCorpus, stripWhitespace)
cleanCorpus <- tm_map(cleanCorpus, removeWords, c("aapl", "apple")) # the reason for removing this is below.
cleanCorpus <- tm_map(cleanCorpus, gsub, pattern = "stocks", replacement = "stock") # Replacing otehr way is WRONG. THINK...!
cleanCorpus <- tm_map(cleanCorpus, gsub, pattern = "calls", replacement = "call") 
cleanCorpus <- tm_map(cleanCorpus, gsub, pattern = "reports", replacement = "report") 
cleanCorpus <- tm_map(cleanCorpus, removePunctuation)
inspect(cleanCorpus[1:4])

# Now, Text data is UNSTRUCTURED data. TO do any analysis on it, we need to convert it into a STRUCTURED data. So MATRIX.
# Term Document Matrix---------
tdm <- TermDocumentMatrix(cleanCorpus)
# Aparsity is 99% means 99% of the entries are 0s.

tdm <-  as.matrix(tdm)
tdm[1:10,1:20]
# can be seen that aapl is everywhere. This is bcoz the tweet is about aapl. So obvious.
#'*>>>>> Backstep 1*'
# Need to remove this. This is the reason why aapl, apple etc has been removed from cleanCorpus.
# Histogram - ---------
freqHist <- rowSums(tdm) # This gives the number of occurrences of each word
freqSignificant <- subset(freqHist, freqHist>25) # # This gives the number of occurrences of each SIGNIFICANT word that occurs >25 times.
sort(freqSignificant)
barplot(freqSignificant,
        las = 2, # las = 0: Parallell to axis, 1: Always horizontal, 2: Always perpendicular
        col = rainbow(25))

sort(names(freqSignificant))
#'*>>>>> Backstep 2*'
# We can see that there are STOCKS and STOCK, CALL and CALLS, REPORT, REPORTS. They can be merged. So GO BACK to cleanCorpus.

# Wordcloud --------

library(wordcloud)# Also there is a library known as library WORDCLOUD2
myWords <- sort(rowSums(tdm), decreasing = TRUE)
set.seed(1)
wordcloud(words = names(myWords),
          freq = myWords,
          max.words = 150,
          random.order = FALSE,
          min.freq = 5,
          colors = brewer.pal(8, 'Dark2'),
          scale = c(7,0.3)) # Max is 5 and Min is 0.3
          
#### SENTIMENT ANALYSIS OF TWITTER DATA

#install.packages("syuzhet")
library(syuzhet)
library(lubridate)
library(dplyr)
library(ggplot2)
library(reshape2)
library(scales)
# Read file --------
appleDf <- read.csv("inputFolder/apple.csv")
myTweets <- iconv(appleDf$text, to = "utf-8")

# Obtain sentiment scores--------
sentimentsApple <- get_nrc_sentiment(myTweets) # This will call NRC Sentiment scores for all the 10 variables in the Sentiment metric
head(sentimentsApple)

# If we need to look at certain specific tweets, for instance, 4th Tweet shows some disgust and fear:
 myTweets[4]
 
 # Now lets look the score for each word, such as UGLY
 get_nrc_sentiment("ugly")
 get_nrc_sentiment("delay")
# Similarly all words contributeto the net score of the tweet
 barplot(colSums(sentimentsApple),
         las = 2,
         col = rainbow(10),# Since there are 10 columns
         ylab = "Count",
         main = "Sentiment scores for Apple tweets")

#'*COMPARISON OF APPLE TWEETS AT DIFFERENT POINTS OF TIME*'  
 # Now we compare this one to the APPLE tweets after some days (probably after some phone release)-------------
 appleDfAfter <- read.csv("inputFolder/apple2.csv") # This is the apple tweet after some time
 myTweetsAfter <- iconv(appleDfAfter$text, to = "utf-8")
 
 sentimentsAppleAfter <- get_nrc_sentiment(myTweetsAfter) # This will call NRC Sentiment scores for all the 10 variables in the Sentiment metric
 head(sentimentsAppleAfter)
 
 # Similarly all words contributeto the net score of the tweet
 barplot(colSums(sentimentsAppleAfter),
         las = 2,
         col = rainbow(10),# Since there are 10 columns
         ylab = "Count",
         main = "Sentiment scores for Apple tweets- AFTER")
# We can compare the tweets of Apple and see that the Sentiment of People have improved over time AFTER the launch of a new iPhone etc... 