setwd("C:/Users/h/Desktop")

#WebScrapping for 5000 Amazon reviews of Brita Standard Replacement Filters for Pitchers

# install.packages("rvest")
# install.packages("curl")
library(rvest)
library(curl)

#Read the URL
url <- "https://www.amazon.com/Brita-Standard-Replacement-Pitchers-Dispensers/product-reviews/B00BG53EF6/ref=cm_cr_getr_d_paging_btm_2?ie=UTF8&reviewerType=avp_only_reviews&pageNumber="
data <- read_html(paste(url,1,sep = ""))
review_content <- data %>% html_nodes(".review-text") %>%  html_text()

for(level in c(2:500)){# 500 pages, each one has 10 eviews (total: 5000 reviews)
  data <- read_html(paste(url,level,sep = ""))
  review_content <- c(review_content,data %>% html_nodes(".review-text") %>% html_text())
}


review <- as.data.frame(review_content)
View(review)
write.csv(review, "Amazon_Reviews.csv")
str(review)

#Sentiment Analysis and classification

# install.packages("tm")
# install.packages("RTextTools")
# install.packages("dplyr")
# install.packages("tidytext")
#  install.packages("RWeka") 
library(tm)
library(SnowballC)
library(wordcloud)
library(RWeka)
library(RTextTools)
library(plyr)
library('stringr')
library('dplyr')
library(tidytext)


#Because of the emojis in reviews, tolower function is not able to understand, so the following line will solve the issue
review$review_content <- iconv(review$review_content, 'UTF-8', 'ASCII')
# OR #usableText=str_replace_all(review$review_content,"[^[:graph:]]", " ") 
#OR #tm_map(review_corpus, function(x) iconv(enc2utf8(x), sub = "byte"))

# transfrom the dataset to a a corpus
review_corpus=Corpus(VectorSource(review$review_content)) 
#class(review$review_content) #character 

# applying transformation functions (also denoted as mappings) to corpora using tm_map

review_corpus <- tm_map(review_corpus, content_transformer(tolower)) #Switch to lower case #tolower just returns a character vector, and not a "PlainTextDocument" like tm_map would like. The tm packages provides the content_transformer function to take care of managing the PlainTextDocument
review_corpus <- tm_map(review_corpus, removeNumbers) #Remove numbers
review_corpus <- tm_map(review_corpus, removePunctuation) #Remove punctuation marks  
review_corpus <- tm_map(review_corpus, removeWords, stopwords("english"))  #List of the stop words: stopwords("english")
review_corpus <- tm_map(review_corpus, stripWhitespace) #Remove extra whitespaces

str(review_corpus)


#After the above transformations the first review looks like:
inspect(review_corpus[1])
# Using Document-Term Matrix (DTM) representation: documents as the rows, terms/words as the columns, frequency of the term in the document as the entries. Because the number of unique words in the corpus the dimension can be large
review_dtm <- DocumentTermMatrix(review_corpus)
#OR  review_dtm <- DocumentTermMatrix(review_corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

review_dtm
dim(review_dtm)
length(review_dtm$dimnames$Terms)

inspect(review_dtm[500:505, 500:505])

#To reduce the dimension of the DTM, we can remove the less frequent terms such that the sparsity is less than 0.95
## The following function takes a second parameters, the sparsity threshold.
## The sparsity threshold works as follows:
## If we say 0.98, this means to only keep terms that appear in 2% or more of the reviews.
## If we say 0.99, that means to only keep terms that appear in 1% or more of the reviews.
# By selecting only Termd that appear in at least 1% of the reviews, the number of Terms in the Document Term Matrix was reduced
review_dtm = removeSparseTerms(review_dtm, 0.99)
review_dtm
dim(review_dtm)

#The first review now looks like
inspect(review_dtm[1,1:20])

#Drawing a simple word cloud
findFreqTerms(review_dtm, 1000)
freq = data.frame(sort(colSums(as.matrix(review_dtm)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=50, colors=brewer.pal(1, "Dark2"))

# to make word cloud more informative
review_dtm_tfidf <- DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.95)
review_dtm_tfidf
# The first document
inspect(review_dtm_tfidf[1,1:20])
#Drawing a more informative word cloud
freq = data.frame(sort(colSums(as.matrix(review_dtm_tfidf)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2"))

library(NLP)
library(Rstem)
require(quanteda)

positive = read.table("positive-words.txt", sep="\t") #list, we can use unlist to convert from list to vector
negative = read.table("negative-words.txt", sep="\t")

str(positive)

# Stemming the positive words and negative words
stem <- function(list){
  for(i in c(1:nrow(list))){
    list[i,1] <- wordStem(String(list[i,1])) # Positive is data.frame':	2039 obs. of  1 variable (only 1 column)
  }
  list <- unique(list[,1] ) # duplicate elements/rows removed
}

positive <- stem(positive)
negative <- stem(negative)


#  Tokenizing & Stemming then labeling all 5000 reviews with positive or negative labels
str(review)

for(level in c(1:nrow(review))){
  str <- String(review$review_content[level])
  tokens <- str[wordpunct_tokenizer(str)]
  stem <- wordStem(tokens)
  positiveCount <- length(intersect(stem,positive))
  negativeCount <- length(intersect(stem,negative))
  review$PositiveWordsCount[level] <- positiveCount
  review$NegativeWordsCount[level] <- negativeCount
  if((positiveCount + negativeCount) == 0){
    positiveCount <- 1 
    negativeCount <- 1
  }
  review$Sentiment_Positive[level] <- positiveCount / (positiveCount + negativeCount) 
  review$Sentiment_Negative[level] <- negativeCount / (positiveCount + negativeCount)
  if(positiveCount >= negativeCount){
    review$result[level] <- "Positive"
  } else {
    review$result[level] <- "Negative"
  }
}



str(review)
write.csv(review, file = "Amazon_Reviews_Labels.csv")


# Sentiment classification using NBC (Naive Bayes Classifier) modeling

# Convert DTM to a data.frame for NBC  modeling
dataSparse <- as.data.frame(as.matrix(review_dtm), row.names = F)
View(dataSparse)
str(dataSparse)

review <- read.csv("Amazon_Reviews_Labels.csv")
numeric_result <- as.numeric(review$result)-1 # Positive=1, Negative=0

# Add the dependent variable to the data.frame
dataSparse <- cbind(dataSparse,numeric_result)

write.csv(dataSparse,file = "Sparse_Representation.csv")


#  NBC (Naive Bayes Classifier) modeling

library(RTextTools)
library(e1071)

#Splitting dataset into train and test datasets
set.seed(2) # we set the seed to make sure that the train and test data will not change every time we divide them by running the sample function

# running sample function to select randomly 80% index numbers of the dataset and use it to divide the dataset into 80% as a train dataset and the remaining 20% as a test dataset
sample_index <- sample(1:nrow(dataSparse),round(0.8*nrow(dataSparse))) #length(sample_index) should be %80 of the dataset

dataSparse.train <- dataSparse[sample_index,] #80% of dataset is train data = 4000 rows
dataSparse.test <- dataSparse[-sample_index,] #20% of dataset is test data = 1000 rows


#e1071 asks the response variable to be numeric or factor
classifier <- naiveBayes(dataSparse.train[,-173], as.factor(dataSparse.train$numeric_result) )

#Predicting the Response for test data
predicted <- predict(classifier, dataSparse.test[,-173])


#Print Prediction summary
summary(predicted)
str(predicted)

#Print the confusion matrix
confusion_Matrix <- table(predicted, dataSparse.test$numeric_result)
print(confusion_Matrix)

#Accuracy
recall_accuracy(dataSparse.test$numeric_result, predicted)

# #Computing accuracy
# accuracy <- (sum(diag(confusion_Matrix))/sum(confusion_Matrix)) * 100 # computing the accuracy by dividing the sum of diagonal matrix (the correct predictions) by the total sum of the matrix
# print(accuracy)


#Adding the prediction column and writing it to a new file
predicted_Data <- data.frame(dataSparse.test,predicted)
write.csv(predicted_Data, file = "Predicted_Review_Results.csv", row.names = FALSE)

