#############################################################
# Machine Learning for Social Sciences
# Felix Hagemeister
# 2022

# R Script: Preprocessing and Document Feature Matrix (DFM)

# Clear environment
# -------------------------------
rm(list = ls())

# Set custom path using system user name
# -------------------------------
if (Sys.getenv("USERNAME") == "felix"){
  setwd("C:/Users/felix/Dropbox/HfP/Teaching/WiSe21/ML/")}
if (Sys.getenv("USERNAME") == "[YOUR USER NAME HERE]"){
  setwd("[YOUR PATH HERE")}

# Load packages
# -------------------------------
library(quanteda)
library(readtext)

# Read in text
# -------------------------------
dat <- readtext("data/manifestos/*.pdf", 
                docvarsfrom = "filenames",
                docvarnames = c("party","year"),
                dvsep = "_", 
                encoding = "UTF-8")

# Note that you can also read in text from various other formats 
# (e.g. .csv, .tab, .json, .xml, .html, .pdf, .doc, .docx, .rtf, .xls, .xlsx).


str(dat)


# Convert to quanteda corpus
# -------------------------------

corp <- corpus(dat)

# show summary
summary(corp)

# Edit corpus object
# -------------------------------

# edit docnames
docid <- paste(dat$party, 
               dat$year,  sep = " ")
docnames(corp) <- docid

# extract document-level variables
docvars(corp, field = "year")

# add document-level variables
docvars(corp, field = "country") <- "Germany"

# show summary of first 5 documents
summary(corp, 5)

# subset corpus
corp_recent <- corpus_subset(corp, year >= 2021)
ndoc(corp_recent)

corp_greens <- corpus_subset(corp, party %in% "greens")
ndoc(corp_greens)


# Change the unit of texts 
# -------------------------------

# set unit to sentences
corp_sent <- corpus_reshape(corp, to = "sentences")
ndoc(corp_sent)

# Restore the original documents
corp_doc <- corpus_reshape(corp_sent, to = "documents")
ndoc(corp_doc)

# get a corpus of sentences with minimum 10 words
corp_sent_long <- corpus_subset(corp_sent, ntoken(corp_sent) >= 10)
ndoc(corp_sent_long)


# Generate a Document Feature Matrix (DFM)
# -------------------------------
# dfm can be generated from tokens or corpus object

dfm <- dfm(corp,
           tolower = TRUE,               
           stem = TRUE,               
           remove_punct = TRUE,
           remove_numbers= TRUE ,
           remove = stopwords("German"),
           ngrams = 1)    

# Explore Data
# -------------------------------

ndoc(dfm) # number of documents
nfeat(dfm) # number of features
dfm

# first five document names
head(docnames(dfm), 5)
# first 20 features in the dfm
head(featnames(dfm), 20)

# how many features in first 10 documents
head(rowSums(dfm), 10)
# how many mentions of first 10 features
head(colSums(dfm), 10)
# 10 most frequently mentioned features
topfeatures(dfm, 10)


# Plot Zipf's Law
# -------------------------------

plot(1:ncol(dfm),sort(colSums(dfm),dec=T),
     main = "Zipf's Law?", ylab="Frequency", xlab = "Frequency Rank")

plot(1:ncol(dfm),sort(colSums(dfm),dec=T),
     main = "Zipf's Law?", ylab="Frequency", xlab = "Frequency Rank", log="xy")

## END