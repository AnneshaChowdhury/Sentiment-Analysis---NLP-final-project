# Note: before running this script, you should install NLTK python library
#       before hand, we will use NLTK to deal with each word in the training 
#       data and testing data.
from nltk.stem import SnowballStemmer 
import nltk
import re
import sys
import os
import math

stemmer = SnowballStemmer("english")
regx = re.compile("[^a-zA-Z]")

categorys = { "pos", "neg" }
token_idf = { "pos": {}, "neg": {} }
token_tfidf = { "pos": {}, "neg": {} }
file_num = { "pos": 0, "neg": 0 }

def ScanTrainFile(filename, category):
    filehandle = open(filename, encoding='utf-8')
    filelines = filehandle.readlines()
    
    tokenlist = {}    
    tokensum = 0
    for line in filelines:
        # First we try to split the line using non alpha characters 
        strings = regx.split(line)
        for str in strings:
            if len(str) > 2:  
                # Get the token appearance in each test document 
                tokensum = tokensum + 1
                times = tokenlist.get(str, 0)
                times = times + 1
                tokenlist[str] = times
     
    # Record how many documents appear this token 
    for token, val in tokenlist.items():
        times = token_idf[category].get(token, 0)
        times = times + 1
        token_idf[category][token] = times 
        
        tmp_tf = token_tfidf[category].get(token, 0)
        tmp_tf = tmp_tf + (val * 1.0 / tokensum)
        token_tfidf[category][token] = tmp_tf


def TrainData(traindatdir):
    for tmpcat in categorys: 
        print ("Start to train " + tmpcat + " data")
        trainpath = traindatdir + "\\train\\" + tmpcat + "\\";
        filenum = 0
        for fn in os.listdir(trainpath):
            fullname = trainpath + fn
            filenum = filenum + 1 
            ScanTrainFile(fullname, tmpcat)
        file_num[tmpcat] = filenum

    # Now we calculate the stastics for TF-IDF algorithm 
    for tmpcat, stat in token_idf.items():
        for token, value in token_idf[tmpcat].items():
            token_idf[tmpcat][token] = math.log(file_num[tmpcat] / (value + 1))
            # Calculate the average tf value 
            token_tfidf[tmpcat][token] = token_tfidf[tmpcat][token] / file_num[tmpcat]
            token_tfidf[tmpcat][token] = token_tfidf[tmpcat][token] * token_idf[tmpcat][token]


# Classify the movie review using TFIDF
def Classify(tokenlist):
    maxlog = -100000000.0
    rescat = ""
    
    # Now we calculate the score for each class 
    for tmpcat in categorys: 
        tmpres = 0
        for word, count in tokenlist.items():
            tfidf = token_tfidf[tmpcat].get(word, 0) 
            tmpres = tmpres + tfidf     
        if tmpres > maxlog:
            maxlog = tmpres
            rescat = tmpcat

    return rescat


def TestData():
    print ("Start to check movies using TF-IDF")
    filehandle = open("critics.csv", encoding='utf-8')
    filehandle.readline()    # Ignore the header 
    filelines = filehandle.readlines()

    review_dat = {}
    # Get all the token list for single IMDB movie 
    for line in filelines:
        reviewparts = line.split(',')
        size = len(reviewparts)
        title = reviewparts[size - 1]
        title = title.replace('\n', '')

        index = 4
        quote = "" 
        while index <= size - 4:
            quote = quote + " " + reviewparts[index]
            index = index + 1
        reviewtoken = review_dat.get(title, {} )

        # We construct a brief information for that review 
        brief_info = quote + " " + title
        strings = regx.split(brief_info)
        for str in strings:
            if len(str) > 2:
                str = stemmer.stem(str)
                reviewtoken[str] = 1
        review_dat[title] = reviewtoken
    
    for title, reviewtoken in review_dat.items(): 
        if len(reviewtoken) > 10:   # Do not have any comments 
            print ("Movie[" + title + "], review:" + Classify(reviewtoken))


if len(sys.argv) != 2:
    print ("Usage: python mvreview.py traindir")
    sys.exit(0)
TrainData(sys.argv[1])
TestData()

