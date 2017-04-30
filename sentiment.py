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
# Token appearance for NativeBayes  
tokendata_NB = { "pos": {}, "neg": {} }
# The token percent in each class 
tokenpercent_NB = { "pos": {}, "neg": {} }

token_tfidf = { "pos": {}, "neg": {} }
token_idf = { "pos": {}, "neg": {} }
filenum_tfidf = { "pos": 0, "neg": 0 }

alphabet = 0    # Alphabet size in trainning dataset   

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
                # Lower than 2, we treat it as an invalid word, first deal with the sentence for Native Bayes  
                str = stemmer.stem(str)
                times = tokendata_NB[category].get(str, 0);
                times = times + 1
                tokendata_NB[category][str] = times
                
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


def CalcTrainStat():
    # Here we try to get all the statastic for the training data 
    tokenlist = {}
    # The total tokens in the training data  
    tokensum = 0
    
    for key, stat in tokendata_NB.items():
        sum = 0
        for token, count in stat.items():
            sum = sum + count
            tokenlist[token] = 1
        stat[u'_num'] = sum    # No token start with '_', nltk return utf-8 string, so we use utf8 key here 
        tokensum = tokensum + sum     # update the total word number in the training dataset 

    alphabet = len(tokenlist) 

    # We calculate the tokenpercent_NB here 
    for key, stat in tokendata_NB.items():
        stat[u'_percent'] = stat[u'_num'] * 1.0 / tokensum
        ratiomap = tokenpercent_NB[key]
        for token, count in stat.items():
            if token.startswith(u'_'):
                continue
            else:
                ratiomap[token] = (stat[token] + 1) * 1.0 / (stat[u'_num'] + alphabet)
    
    # Now we calculate the stastics for TF-IDF algorithm 
    for cat, stat in token_idf.items():
        for token, value in token_idf[cat].items():
            token_idf[cat][token] = math.log(filenum_tfidf[cat] / (value + 1))
            # Calculate the average tf value 
            token_tfidf[cat][token] = token_tfidf[cat][token] / filenum_tfidf[cat]
            token_tfidf[cat][token] = token_tfidf[cat][token] * token_idf[cat][token]

def TrainData(dirpath):
    for cat in categorys: 
        print ("Start to train " + cat + " data")
        trainpath = dirpath + "\\train\\" + cat + "\\";
        filenum = 0
        for fn in os.listdir(trainpath):
            fullname = trainpath + fn
            filenum = filenum + 1 
            ScanTrainFile(fullname, cat)
        filenum_tfidf[cat] = filenum

    CalcTrainStat()


def Classify_NB(filename, category):
    filehandle = open(filename, encoding='utf-8')
    filelines = filehandle.readlines()
    tokenappear = {}   # store the word appear times in this document 
    
    for line in filelines:
        strings = regx.split(line)
        for str in strings:
            if len(str) > 2:
                str = stemmer.stem(str)
                times = tokenappear.get(str, 0);
                times = times + 1
                tokenappear[str] = times

    maxlog = -100000000.0
    rescat = ""
    
    # Now we calculate the tokenpercent_NB for each class 
    for tmpcat in categorys: 
        tmplog = math.log(tokendata_NB[tmpcat][u'_percent'])
        for word, count in tokenappear.items():
            icount = 0
            wordappear = tokendata_NB[tmpcat].get(word, 0)
            logforword = 0
            
            if wordappear == 0:
                logforword = 1.0 / (tokendata_NB[tmpcat][u'_num'] + alphabet)
            else:
                logforword = tokenpercent_NB[tmpcat][word]
            while icount < count:
                tmplog = tmplog + math.log(logforword)
                icount = icount + 1

        if tmplog > maxlog:
            maxlog = tmplog
            rescat = tmpcat
    
    if rescat == category:
        return True
    return False


def Classify_TFIDF(filename, category):
    filehandle = open(filename, encoding='utf-8')
    filelines = filehandle.readlines()
    tokenlist = {}  
    
    for line in filelines:
        strings = regx.split(line)
        for str in strings:
            if len(str) > 2:
                str = stemmer.stem(str)
                tokenlist[str] = 1

    maxlog = -100000000.0
    rescat = ""
    
    # Now we calculate the tokenpercent_NB for each class 
    for tmpcat in categorys: 
        tmpres = 0
        for word, count in tokenlist.items():
            tfidf = token_tfidf[tmpcat].get(word, 0) 
            tmpres = tmpres + tfidf     

        if tmpres > maxlog:
            maxlog = tmpres
            rescat = tmpcat
    
    if rescat == category:
        return True
    return False


def TestData(dirpath):
    # Test using Native Bayes 
    print ("Start to test data using Native Bayes")
    for tmpcat in categorys: 
        testpath = dirpath + "\\test\\" + tmpcat + "\\"
        file_num = 0
        rightnum = 0
        
        for fn in os.listdir(testpath):
            file_num = file_num + 1
            fullname = testpath + fn
            if Classify_NB(fullname, tmpcat):
                rightnum = rightnum + 1
        print ("Testing class [" + tmpcat + "], the accuracy [Native Bayes] is: " + str(rightnum * 100.0 / file_num))
    
    # Test using TF-IDF
    print ("Start to test data using TF-IDF")
    for tmpcat in categorys: 
        testpath = dirpath + "\\test\\" + tmpcat + "\\"
        file_num = 0
        rightnum = 0
        
        for fn in os.listdir(testpath):
            file_num = file_num + 1
            fullname = testpath + fn
            if Classify_TFIDF(fullname, tmpcat):
                rightnum = rightnum + 1
        print ("Testing class [" + tmpcat + "], the accuracy [TF-IDF] is: " + str(rightnum * 100.0 / file_num))


if len(sys.argv) != 2:
    print ("Usage: python sentiment.py datadir")
    sys.exit(0)

TrainData(sys.argv[1])
TestData(sys.argv[1])

