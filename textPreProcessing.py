# input arg: /Users/shaye/Code/python/textPre_lda/testArticle
import sys , os
import string
import nltk
import re
import numpy, scipy
import sklearn

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def doc_cleanup(doc):
    # remove every thing except english words, then remove stopwords
    text = doc.read()
    text_letters_only = re.sub("[^a-zA-Z]"," ",text)
    text_letters_lower = text_letters_only.lower()
    words_list = text_letters_lower.split()
    stopword_set = set(stopwords.words("english"))
    cleanwords_list = [w for w in words_list if not w in stopword_set]
    words = " ".join( cleanwords_list )   

    return(words)



def save_clean_docs(input_folder,output_folder):
    output_foldername=os.path.basename(os.path.normpath(output_folder))#=only last part of path
    all_docs_output_file=open(output_folder+"All","w")
    for item in os.listdir(input_folder):
        if item == '.DS_Store' or item == output_foldername:
            continue
        output_file = open(output_folder+item+"c","w")
        in_file = open(input_folder+item,'r')
        words = doc_cleanup(in_file)
        doc_length = len(words)
        output_file.write( "%s"% words)
        all_docs_output_file.write("%s"% words+'\n')
        in_file.close()
        output_file.close()
        
    all_docs_output_file.close()
    
    
        
def get_input_output_folders(in_arg):
    input_folder = in_arg[1]
    if not input_folder.endswith(os.sep):   #if input folder path doesn't end with '/'
        input_folder = input_folder + os.sep   #put a '/' at the end of input folder path
    if len(in_arg) < 3:
        if not os.path.exists(input_folder+'cleanDocs/'):
           os.mkdir(input_folder+'cleanDocs/')
           output_folder=input_folder+'cleanDocs/'
        else:
            output_folder=input_folder+'cleanDocs/'
    else:
        output_folder = in_arg[2]
    return(input_folder,output_folder)



def get_bagwords(all_docs_file):
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer=CountVectorizer(analyzer="word",tokenizer = None,\
                               preprocessor = None,stop_words = None,\
                               max_features = None)
    all_docs=open(all_docs_file,'r')
    docs_text_aslist = all_docs.readlines()
    train_data_features = vectorizer.fit_transform(docs_text_aslist)
    train_data_features_array = train_data_features.toarray()
    return(train_data_features_array)

    

def create_dictionary(doc_filepath):
    offset = 1 #Default=1:term ids start from 0, 0:term ids start from 1
    doc_file=open(doc_filepath,'r')
    doc_text=doc_file.read()
    dictnry=dict()
    count=0-offset
    for word in doc_text.split():
        if word not in dictnry:
            count += 1
            dictnry[word] = count
        #print ("%s %d" % (word,dictnry[word]))        
    return (dictnry)
        

def create_word_count_pairs(doc_filepath,dictnry):
    doc_file = open(doc_filepath,'r')
    #doc_text=doc_file.read()
    doc_folder = os.path.normpath(doc_filepath + os.sep + os.pardir)+'/'#go one directory up
    doc_word_count_pair = open(doc_folder+'wordID_count','w')
    for line in doc_file:
        linedict_counts = dict() #dict for each line
        for word in line.split():
            if word not in linedict_counts:
                linedict_counts[word] = 1
            else:
                linedict_counts[word] +=1
        for word in linedict_counts:        
            if word in dictnry:
                word_code = dictnry[word]
                #print word_code,':',linedict_counts[word]
                doc_word_count_pair.write("%d%s%d "%(word_code,':',linedict_counts[word]))
        doc_word_count_pair.write('\n')

    return()
            


def main(inputs):   
    input_folder, output_folder = get_input_output_folders(inputs)
    save_clean_docs(input_folder,output_folder)
    doc_filepath = output_folder+'All'
    dictnry=create_dictionary(doc_filepath)
    create_word_count_pairs(doc_filepath,dictnry)
    #bagwords = get_bagwords(all_docs_file)
    print 'end of textPreProcessing'
            
if __name__ == '__main__':
    inputs = sys.argv
    main(inputs)            
