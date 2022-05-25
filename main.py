# tutorial: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
# beberapa kode pra-pengolahan tidak digunakan
# fungsi TfidfVectorizer tidak bisa untuk data yang ditokenisasi

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import naive_bayes
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import pickle
import re

def load_dataset(f_dataset):
    # muat dataset
    data = pd.read_csv(f_dataset, encoding='latin-1')
    #print(data.head())
    return data

def remove_closewords(f_closewords, data):
    with open(f_closewords, 'r', encoding='latin-1') as f:
        data_closewords = f.read()
    data_closewords = data_closewords.split()
    #print(data_closewords)
    pattern = '('
    for dc in data_closewords:
        pattern += dc + '|'
    pattern += '1)'
    #print(pattern)
    compiled_words = re.compile(r'\b' + pattern + r'\b')
    data = [re.sub(compiled_words, '', row) for row in data]
    data = [re.sub(r' +', ' ', row) for row in data]
    return data

def pre_processing(data, f_closewords):
    #pra-pengolahan
    teks_clean = [re.sub(r'\b\w{,1}\b', '', ts) for ts in data['teks']]
    teks_clean = [tc.strip() for tc in teks_clean]
    teks_clean = [re.sub(r' +', ' ', tc) for tc in teks_clean]
    teks_clean = [row.lower() for row in teks_clean]            # lowercase - default

    #teks_clean = remove_closewords(f_closewords, teks_clean)    # remove closewords #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    data['teks'] = teks_clean

    return data

def split_and_encode(data):
    # split dataset
    train_x, test_x, train_y, test_y = model_selection.train_test_split(data['teks'], data['kategori'], test_size=0.2)
    #print(train_y.head())

    # ubah nilai kelas (kategori) menjadi numerik
    encoder = LabelEncoder()
    train_y_vect = encoder.fit_transform(train_y)
    test_y_vect = encoder.fit_transform(test_y)
    #print(train_y_vect)
    return train_x, test_x, train_y, test_y, train_y_vect, test_y_vect

def vector_space_model(data, train_x, test_x):
    # buat vektor data untuk train dan test (tf-idf)
    tfidf_vect = TfidfVectorizer(max_features=5000)
    tfidf_vect.fit(data['teks'])
    train_x_tfidf = tfidf_vect.transform(train_x)
    test_x_tfidf = tfidf_vect.transform(test_x)
    #print(tfidf_vect.vocabulary_)
    #print(train_x_tfidf)
    return train_x_tfidf, test_x_tfidf, tfidf_vect

def train_classifier(model_name, train_x_vect, train_y_vect, vocab, f_model):   # split 80:20 (train:test)
    if model_name=='NB':
        # model naive-bayes
        model = naive_bayes.MultinomialNB()
    elif model_name=='SVM':
        # model svm
        model = svm.SVC(C=1, kernel='rbf') # linear kernel and C values in [1, 10, 100, 1000]
        #model = svm.NuSVC(kernel='linear', gamma='auto')
        
    # train model
    model.fit(train_x_vect, train_y_vect)
    
    # cross-validation
    cv = cross_val_score(model, test_x_vect, test_y_vect, cv=10)
    print('\nINFORMASI PELATIHAN MACHINE LEARNING')
    print('... Model ML\t\t: ',model_name)
    print('... 10-Cross-Validation\t: ', cv)
    print('... Rata-Rata Akurasi\t: ',sum(cv)/len(cv)*100 ,' %\n')

    # save model
    file = open(f_model, 'wb')
    pickle.dump(vocab, file)
    pickle.dump(model, file)

def test_classifier(model_name, test_x_vect, test_y_vect, f_model):
    # load model
    file = open(f_model, 'rb')
    vocab = pickle.load(file)
    model = pickle.load(file)

    # predict model
    predict = model.predict(test_x_vect)
    # report model
    print('\nINFORMASI PENGUJIAN MACHINE LEARNING')
    print('... Model ML\t\t: ',model_name)
    print('... Akurasi {}\t\t: '.format(model_name), accuracy_score(predict, test_y_vect)*100, '%\n')
    #target_names = ['IND', 'MSA']
    #print(classification_report(test_y_vect, predict, target_names=target_names))

def predict_classifier(data, f_model):
    # load model
    file = open(f_model, 'rb')
    vocab = pickle.load(file)
    model = pickle.load(file)
    data_vect = vocab.transform(data)
    result = model.predict(data_vect)
    return result

###================ MAIN ================###

np.random.seed(500)

###================ TRAIN & SAVE MODEL (DATASET SAMA)

f_dataset = '../../datasets/ds-ind-msa-10K.csv'
f_closewords = '../../datasets/close-words-10K.txt'
data = load_dataset(f_dataset)
#print(data.head())

data = pre_processing(data, f_closewords)
#print(data.head())

train_x, test_x, train_y, test_y, train_y_vect, test_y_vect = split_and_encode(data)
train_x_vect, test_x_vect, vocab = vector_space_model(data, train_x, test_x)

model_name = 'NB' #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

f_model = '../../datasets/{}.model'.format(model_name)

# train
train_classifier(model_name, train_x_vect, train_y_vect, vocab, f_model)

# test
test_classifier(model_name, train_x_vect, train_y_vect, f_model) # pengujian data latih (80)
test_classifier(model_name, test_x_vect, test_y_vect, f_model) # pengujian data uji (20)


###================ PREDICT MODEL (DATASET BEDA) - pindah ke prediksi.py

""" print('\n... Proses prediksi bahasa Indonesia-Malaysia')

f_dataset = '../../datasets/ds-ferdiana2.csv'
f_closewords = '../../datasets/close-words-10K.txt'

data = load_dataset(f_dataset)
data = pre_processing(data, f_closewords)

model_name = 'NB'

f_model = '../../datasets/'+model_name+'.model'
result = predict_classifier(data['teks'], f_model)

label = [('INDONESIA' if r==0 else 'MALAYSIA') for r in result]
data['prediksi '+model_name] = label

f_prediksi = '../../datasets/hasil-prediksi-'+model_name+'.csv'
data.to_csv(f_prediksi, index=False)

print('    >>> Prediksi selesai')
print('    >>> Hasil prediksi: ', f_prediksi)
print() """

#print(data.head())