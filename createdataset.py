import re
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk import FreqDist

import matplotlib.pyplot as plt

def show_list(data):
    i = 1
    for d in data:
        print(i,'. ',d)
        i+=1

def create_sub_corpus(file_korpus_ori, row, file_korpus):
    with open(file_korpus_ori, 'r', encoding='latin-1') as korpus:
        teks_korpus = korpus.read()
        teks_korpus = teks_korpus.split('\n')
        dump = []
        for i in range(0, row):
            if i == row-1:
                dump.append(teks_korpus[i])
            else:
                dump.append(teks_korpus[i]+'\n')
    
    with open(file_korpus, 'w') as file_txt:
        file_txt.writelines(dump)
        file_txt.flush()

def clean_corpus(teks_korpus):
    teks_split = teks_korpus.split('\n')
    #show_list(teks_split)

    teks_clean = [re.sub(r'[^a-zA-Z ]+', '', ts) for ts in teks_split]
    teks_clean = [tc.strip() for tc in teks_clean]
    teks_clean = [re.sub(r' +', ' ', tc) for tc in teks_clean]
    teks_clean = [tc+'\n' for tc in teks_clean]
    #show_list(teks_clean)

    return teks_clean

def create_dataset(file_korpus, label):
    with open(file_korpus, 'r', encoding='latin-1') as korpus:
        teks_korpus = korpus.read()
        #print(teks_korpus)

    teks_clean = clean_corpus(teks_korpus)

    # BELUM PERLU
    # with open('../../datasets/tes-ind.csv', 'w') as file_csv:
    #     file_csv.write('teks\n')  # kolom teks
    #     file_csv.writelines(teks_clean)
    #dataset = pd.read_csv('../../datasets/tes-ind.csv', encoding='latin-1')

    dataset = pd.DataFrame(teks_clean, columns=['teks'])
    #print(dataset.head())
    dataset['kategori'] = label
    dataset['id'] = np.random.randint(1, 1000000, dataset.shape[0])
    #print(dataset.head())
    return dataset

def merger_dataset(ds1, ds2, file_csv):
    ds = ds1
    ds = ds.append(ds2, ignore_index = True)
    #print(ds)
    ds = ds.sort_values(['id'])
    
    n = len(ds['id'])
    ds['id'] = [i+1 for i in range(0, n)]
    #print(ds.head())
    
    ds = ds.reindex(columns=['id', 'teks', 'kategori'])
    #print(ds.head())    
    
    ds.to_csv(file_csv, index=False)

    return ds

def create_stopword(ds, top_n, file_sw):   # stopword indonesia dan malaysia digabung, beda dgn common word
    teks_data = ds['teks'].values.tolist()
    teks_str = ' '.join(map(str, teks_data))
    teks_token = word_tokenize(teks_str)
    teks_token = [t.lower() for t in teks_token]
    fdist = FreqDist(teks_token)
    stopword = fdist.most_common(top_n)
    with open(file_sw, 'w') as filex:
        for w,v in stopword:
            filex.write(w+'\n')
            filex.flush()

def create_commonwords(f_korpus_teks, top_n, file_sw):
    with open(f_korpus_teks, 'r', encoding='latin-1') as korpus:
        teks_korpus = korpus.read()
        #print(teks_korpus)
    teks_clean = clean_corpus(teks_korpus)
    teks_str = ' '.join(map(str, teks_clean))
    teks_token = word_tokenize(teks_str)
    teks_token = [t.lower() for t in teks_token]
    fdist = FreqDist(teks_token)
    commonwords = fdist.most_common(top_n)
    with open(file_sw, 'w') as filex:
        for w,v in commonwords:
            filex.write(w+'\n')
            filex.flush()

def create_closeword(f_korpus_ind, f_korpus_msa, f_close_words):
    with open(f_korpus_ind, 'r', encoding='latin-1') as korpus:
        teks_korpus = korpus.read()
    teks_clean = clean_corpus(teks_korpus)
    teks_str = ' '.join(map(str, teks_clean))
    token_ind = word_tokenize(teks_str)
    token_ind = [re.sub(r'\b\w{,1}\b', '', ts) for ts in token_ind]
    token_ind = [t.lower()+'\n' for t in token_ind]

    with open(f_korpus_msa, 'r', encoding='latin-1') as korpus:
        teks_korpus = korpus.read()
    teks_clean = clean_corpus(teks_korpus)
    teks_str = ' '.join(map(str, teks_clean))
    token_msa = word_tokenize(teks_str)
    token_msa = [re.sub(r'\b\w{,1}\b', '', ts) for ts in token_msa]
    token_msa = [t.lower()+'\n' for t in token_msa]
    
    closewords = list(set(token_ind) & set(token_msa))

    with open(f_close_words, 'w') as file_txt:
        file_txt.writelines(closewords)
        file_txt.flush()


#####''' MAIN PROGRAM '''#####

print('\n... Info korpus original Indonesia dan Malaysia')
f_korpus_asli_ind = '../../datasets/uni-leipzig/ind-id_web_2013_1M/ind-id_web_2013_1M-sentences.txt'  # 118 MB
f_korpus_asli_msa = '../../datasets/uni-leipzig/msa-my_web_2013_1M/msa-my_web_2013_1M-sentences.txt'  # 131 MB
print('    >>> Korpus Indonesia: ', f_korpus_asli_ind)
print('    >>> Korpus Malaysia: ', f_korpus_asli_msa)

print('\n... Proses pembuatan sub korpus Indonesia dan Malaysia')
f_korpus_teks_ind = '../../datasets/ind-5K.txt'
f_korpus_teks_msa = '../../datasets/msa-5K.txt'
maks_baris = 5000 #5K

create_sub_corpus(f_korpus_asli_ind, maks_baris, f_korpus_teks_ind)
create_sub_corpus(f_korpus_asli_msa, maks_baris, f_korpus_teks_msa)
print('    >>> File korpus Indonesia dan Malaysia berhasil dibuat.')

print('\n... Proses pembuatan dataset Indonesia-Malaysia')
f_dataset_ind_msa = '../../datasets/ds-ind-msa-10K.csv'
dataset_ind = create_dataset(f_korpus_teks_ind, 'Indonesia')
dataset_msa = create_dataset(f_korpus_teks_msa, 'Malaysia')
dataset = merger_dataset(dataset_ind, dataset_msa, f_dataset_ind_msa)
print('    >>> Dataset Indonesia dan Malaysia berhasil dibuat. ')
print('    >>> Lokasi file: ', f_dataset_ind_msa)

print('\n... Proses pembuatan stopword Indonesia-Malaysia')
f_stopword_ind_msa = '../../datasets/stopword-ind-msa-10K.txt'
create_stopword(dataset, 30, f_stopword_ind_msa)
print('    >>> File stopword Indonesia dan Malaysia berhasil dibuat. ')
print('    >>> Lokasi file: ', f_stopword_ind_msa)

print('\n... Proses pembuatan common-words Indonesia dan Malaysia')
f_common_words_ind = '../../datasets/common-words-ind-10K.txt'
f_common_words_msa = '../../datasets/common-words-msa-10K.txt'
create_commonwords(f_korpus_teks_ind, 30, f_common_words_ind)
create_commonwords(f_korpus_teks_msa, 30, f_common_words_msa)
print('    >>> File common-words Indonesia berhasil dibuat. ')
print('    >>> Lokasi file common-word Indonesia: ', f_common_words_ind)
print('    >>> Lokasi file common-word Malaysia: ', f_common_words_msa)

print('\n... Proses pembuatan close-words Indonesia dan Malaysia')
f_close_words = '../../datasets/close-words-10K.txt'
create_closeword(f_korpus_teks_ind, f_korpus_teks_msa, f_close_words)
print('    >>> File close-words Indonesia Malaysia berhasil dibuat. ')
print('    >>> Lokasi file close-words: ', f_close_words)

print('\r')