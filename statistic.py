import numpy as np
import re
import collections
from nltk import word_tokenize
import matplotlib.pyplot as plt

def show_list(data):
    i = 1
    for d in data:
        print(i,'.',d)
        i+=1

def clean_corpus(teks_korpus):
    teks_korpus = teks_korpus.lower()
    teks_split = teks_korpus.split('\n')
    #show_list(teks_split)
    
    teks_clean = [re.sub(r'[^a-zA-Z ]+', '', ts) for ts in teks_split]
    teks_clean = [re.sub(r'\b\w{,3}\b', '', ts) for ts in teks_clean]
    teks_clean = [tc.strip() for tc in teks_clean]
    teks_clean = [re.sub(r' +', ' ', tc) for tc in teks_clean]
    teks_clean = [tc+'\n' for tc in teks_clean]
    teks = '' 
    teks = teks.join(teks_clean)

    return teks
    
def statistik_korpus(f_korpus):
    with open(f_korpus) as f:
        korpus = f.read()

    korpus = clean_corpus(korpus)
    korpus = korpus.split()
    print('Jumlah Token\t: ', len(set(korpus)))

    vmin = 1000
    vmax = 0
    for k in korpus:
        if len(k) < vmin:
            vmin = len(k)
        if len(k) > vmax:
            vmax = len(k)

    print('Jumlah Kalimat Terpendek\t: ', vmin)
    print('Jumlah Kalimat Terpanjang\t: ', vmax)

def common_words(korpus, country, top_n):            # korpus
    wordcount = collections.defaultdict(int)
    pattern = r"\W"
    for word in korpus.lower().split():
        word = re.sub(pattern, '', word)
        wordcount[word] += 1

    mc = sorted(wordcount.items(), key=lambda k_v: k_v[1], reverse=True)[:top_n]

    mc = dict(mc)
    names = list(mc.keys())
    values = list(mc.values())
    plt.style.use('ggplot')
    plt.bar(range(len(mc)),values,tick_label=names, color = (0.5,0.1,0.5,0.6))

    name = 'INDONESIAN' if country=='IND' else 'MALAYSIAN'

    plt.title('{} TOP OF {} WORDS'.format(top_n, name))
    plt.xlabel('WORDS')
    plt.ylabel('FREQUENCIES')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('../../datasets/{}.png'.format(name), dpi = 300)
    plt.show()

def close_words(indkorpus, msakorpus, top_n):
    ind_token = word_tokenize(indkorpus)
    msa_token = word_tokenize(msakorpus)
    closewords = list(set(ind_token) & set(msa_token))

    data = {}
    for cw in closewords:
        indfreq = re.findall(cw, indkorpus)
        msafreq = re.findall(cw, msakorpus)
        data[cw] = len(indfreq+msafreq)
    #print(data)

    mc = sorted(data.items(), key=lambda item: item[1], reverse=True)[:top_n]
    mc = dict(mc)
    names = list(mc.keys())
    values = list(mc.values())
    plt.style.use('ggplot')
    plt.bar(range(len(mc)),values,tick_label=names, color = (0.5,0.1,0.5,0.6))

    plt.title('{} TOP OF INDONESIAN-MALAYSIAN CLOSE WORD'.format(top_n))
    plt.xlabel('WORDS')
    plt.ylabel('FREQUENCIES')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('../../datasets/close-word.png', dpi = 300)
    plt.show()

    return closewords


###======================== MAIN ========================###

np.random.seed(500)

f_indkorpus = '../../datasets/ind-5K.txt'
f_msakorpus = '../../datasets/msa-5K.txt'

statistik_korpus(f_indkorpus)
statistik_korpus(f_msakorpus)

with open(f_indkorpus, 'r') as f:
    korpus = f.read()
    indkorpus = clean_corpus(korpus)

with open(f_msakorpus, 'r') as f:
    korpus = f.read()
    msakorpus = clean_corpus(korpus)

common_words(indkorpus, 'IND', 20)
common_words(msakorpus, 'MSA', 20)

closew = close_words(indkorpus, msakorpus, 20)
print(closew)