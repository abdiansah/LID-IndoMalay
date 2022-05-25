import pickle
import pandas as pd
import re

def prediksi(kalimat, model_name):
    f_model = '../../datasets/'+model_name+'.model'
    # load model
    file = open(f_model, 'rb')
    vocab = pickle.load(file)
    model = pickle.load(file)

    kalimat = kalimat.lower()
    kalimat = re.sub(r'[^a-zA-Z ]+', '', kalimat)
    kalimat = re.sub(r'\b\w{,1}\b', '', kalimat)
    kalimat = kalimat.strip()
    kalimat = re.sub(r' +', ' ', kalimat)

    d = {'teks': [kalimat]}
    data = pd.DataFrame(data=d)

    data_vect = vocab.transform(data['teks'])
    result = model.predict(data_vect)
    label = 'INDONESIA' if result==0 else 'MALAYSIA'
    
    return label


###=================== MAIN ===================###

print('\nPREDIKSI KALIMAT BAHASA INDONESIA ATAU MALAYSIA')
kalimat = ''
while 1:
    kalimat = input('\n...Masukan Kalimat\t: ')
    if kalimat == 'xxx':
        print('\n   Terima Kasih')
        break
    else:
        print('...Hasil Prediksi\t:',prediksi(kalimat, 'NB'))
        print('\n   Masukan \'xxx\' untuk keluar')
print()
