import os
from os import listdir
from os.path import isfile, join


data = "NP.txt" 
### print possible datasets ###
if os.path.exists('./dbase'):
    cwd = os.getcwd()
    datasets = [f for f in listdir('./dbase/raw') if isfile(join('./dbase/raw/', f))]
    print('\n'+'#'*50+'\n\n'"Possible datasets are:\n")
    for f in datasets:
        print('"'+cwd+'/dbase/raw/'+f+'"')
    print('\n'+'#'*50+'\n')

### choose parameters, run autoencoder ###
data = input("Please enter the path to the desired dataset. Possibilities are listed above\n")
encoded_data = "/home/pgian1/practice/csc534/project/dbase/encoded/" + data[46:-4] + "_encoded.txt"
encoded = []
with open(data) as dataReader:
    for line in dataReader:
        encoded_tran = ""
        for base in line:
            if base == 'A' or base == 'a':
                encoded_tran = encoded_tran + '0 '
            if base == 'T' or base == 't':
                encoded_tran = encoded_tran + '1 '
            if base == 'C' or base == 'c':
                encoded_tran = encoded_tran + '2 '
            if base == 'G' or base == 'g':
                encoded_tran = encoded_tran + '3 '
        encoded.append(encoded_tran)
with open(encoded_data, 'w') as encodedReader:
    for transcript in encoded:
        encodedReader.write(transcript+'\n')
