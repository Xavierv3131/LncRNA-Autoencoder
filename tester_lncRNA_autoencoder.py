import os
from keras.models import load_model  # Will be used to retrieve the saved weights of the trained NN.
import numpy as np

import logging
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lncEncoder_200 = load_model(r'./weights/encoder_weights_200_.h5')
lncDecoder_200 = load_model(r'./weights/decoder_weights_200_.h5')
codEncoder_200 = load_model(r'./weights/encoder_weights_200_.h5')
codDecoder_200 = load_model(r'./weights/decoder_weights_200_.h5')

lncEncoder_500 = load_model(r'./weights/encoder_weights_500_.h5')
lncDecoder_500 = load_model(r'./weights/decoder_weights_500_.h5')
codEncoder_500 = load_model(r'./weights/encoder_weights_500_.h5')
codDecoder_500 = load_model(r'./weights/decoder_weights_500_.h5')

lncEncoder_1000 = load_model(r'./weights/encoder_weights_1000_.h5')
lncDecoder_1000 = load_model(r'./weights/decoder_weights_1000_.h5')
codEncoder_1000 = load_model(r'./weights/encoder_weights_1000_.h5')
codDecoder_1000 = load_model(r'./weights/decoder_weights_1000_.h5')

diff=[]
lncData200=[]
codData200=[]
lncData500=[]
codData500=[]
lncData1000=[]
codData1000=[]

def MSE(tran, recon, size):   #calculates the MSE
    """calculates the Mean Square Error between the original and
       reconstructed transcript

    Parameters
    ----------
    tran: list, required
        The original transcript
    recon: list, required
        The reconstructed transcript
    size: int, required
        The length of the transcripts

    Returns
    -------
    mse: str
        The Mean Square error in string format
    """

    del diff[:] #holds the differences
    i=0
    while i < size: #for all bases
        diff.append(tran[i] - recon[i]) #calculate diff between actual and reconstructed
        i=i+1
    error = 0
    for num in diff:
        error = error + (num*num) #sum squares 
    mse = error/size
    return(str(mse))

if __name__ == '__main__':

    if not os.path.exists(r'./errors'):
        os.mkdir(r'./errors')

### Process Data ###
    print("\nProcessing Data\n")
    with open('/home/pgian1/practice/csc534/project/dbase/test/200_2000_lncs_encoded.txt') as lfh:
            lines = lfh.readlines()
            for line in lines:
                line = line.strip('\n')
                rna = line.split()
                length = len(rna)
                if(length >= 200 and length < 500):
                    rna = [int(i) for i in rna]
                    rna = rna[:200]
                    lncData200.append(rna)
                if(length >= 500 and length < 1000):
                    rna = [int(i) for i in rna]
                    rna = rna[:500]
                    lncData500.append(rna)
                if(length >= 1000 and length <= 2000):
                    rna = [int(i) for i in rna]
                    rna = rna[:1000]
                    lncData1000.append(rna)
    lfh.close()

    with open('/home/pgian1/practice/csc534/project/dbase/test/200_2000_coding_encoded.txt') as cfh:
            lines = cfh.readlines()
            for line in lines:
                line = line.strip('\n')
                rna = line.split()
                length = len(rna)
                if(length >= 200 and length < 500):
                    rna = [int(i) for i in rna]
                    rna = rna[:200]
                    codData200.append(rna)
                if(length >= 500 and length < 1000):
                    rna = [int(i) for i in rna]
                    rna = rna[:500]
                    codData500.append(rna)
                if(length >= 1000 and length <= 2000):
                    rna = [int(i) for i in rna]
                    rna = rna[:1000]
                    codData1000.append(rna)
    cfh.close()
###############################=-200-499 nucleotides-=###############################
    print('\n'+'#'*22+"=-200-499 nucleotides-="+'#'*22+'\n')
    print('\n'+'#'+"="*65+'#')
    print("Loading model trained on lncRNAs in range 200-499 nucleotides")
    print('#'+"="*65+'#'+'\n')
    print("Testing model on actual lncRNAs in range 200-499 nucleotides")
    with open ("./errors/499lnc_499lncModel.txt", 'w') as lefh:
        lnc_inputs = np.array(lncData200)
        lnc_recons = lncDecoder_200.predict(lncEncoder_200.predict(lnc_inputs))
        print("...Loading")

        n=0
        while n < len(lnc_inputs):
            lefh.write(MSE(lnc_inputs[n],lnc_recons[n],200)+'\n')
            n=n+1
    lefh.close()
    print("_"*67+'\n')
    print("Testing model on coding RNAs in range 200-499 nucleotides\n")
    with open ("./errors/499cod_499lncModel.txt", 'w') as cefh:
        cod_inputs = np.array(codData200)
        cod_recons = lncDecoder_200.predict(lncEncoder_200.predict(cod_inputs))
        print("Loading...")

        n=0
        while n < len(cod_inputs):
            cefh.write(MSE(cod_inputs[n],cod_recons[n],200)+'\n')
            n=n+1
    cefh.close()
    print('#'+"="*65+'#'+'\n')
#################################################
    print('\n'+'#'+"="*65+'#')
    print("Loading model trained on coding RNAs in range 200-499 nucleotides")
    print('#'+"="*65+'#'+'\n')
    print("Testing model on actual lncRNAs in range 200-499 nucleotides")
    with open ("./errors/499lnc_499codModel.txt", 'w') as lefh:
        lnc_inputs = np.array(lncData200)
        lnc_recons = codDecoder_200.predict(codEncoder_200.predict(lnc_inputs))
        print("Loading...")

        n=0
        while n < len(lnc_inputs):
            lefh.write(MSE(lnc_inputs[n],lnc_recons[n],200)+'\n')
            n=n+1
    lefh.close()
    print("_"*67+'\n')
    print("Testing model on coding RNAs in range 200-499 nucleotides\n")
    with open ("./errors/499cod_499codModel.txt", 'w') as cefh:
        cod_inputs = np.array(codData200)
        cod_recons = codDecoder_200.predict(codEncoder_200.predict(cod_inputs))
        print("Loading...")

        n=0
        while n < len(cod_inputs):
            cefh.write(MSE(cod_inputs[n],cod_recons[n],200)+'\n')
            n=n+1
    cefh.close()
    print('#'+"="*65+'#'+'\n')
###############################=-500-999 nucleotides-=###############################
    print('\n'+'#'*22+"=-500-999 nucleotides-="+'#'*22+'\n')
    print('\n'+'#'+"="*65+'#')
    print("Loading model trained on lncRNAs in range 500-999 nucleotides")
    print('#'+"="*65+'#'+'\n')
    print("Testing model on actual lncRNAs in range 500-999 nucleotides")
    with open ("./errors/999lnc_999lncModel.txt", 'w') as lefh:
        lnc_inputs = np.array(lncData500)
        lnc_recons = lncDecoder_500.predict(lncEncoder_500.predict(lnc_inputs))
        print("Loading...")

        n=0
        while n < len(lnc_inputs):
            lefh.write(MSE(lnc_inputs[n],lnc_recons[n],500)+'\n')
            n=n+1
    lefh.close()
    print("_"*67+'\n')
    print("Testing model on coding RNAs in range 500-999 nucleotides\n")
    with open ("./errors/999cod_999lncModel.txt", 'w') as cefh:
        cod_inputs = np.array(codData500)
        cod_recons = lncDecoder_500.predict(lncEncoder_500.predict(cod_inputs))
        print("Loading...")

        n=0
        while n < len(cod_inputs):
            cefh.write(MSE(cod_inputs[n],cod_recons[n],500)+'\n')
            n=n+1
    cefh.close()
    print('#'+"="*65+'#'+'\n')
#################################################
    print('\n'+'#'+"="*65+'#')
    print("Loading model trained on coding RNAs in range 500-999 nucleotides")
    print('#'+"="*65+'#'+'\n')
    print("Testing model on actual lncRNAs in range 500-999 nucleotides")
    with open ("./errors/999lnc_999codModel.txt", 'w') as lefh:
        lnc_inputs = np.array(lncData500)
        lnc_recons = codDecoder_500.predict(codEncoder_500.predict(lnc_inputs))
        print("Loading...")

        n=0
        while n < len(lnc_inputs):
            lefh.write(MSE(lnc_inputs[n],lnc_recons[n],500)+'\n')
            n=n+1
    lefh.close()
    print("_"*67+'\n')
    print("Testing model on coding RNAs in range 500-999 nucleotides\n")
    with open ("./errors/999cod_999codModel.txt", 'w') as cefh:
        cod_inputs = np.array(codData500)
        cod_recons = codDecoder_500.predict(codEncoder_500.predict(cod_inputs))
        print("Loading...")

        n=0
        while n < len(cod_inputs):
            cefh.write(MSE(cod_inputs[n],cod_recons[n],500)+'\n')
            n=n+1
    cefh.close()
    print('#'+"="*65+'#'+'\n')
###############################=-1000-2000 nucleotides-=###############################
    print('\n'+'#'*22+"=-1000-2000 nucleotides-="+'#'*22+'\n')
    print('\n'+'#'+"="*65+'#')
    print("Loading model trained on lncRNAs in range 1000-2000 nucleotides")
    print('#'+"="*65+'#'+'\n')
    print("Testing model on actual lncRNAs in range 1000-2000 nucleotides")
    with open ("./errors/2000lnc_2000lncModel.txt", 'w') as lefh:
        lnc_inputs = np.array(lncData1000)
        lnc_recons = lncDecoder_1000.predict(lncEncoder_1000.predict(lnc_inputs))
        print("Loading...")

        n=0
        while n < len(lnc_inputs):
            lefh.write(MSE(lnc_inputs[n],lnc_recons[n],1000)+'\n')
            n=n+1
    lefh.close()
    print("_"*67+'\n')
    print("Testing model on coding RNAs in range 1000-2000 nucleotides\n")
    with open ("./errors/2000cod_2000lncModel.txt", 'w') as cefh:
        cod_inputs = np.array(codData1000)
        cod_recons = lncDecoder_1000.predict(lncEncoder_1000.predict(cod_inputs))
        print("Loading...")

        n=0
        while n < len(cod_inputs):
            cefh.write(MSE(cod_inputs[n],cod_recons[n],1000)+'\n')
            n=n+1
    cefh.close()
    print('#'+"="*65+'#'+'\n')
#################################################
    print('\n'+'#'+"="*65+'#')
    print("Loading model trained on coding RNAs in range 1000-2000 nucleotides")
    print('#'+"="*65+'#'+'\n')
    print("Testing model on actual lncRNAs in range 1000-2000 nucleotides")
    with open ("./errors/2000lnc_2000codModel.txt", 'w') as lefh:
        lnc_inputs = np.array(lncData1000)
        lnc_recons = codDecoder_1000.predict(codEncoder_1000.predict(lnc_inputs))
        print("Loading...")

        n=0
        while n < len(lnc_inputs):
            lefh.write(MSE(lnc_inputs[n],lnc_recons[n],1000)+'\n')
            n=n+1
    lefh.close()
    print("_"*67+'\n')
    print("Testing model on coding RNAs in range 1000-2000 nucleotides\n")
    with open ("./errors/2000cod_2000codModel.txt", 'w') as cefh:
        cod_inputs = np.array(codData1000)
        cod_recons = codDecoder_1000.predict(codEncoder_1000.predict(cod_inputs))    # predict the reconstruction
        print("Loading...")

        n=0
        while n < len(cod_inputs):
            cefh.write(MSE(cod_inputs[n],cod_recons[n],1000)+'\n')
            n=n+1
    cefh.close()
   print('#'+"="*65+'#'+'\n')
print("\n...Finished testing!")
