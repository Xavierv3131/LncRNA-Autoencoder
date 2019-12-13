import sys
import os
import logging
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np
from os import listdir
from os.path import isfile, join
import logging                                      #|
logging.getLogger('tensorflow').disabled = True     #|This hides those pesky warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            #|
###############################################################################################
#
# PURPOSE:  To generate an autoencoder that recognizes abstract features of lncRNAs. Known lncRNA transcripts and known coding transcripts.  This
#           model will be used to predict if any given transcript can be classified as a 
#           lncRNA or not
#
# AUTHORS:      Peter Giangrasso, Xavier Vogel, Rebecca Smith
# ADAPTED FROM: https://www.youtube.com/watch?v=uCaPP4blYAg
# DATE:         12.12.19
#
###############################################################################################

dataPath = ''
data1    = []
data2    = []
data3    = []


class AutoEncoder:
    """
    A class used to generate an RNA transcript autoencoder

    ...

    Attributes
    ----------
    dataPath: str
        a string indicating the location of the training data
    data1 : list
        a list containing transcripts between 200-499bp, chopped at 200bp
    data2 : list
        a list containing transcripts between 500-999bp, chopped at 500bp
    data3 : list
        a list containing transcripts between 1000-2000bp, chopped at 1000bp
    dataset : list
        the dataset used by the autoencoder, either data1, data2, or data3
    encoding_dim : int
        the length of the transcripts being encoded
    batch_size : int
        the amount of transcripts used in one training iteration
    epochs : int
        the number of iterations through the whole dataset
    name : str
        a string used to give meaningful names to output files

    Methods
    -------
    encoder()
        compresses the transcripts in the dataset
    decoder()
        decompresses the transcripts in the dataset
    encoder_decoder()
        builds the autoencoder using the encoder() and decoder() methods
    fit(batch_size=20, epochs=20)
        trains the autoencoder using the specified batch size and epochs
    save(name)
        saves the weights for the autoencoder model
    """

    dataPath = ''
    data1    = []
    data2    = []
    data3    = []

    def __init__(self, dataset, encoding_dim):
        """
        Parameters
        ----------
        dataset : list
            the dataset used by the autoencoder, either data1, data2, or data3
        encoding_dim : int
            the length of the transcripts being encoded
        """
        self.encoding_dim = encoding_dim
        self.dataset = dataset

    def encoder(self):
        """compresses the transcripts in the dataset

        The activation function used is rectified linear unit (relu)
        The Dense function is lossy and loses some transcript information

        Raises
        ------
        IndexError
            If shape of dataset is unexpected, or empty

        Returns
        ------
        model
            The current encoder model
        """

        try:
            inputs = Input(shape=(self.dataset[0].shape))   #This handles the input to the encoder. To generate a NN, we need to get the shape of the neurons
        except IndexError:
            print("ERROR: There were no transcripts found... Are inputs encoded?")
            sys.exit()
        encoded = Dense(self.encoding_dim, activation='relu')(inputs)   #Dense simply compresses the inputs using REctified Linear Units
        model = Model(inputs, encoded)  #create a model to be used as the encoder of our autoencoder
        self.encoder= model
        return model

    def decoder(self):
        """decompresses the transcripts in the dataset

        The reconstructed transcript is a prediction and
        not identical to the original.

        Returns
        ------
        model
            The current decoder model
        """

        inputs = Input(shape=(self.encoding_dim,))   #input shape of neurons comes from the encoded data
        decoded = Dense(self.encoding_dim)(inputs)                 #Now reconstruct
        model = Model(inputs, decoded)              #create the model that will be used as the decoder
        self.decoder = model
        return model

    def encoder_decoder(self):
        """builds the autoencoder using the encoder() and decoder() methods

        The reconstructed transcript is a prediction and
        not identical to the original.

        Returns
        ------
        model
            The current autoencoder model
        """

        ec = self.encoder() #make instance of encoder
        dc = self.decoder() #make instance of decoder
        inputs = Input(shape=self.dataset[0].shape)   #input is the shape of the sample
        ec_out = ec(inputs) #output of the encoder given input from data
        dc_out = dc(ec_out) #output of the decoder given input from encoder
        model = Model(inputs, dc_out)   #create the model using input data and decoder output 
        self.model = model
        return model

    def fit(self, batch_size=20, epochs=20):
        """trains the autoencoder using the specified batch size and epochs

        If the argument `batch_size`or `epochs` isn't passed in, the default
        size of 20 is used

        The autoencoder is optimized with stochastic gradient descent

        Parameters
        ----------
        batch_size : int, optional
             the amount of transcripts used in one training iteration (default is 20)
        epochs : int
            the number of iterations through the whole dataset (default is 20)
        """

        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = './log/'  #used to log our loss for visualization
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                    histogram_freq=0,
                                                    write_graph=True,
                                                    write_images=True)
        self.model.fit(self.dataset, self.dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[tbCallBack])

    def save(self, name):
        """saves the weights for the autoencoder model

        the encoder, decoder, and autoencoder weights are written
        as .h5 files

        Parameters
        ----------
        name : str, required
             a string used to give meaningful names to output files
        """

        self.name = name
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        ewPath="/home/pgian1/practice/csc534/project/weights/encoder_weights_"+name+".h5" #name of file containing encoder weights
        dwPath="/home/pgian1/practice/csc534/project/weights/decoder_weights_"+name+".h5" #name of file containing decoder weights
        wPath="/home/pgian1/practice/csc534/project/weights/ae_weights_"+name+".h5"   #name of file containing autoencoder weights
        self.encoder.save(ewPath)
        self.decoder.save(dwPath)
        self.model.save(wPath)

if __name__ == '__main__':

    ### print possible datasets ###
    if os.path.exists('./dbase/train'):
        cwd = os.getcwd()
        datasets = [f for f in listdir('./dbase/train/') if isfile(join('./dbase/train/', f))]
        print('\n'+'#'*50+'\n\n'"Possible training datasets are:\n")
        for f in datasets:
            print('"'+cwd+'/dbase/train/'+f+'"')
        print('\n'+'#'*50+'\n')

    ### choose parameters, run autoencoder ###
    dataPath = input("Please enter the path to the desired dataset. Possibilities are listed above\n")
    bs = input("Please enter the desired batch size: ")
    ep = input("Please enter the desired number of epochs: ")
    print ("Loading")

    ### Process Data by generating array and chopping to desired len ###
    with open(dataPath) as fh:
            lines = fh.readlines()
            for line in lines:
                line = line.strip('\n')
                rna = line.split()
                length = len(rna)
                if(length >= 200 and length < 500):
                    rna = [int(i) for i in rna]
                    rna = rna[:200]
                    data1.append(rna)
                if(length >= 500 and length < 1000):
                    rna = [int(i) for i in rna]
                    rna = rna[:500]
                    data2.append(rna)
                if(length >= 1000 and length < 2000):
                    rna = [int(i) for i in rna]
                    rna = rna[:1000]
                    data3.append(rna)

    ### Run the AutoEncoder for size 200-499 ###
    print("\nStarting AutoEncoder for size 200-500\n")
    ae1 = AutoEncoder(encoding_dim=200, dataset=np.array(data1))
    ae1.encoder_decoder()
    ae1.fit(batch_size=int(bs), epochs=int(ep))
    ae1.save(name="200_")
    print("model saved")

    ### Run the AutoEncoder for size 500-1000 ###
    print("\nStarting AutoEncoder for size 500-1000\n")
    ae2 = AutoEncoder(encoding_dim=500, dataset=np.array(data2))
    ae2.encoder_decoder()
    ae2.fit(batch_size=int(bs), epochs=int(ep))
    ae2.save(name="500_")
    print("model saved")

    ### Run the AutoEncoder for size 1000-2000 ###
    print("\nStarting AutoEncoder for size 1000-2000\n")
    ae3 = AutoEncoder(encoding_dim=1000, dataset=np.array(data3))
    ae3.encoder_decoder()
    ae3.fit(batch_size=int(bs), epochs=int(ep))
    ae3.save(name="1000_")
    print("Model saved!")
