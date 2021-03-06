'''
Reference implementation of SDNE

Author: Xuanrong Yao, Daixin wang

for more detail, refer to the paper:
SDNE : structral deep network embedding
Wang, Daixin and Cui, Peng and Zhu, Wenwu
Knowledge Discovery and Data Mining (KDD), 2016
'''

from config import Config
from graph import Graph
from model.sdne import SDNE
from utils.utils import *
import scipy.io as sio
import time
from optparse import OptionParser
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = OptionParser()
    parser.add_option("-c",dest = "config_file", action = "store", metavar = "CONFIG FILE")
    options, _ = parser.parse_args()
    if options.config_file == None:
        raise IOError("no config file specified")

    config = Config(options.config_file)
    
    train_graph_data = Graph(config.train_graph_file, config.ng_sample_ratio)
   
    if config.origin_graph_file:
        origin_graph_data = Graph(config.origin_graph_file, config.ng_sample_ratio)

    if config.label_file:
        #load label for classification
        train_graph_data.load_label_data(config.label_file)
    
    config.struct[0] = train_graph_data.N
    
    model = SDNE(config)
    model.do_variables_init(train_graph_data)
    embedding = None
    norm_weights = None
    while (True):
        mini_batch, index = train_graph_data.sample(config.batch_size, do_shuffle = False) #insert index
        if embedding is None:
            embedding = model.get_embedding(mini_batch, index)
        else:
            embedding = np.vstack((embedding, model.get_embedding(mini_batch, index)))
        if train_graph_data.is_epoch_end:
            break

    epochs = 0
    batch_n = 0


    tt = time.ctime().replace(' ','-')
    path = "./result/" + config.embedding_filename + '-' + tt
    os.system("mkdir " + path)
    fout = open(path + "/log.txt","w")
    #model.save_model(path + '/epoch0.model')

    sio.savemat(path + '/embedding.mat',{'embedding':embedding})
    print "!!!!!!!!!!!!!"
    while (True):
        if train_graph_data.is_epoch_end:
            loss = 0
            if epochs % config.display == 0:
                embedding = None
                norm_weights = None
                while (True):
                    mini_batch, index = train_graph_data.sample(config.batch_size, do_shuffle = False)  #return the 'index' together
                    loss += model.get_loss(mini_batch, index)
                    if embedding is None:
                        embedding = model.get_embedding(mini_batch, index)  #Feed here 'index' together
                        norm_weights = model.get_norm_weight(mini_batch, index) 
                    else:
                        embedding = np.vstack((embedding, model.get_embedding(mini_batch, index)))
                        norm_weights = np.vstack((norm_weights, model.get_norm_weight(mini_batch, index)))
                    if train_graph_data.is_epoch_end:
                        break

                print "Epoch : %d loss : %.3f" % (epochs, loss)
                print >>fout, "Epoch : %d loss : %.3f" % (epochs, loss)
                
                if config.check_reconstruction:
                    print >> fout, epochs, "reconstruction:", check_reconstruction(embedding, train_graph_data, config.check_reconstruction)
                if config.check_classification:
                    data, index = train_graph_data.sample(train_graph_data.N, do_shuffle = False,  with_label = True)
                    print >> fout, epochs, "classification_0.7", check_multi_label_classification(embedding, data.label, test_ratio=0.3)
                    print >> fout, epochs, "classification_0.9", check_multi_label_classification(embedding, data.label, test_ratio=0.1)

                fout.flush()
                #model.save_model(path + '/epoch' + str(epochs) + ".model")
            if epochs == config.epochs_limit:
                print "exceed epochs limit terminating"
                break
            epochs += 1
        mini_batch, index = train_graph_data.sample(config.batch_size)
        loss = model.fit(mini_batch, index)

    #sio.savemat(path + '/embedding.mat',{'embedding':embedding})
    #sio.savemat(path + '/norm_weights.mat',{'norm_weights':norm_weights})
    #sio.savemat(path + '/adj_matrix.mat',{'adj_matrix':train_graph_data.adj_matrix.toarray()})
    fout.close()
