from data_preprocessor import *
import tensorflow as tf
import time
import argparse

from model.u_autoRec import uAutoRec
from model.cdae import CDAE
from model.multivae import multiVAE
from model.JCA import JCA

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':

    date = time.strftime('%y-%m-%d', time.localtime())
    current_time = time.strftime('%H:%M:%S', time.localtime())
    data_name = 'ml-1m'
    base = 'u'

    # TRAP mode
    using_trap = 0
    lambda_u = 0.01 # Grid search needed

    # for CDAE
    corruption_ratio = 0.2

    # for JCA
    lambda_i = 0.01 # Grid search needed

    # Configurations
    parser = argparse.ArgumentParser(description='JCA')

    parser.add_argument('--train_epoch', type=int, default=100) #default200
    parser.add_argument('--batch_size', type=int, default=1500)
    parser.add_argument('--display_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lambda_value', type=float, default=0.001)
    parser.add_argument('--lambda_u', type=float, default=lambda_u)
    parser.add_argument('--lambda_i', type=float, default=lambda_i)
    parser.add_argument('--margin', type=float, default=0.15)
    parser.add_argument('--optimizer_method', choices=['Adam', 'Adadelta', 'Adagrad', 'RMSProp', 'GradientDescent',
                                                       'Momentum'], default='Adam')
    parser.add_argument('--corruption_ratio', type=float, default=corruption_ratio)

    parser.add_argument('--using_trap', type=int, default=using_trap)
    parser.add_argument('--g_act', choices=['Sigmoid', 'Relu', 'Elu', 'Tanh', "Identity"], default='Sigmoid')
    parser.add_argument('--f_act', choices=['Sigmoid', 'Relu', 'Elu', 'Tanh', "Identity"], default='Sigmoid')
    parser.add_argument('--U_hidden_neuron', type=int, default=160)
    parser.add_argument('--I_hidden_neuron', type=int, default=160)
    parser.add_argument('--base', type=str, default=base)
    parser.add_argument('--neg_sample_rate', type=int, default=1)
    args = parser.parse_args()

    sess = tf.Session()

    # Datasets
    #train_R, test_R = VideoGame.test()
    #train_R, test_R = yelp.test()
    train_R, test_R = ml1m.test()
    #train_R, test_R = ml100k.test()
    print("***train_R, test_R shape***", train_R.shape, test_R.shape)

    # Saving path
    metric_path = './metric_results_test/' + date + '/'
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    metric_path = metric_path + '/' + str(parser.description) + "_" + str(current_time)

    # Training and Testing for each model
    # AutoRec
    # u_autorec = uAutoRec(sess, args, train_R, test_R, metric_path, date, data_name)
    # u_autorec.run(train_R, test_R)

    # CDAE
    # cdae = CDAE(sess, args, train_R, test_R, metric_path, date, data_name)
    # cdae.run(train_R, test_R)

    # multiVAE
    multivae = multiVAE(sess, args, train_R, test_R, metric_path, date, data_name)
    multivae.run(train_R, test_R)

    # JCA
    #jca = JCA(sess, args, train_R, test_R, metric_path, date, data_name)
    #jca.run(train_R, test_R)
