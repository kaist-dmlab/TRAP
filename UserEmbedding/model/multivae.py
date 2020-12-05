"""
Dongmin Park
Knowledge and Service Engineering Department, KAIST
dongminpark@kaist.ac.kr
"""
import tensorflow as tf
import time
import numpy as np
import os
import matplotlib
import copy
import utility


class multiVAE:

    def __init__(self, sess, args, train_R, vali_R, metric_path, date, data_name,
                 result_path=None):

        if args.f_act == "Sigmoid":
            f_act = tf.nn.sigmoid
        elif args.f_act == "Relu":
            f_act = tf.nn.relu
        elif args.f_act == "Tanh":
            f_act = tf.nn.tanh
        elif args.f_act == "Identity":
            f_act = tf.identity
        elif args.f_act == "Elu":
            f_act = tf.nn.elu
        else:
            raise NotImplementedError("ERROR")

        if args.g_act == "Sigmoid":
            g_act = tf.nn.sigmoid
        elif args.g_act == "Relu":
            g_act = tf.nn.relu
        elif args.g_act == "Tanh":
            g_act = tf.nn.tanh
        elif args.g_act == "Identity":
            g_act = tf.identity
        elif args.g_act == "Elu":
            g_act = tf.nn.elu
        else:
            raise NotImplementedError("ERROR")

        self.sess = sess
        self.args = args

        self.base = args.base

        self.num_rows = train_R.shape[0]
        self.num_cols = train_R.shape[1]
        self.U_hidden_neuron = args.U_hidden_neuron
        self.I_hidden_neuron = args.I_hidden_neuron

        self.train_R = train_R
        self.vali_R = vali_R
        self.num_test_ratings = np.sum(vali_R)

        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch_U = int(self.num_rows / float(self.batch_size)) + 1
        self.num_batch_I = int(self.num_cols / float(self.batch_size)) + 1

        self.lr = args.lr  # learning rate
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step
        self.margin = args.margin

        self.using_trap = args.using_trap

        self.f_act = f_act  # the activation function for the output layer
        self.g_act = g_act  # the activation function for the hidden layer

        self.global_step = tf.Variable(0, trainable=False)

        self.lambda_value = args.lambda_value
        self.lambda_u = args.lambda_u  # macroscopic regularizer for user embeddings

        self.result_path = result_path
        self.metric_path = metric_path
        self.date = date  # today's date
        self.data_name = data_name

        self.neg_sample_rate = args.neg_sample_rate
        self.U_OH_mat = np.eye(self.num_rows, dtype=float)

        self.max_f1_avg = 0
        self.max_r_f_table = np.zeros((self.num_rows, 4))
        self.max_epoch = 0

        self.total_anneal = self.train_epoch * self.num_batch_U * 0.5
        self.update_cnt = 0
        self.anneal_cap = 0

        print('**********multiVAE**********')
        print(self.args)
        self.prepare_model()

    def run(self, train_R, vali_R):
        self.train_R = train_R
        self.vali_R = vali_R
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(self.train_epoch):
            self.train_model(epoch_itr)
            if epoch_itr % 1 == 0:
                self.test_model(epoch_itr)
        return self.make_records()

    def prepare_model(self):
        # placeholders with default values when scoring
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)

        # input rating vector
        self.input_R_U = tf.placeholder(dtype=tf.float32, shape=[None, self.num_cols], name="input_R_U")
        self.input_R_U_index = tf.placeholder(dtype=tf.float32, shape=[None, self.num_rows], name="input_R_U_index")

        self.input_P_cor = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="input_P_cor")
        self.input_N_cor = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="input_N_cor")

        # input indicator vector indicator
        self.row_idx = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="row_idx")
        self.col_idx = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="col_idx")

        # multiVAE Parameters
        # encoding layer weights
        UW1 = tf.get_variable(name="UW1", initializer=tf.truncated_normal(shape=[self.num_cols, 2*self.U_hidden_neuron], mean=0, stddev=0.03), dtype=tf.float32)
        # encoding layer bias
        Ub1 = tf.get_variable(name="Ub1", initializer=tf.truncated_normal(shape=[1, 2*self.U_hidden_neuron], mean=0, stddev=0.03), dtype=tf.float32)

        # decoding layer weights
        UW2 = tf.get_variable(name="UW2", initializer=tf.truncated_normal(shape=[self.U_hidden_neuron, self.num_cols], mean=0, stddev=0.03), dtype=tf.float32)
        # decoding layer bias
        Ub2 = tf.get_variable(name="Ub2", initializer=tf.truncated_normal(shape=[1, self.num_cols], mean=0, stddev=0.03), dtype=tf.float32)

        # TRAP: Microscopic Regularizer
        ObjScale_param = tf.get_variable(name="ObjScale_param", initializer=tf.random_uniform(shape=[1, self.num_rows]), dtype=tf.float32)

        pre_Encoder = tf.matmul(self.input_R_U, UW1) + Ub1  # input to the hidden layer
        scaling = tf.transpose(tf.matmul(ObjScale_param, tf.transpose(self.input_R_U_index)))  # (1, ?)

        # Encoding
        mu_q = None
        logvar_q = None
        if self.using_trap == 0:
            self.U_Encoder = self.g_act(pre_Encoder)
            mu_q = self.U_Encoder[:, :self.U_hidden_neuron]
            logvar_q = self.U_Encoder[:, self.U_hidden_neuron:]

        else:
            self.g_act = tf.nn.tanh
            self.U_Encoder = self.g_act(pre_Encoder)  # output of the hidden layer
            mu_q = self.g_act(pre_Encoder[:, :self.U_hidden_neuron] * scaling)
            logvar_q = self.g_act(pre_Encoder[:, self.U_hidden_neuron:])

        self.plotting = mu_q
        self.scaling = scaling
        self.plotting2 = self.U_Encoder

        # multiVAE loss
        std_q = tf.exp(0.5 * logvar_q)
        KL = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1), axis=1))

        # reparameterization
        epsilon = tf.random_normal(tf.shape(std_q))
        sampled_z = mu_q + self.is_training_ph * epsilon * std_q

        # Decoder
        U_pre_Decoder = tf.matmul(sampled_z, UW2) + Ub2  # input to the output layer
        self.U_Decoder = self.f_act(U_pre_Decoder)  # output of the output layer

        self.Decoder = tf.nn.log_softmax(self.U_Decoder) #log_softmax_var

        # multinomial likelihood loss
        neg_ll = -tf.reduce_mean(tf.reduce_sum(self.Decoder * self.input_R_U, axis=-1))

        # Prameter Regularization with frobinius norm to avoid overfitting
        pre_cost2 = tf.square(self.l2_norm(UW2)) + tf.square(self.l2_norm(UW1)) \
                    + tf.square(self.l2_norm(Ub1)) + tf.square(self.l2_norm(Ub2))
        cost2 = self.lambda_value * pre_cost2

        # TRAP: Macroscopic Regularizer
        self.pre_cost3 = tf.square(self.l2_norm(self.U_Encoder))
        cost3 = self.lambda_u * self.pre_cost3

        # Final loss function
        self.cost = None
        if self.using_trap == 0:
            self.cost = neg_ll + 0.2 * KL + cost2 #neg_ELBO
        else:
            self.cost = neg_ll + 0.2 * KL + cost2 + cost3

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "Adagrad":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optimizer_method == "GradientDescent":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optimizer_method == "Momentum":
            optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
        else:
            raise ValueError("Optimizer Key ERROR")

        gvs = optimizer.compute_gradients(self.cost)
        self.optimizer = optimizer.apply_gradients(gvs, global_step=self.global_step)

    def train_model(self, itr):
        start_time = time.time()
        random_row_idx = np.random.permutation(self.num_rows)  # randomly permute the rows
        batch_cost = 0
        anneal = 0

        for i in range(self.num_batch_U):  # iterate each batch
            if i == self.num_batch_U - 1:
                row_idx = random_row_idx[i * self.batch_size:]
            else:
                row_idx = random_row_idx[(i * self.batch_size):((i + 1) * self.batch_size)]

            if self.total_anneal > 0:
                anneal = min(self.anneal_cap, 1. * self.update_cnt / self.total_anneal)
            else:
                anneal = self.anneal_cap

            input_R_U = self.train_R[row_idx, :]
            _, cost = self.sess.run(  # do the optimization by the minibatch
                [self.optimizer, self.cost],
                feed_dict={
                    self.input_R_U: input_R_U,
                    self.input_R_U_index: self.U_OH_mat[row_idx, :],
                    self.is_training_ph: 1,
                    self.anneal_ph: 0.2,
                    self.row_idx: np.reshape(row_idx, (len(row_idx), 1))})
            batch_cost = batch_cost + cost

        if itr % self.display_step == 0:
            print ("Training //", "Epoch %d //" % itr, " Total cost = {:.2f}".format(batch_cost),
                   "Elapsed time : %d sec //" % (time.time() - start_time)) #, "Sampling time: %d s //" %(ts)

    def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
        start_time = time.time()
        _, Encoder, Decoder = self.sess.run([self.cost, self.U_Encoder[:, :self.U_hidden_neuron], self.Decoder],
                                   feed_dict={
                                        self.input_R_U: self.train_R,
                                        self.input_R_U_index: self.U_OH_mat,
                                        self.is_training_ph: 0,
                                        self.anneal_ph: 0.2,
                                        self.row_idx: np.reshape(range(self.num_rows), (self.num_rows, 1))})
        if itr % self.display_step == 0:

            pre_numerator = np.multiply((Decoder - self.vali_R), self.vali_R)
            numerator = np.sum(np.square(pre_numerator))
            denominator = self.num_test_ratings
            RMSE = np.sqrt(numerator / float(denominator))

            if itr % 1 == 0:
                if self.base == 'i':
                    [precision, recall, f_score, NDCG, r_f_table] = utility.test_model_all(Decoder.T, self.vali_R.T,
                                                                                self.train_R.T)
                else:
                    [precision, recall, f_score, NDCG, r_f_table] = utility.test_model_all(Decoder, self.vali_R, self.train_R)

            if self.max_f1_avg < f_score[2]:
                self.max_f1_avg = f_score[2]
                self.max_r_f_table = r_f_table
                self.max_epoch = itr
                self.max_embedded_x = Encoder

            print (
                "Testing //", "Epoch %d //" % itr, " Total cost = {:.2f}".format(numerator),
                " RMSE = {:.5f}".format(RMSE),
                "Elapsed time : %d sec" % (time.time() - start_time))
            print ("=" * 100)

    def make_records(self):  # record all the results' details into files
        _, Decoder = self.sess.run([self.cost, self.Decoder],
                                   feed_dict={
                                       self.input_R_U: self.train_R,
                                       self.input_R_U_index: self.U_OH_mat,
                                       self.is_training_ph: 0,
                                       self.anneal_ph: 0.2,
                                       self.row_idx: np.reshape(range(self.num_rows), (self.num_rows, 1))})
        if self.base == 'i':
            [precision, recall, f_score, NDCG, r_f_table] = utility.test_model_all(Decoder.T, self.vali_R.T, self.train_R.T)
        else:
            [precision, recall, f_score, NDCG, r_f_table] = utility.test_model_all(Decoder, self.vali_R, self.train_R)

        utility.metric_record(precision, recall, f_score, NDCG, self.args, self.metric_path)

        utility.test_model_factor(Decoder, self.vali_R, self.train_R)

        print("******** max_epoch ********")
        print(self.max_epoch)

        return precision, recall, f_score, NDCG

    @staticmethod
    def l2_norm(tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))
