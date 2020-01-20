"""
JCA from Ziwei Zhu
Computer Science and Engineering Department, Texas A&M University
zhuziwei@tamu.edu

modified by Dongmin Park
Knowledge and Service Engineering Department, KAIST
dongminpark@kaist.ac.kr
"""
import tensorflow as tf
import time
import numpy as np
import utility


class JCA:

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

        self.lambda_value = args.lambda_value  # regularization term trade-off
        self.lambda_u = args.lambda_u
        self.lambda_i = args.lambda_i

        self.result_path = result_path
        self.metric_path = metric_path
        self.date = date  # today's date
        self.data_name = data_name

        self.neg_sample_rate = args.neg_sample_rate
        self.U_OH_mat = np.eye(self.num_rows, dtype=float)
        self.I_OH_mat = np.eye(self.num_cols, dtype=float)

        self.max_f1_avg = 0
        self.max_r_f_table = np.zeros((self.num_rows, 4))
        self.max_epoch = 0

        print('**********JCA**********')
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

        # input rating vector
        self.input_R_U = tf.placeholder(dtype=tf.float32, shape=[None, self.num_cols], name="input_R_U")
        self.input_R_I = tf.placeholder(dtype=tf.float32, shape=[self.num_rows, None], name="input_R_I")
        self.input_OH_U = tf.placeholder(dtype=tf.float32, shape=[None, self.num_rows], name="input_OH_U")  # ??
        self.input_OH_I = tf.placeholder(dtype=tf.float32, shape=[None, self.num_cols], name="input_OH_I")
        self.input_P_cor = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="input_P_cor")
        self.input_N_cor = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="input_N_cor")

        # input indicator vector indicator
        self.row_idx = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="row_idx")
        self.col_idx = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="col_idx")

        # JCA parameters
        # user component
        # encoding layer weights
        UW1 = tf.get_variable(name="UW1", initializer=tf.truncated_normal(shape=[self.num_cols, self.U_hidden_neuron], mean=0, stddev=0.03), dtype=tf.float32)
        # encoding layer bias
        Ub1 = tf.get_variable(name="Ub1", initializer=tf.truncated_normal(shape=[1, self.U_hidden_neuron], mean=0, stddev=0.03), dtype=tf.float32)

        # decoding layer weights
        UW2 = tf.get_variable(name="UW2", initializer=tf.truncated_normal(shape=[self.U_hidden_neuron, self.num_cols], mean=0, stddev=0.03), dtype=tf.float32)
        # decoding layer bias
        Ub2 = tf.get_variable(name="Ub2", initializer=tf.truncated_normal(shape=[1, self.num_cols], mean=0, stddev=0.03), dtype=tf.float32)

        # item component
        # encoding layer weights
        IW1 = tf.get_variable(name="IW1", initializer=tf.truncated_normal(shape=[self.num_rows, self.I_hidden_neuron], mean=0, stddev=0.03), dtype=tf.float32)
        # encoding layer bias
        Ib1 = tf.get_variable(name="Ib1", initializer=tf.truncated_normal(shape=[1, self.I_hidden_neuron], mean=0, stddev=0.03), dtype=tf.float32)

        # decoding layer weights
        IW2 = tf.get_variable(name="IW2", initializer=tf.truncated_normal(shape=[self.I_hidden_neuron, self.num_rows], mean=0, stddev=0.03), dtype=tf.float32)
        # decoding layer bias
        Ib2 = tf.get_variable(name="Ib2", initializer=tf.truncated_normal(shape=[1, self.num_rows], mean=0, stddev=0.03), dtype=tf.float32)

        # TRAP: Microscopic Regularizer
        U_ObjScale_param = tf.get_variable(name="U_ObjScale_param", initializer=tf.random_uniform(shape=[1, self.num_rows]), dtype=tf.float32)
        I_ObjScale_param = tf.get_variable(name="I_ObjScale_param", initializer=tf.random_uniform(shape=[1, self.num_cols]), dtype=tf.float32)

        # JCA Architecture + TRAP
        # user component
        # Encoder
        U_pre_Encoder = tf.matmul(self.input_R_U, UW1) + Ub1  # input to the hidden layer
        U_scaling = tf.transpose(tf.matmul(U_ObjScale_param, tf.transpose(self.input_OH_U)))  # (1, ?)
        self.U_Encoder = None
        self.plotting1 = None
        if self.using_trap == 0:
            self.U_Encoder = self.g_act(U_pre_Encoder)
            self.plotting1 = U_pre_Encoder
        else:
            self.g_act = tf.nn.tanh
            self.U_Encoder = self.g_act(U_pre_Encoder * U_scaling)  # output of the hidden layer
            self.plotting1 = U_pre_Encoder * U_scaling
        self.U_scaling = U_scaling
        # Decoder
        U_pre_Decoder = tf.matmul(self.U_Encoder, UW2) + Ub2  # input to the output layer
        self.U_Decoder = self.f_act(U_pre_Decoder)  # output of the output layer

        # item component
        # Encoder
        I_pre_Encoder = tf.matmul(tf.transpose(self.input_R_I), IW1) + Ib1  # input to the hidden layer
        I_scaling = tf.transpose(tf.matmul(I_ObjScale_param, tf.transpose(self.input_OH_I)))  # (1, ?)
        self.I_Encoder = None
        self.plotting2 = None
        if self.using_trap == 0:
            self.I_Encoder = self.g_act(I_pre_Encoder)
            self.plotting2 = I_pre_Encoder
        else:
            self.g_act = tf.nn.tanh
            self.I_Encoder = self.g_act(I_pre_Encoder * I_scaling)  # output of the hidden layer
            self.plotting2 = I_pre_Encoder * I_scaling
        self.I_scaling = I_scaling
        # Decoder
        I_pre_Decoder = tf.matmul(self.I_Encoder, IW2) + Ib2  # input to the output layer
        self.I_Decoder = self.f_act(I_pre_Decoder)  # output of the output layer  #(?,942)

        # final output
        self.Decoder = ((tf.transpose(tf.gather_nd(tf.transpose(self.U_Decoder), self.col_idx)))
                        + tf.gather_nd(tf.transpose(self.I_Decoder), self.row_idx)) / 2.0

        # pairwise hinge-based loss function
        pos_data = tf.gather_nd(self.Decoder, self.input_P_cor)
        neg_data = tf.gather_nd(self.Decoder, self.input_N_cor)

        pre_cost1 = tf.maximum(neg_data - pos_data + self.margin, tf.zeros(tf.shape(neg_data)))
        cost1 = tf.reduce_sum(pre_cost1)  # prediction squared error

        # Frobinius Norm Regularization for Parameters to avoid overfitting
        pre_cost2 = tf.square(self.l2_norm(UW2)) + tf.square(self.l2_norm(UW1)) \
                    + tf.square(self.l2_norm(IW2)) + tf.square(self.l2_norm(IW1))\
                    + tf.square(self.l2_norm(Ib1)) + tf.square(self.l2_norm(Ib2))\
                    + tf.square(self.l2_norm(Ub1)) + tf.square(self.l2_norm(Ub2))
        cost2 = self.lambda_value * 0.5 * pre_cost2  # regularization term

        # TRAP: Macroscopic Regularizer
        self.pre_cost3 = tf.square(self.l2_norm(self.U_Encoder))
        self.pre_cost4 = tf.square(self.l2_norm(self.I_Encoder))

        self.pre_cost3 = self.l1_norm(self.U_Encoder)
        self.pre_cost4 = self.l1_norm(self.I_Encoder)

        cost3 = self.lambda_u * self.pre_cost3
        cost4 = self.lambda_i * self.pre_cost4

        # Final loss function
        self.cost = None
        if self.using_trap == 0:
            self.cost = cost1 + cost2
        else:
            self.cost = cost1 + cost2 + cost3 + cost4

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
        random_col_idx = np.random.permutation(self.num_cols)  # randomly permute the cols
        batch_cost = 0
        ts = 0
        for i in range(self.num_batch_U):  # iterate each batch
            if i == self.num_batch_U - 1:
                row_idx = random_row_idx[i * self.batch_size:]
            else:
                row_idx = random_row_idx[(i * self.batch_size):((i + 1) * self.batch_size)]
            for j in range(self.num_batch_I):
                # get the indices of the current batch
                if j == self.num_batch_I - 1:
                    col_idx = random_col_idx[j * self.batch_size:]
                else:
                    col_idx = random_col_idx[(j * self.batch_size):((j + 1) * self.batch_size)]
                ts1 = time.time()
                p_input, n_input = utility.pairwise_neg_sampling(self.train_R, row_idx, col_idx, self.neg_sample_rate)

                ts2 = time.time()
                ts += (ts2 - ts1)

                input_R_U = self.train_R[row_idx, :]
                input_R_I = self.train_R[:, col_idx]
                _, cost = self.sess.run(  # do the optimization by the minibatch
                    [self.optimizer, self.cost],
                    feed_dict={
                        self.input_R_U: input_R_U,
                        self.input_R_I: input_R_I,
                        self.input_OH_U: self.U_OH_mat[row_idx, :],
                        self.input_OH_I: self.I_OH_mat[col_idx, :],
                        self.input_P_cor: p_input,
                        self.input_N_cor: n_input,
                        self.row_idx: np.reshape(row_idx, (len(row_idx), 1)),
                        self.col_idx: np.reshape(col_idx, (len(col_idx), 1))})
                batch_cost = batch_cost + cost

        if itr % self.display_step == 0:
            print ("Training //", "Epoch %d //" % itr, " Total cost = {:.2f}".format(batch_cost),
                   "Elapsed time : %d sec //" % (time.time() - start_time), "Sampling time: %d s //" %(ts))

    def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
        start_time = time.time()
        _, Decoder = self.sess.run([self.cost, self.Decoder],
                                   feed_dict={
                                        self.input_R_U: self.train_R,
                                        self.input_R_I: self.train_R,
                                        self.input_OH_U: self.U_OH_mat,
                                        self.input_OH_I: self.I_OH_mat,
                                        self.input_P_cor: [[0, 0]],
                                        self.input_N_cor: [[0, 0]],
                                        self.row_idx: np.reshape(range(self.num_rows), (self.num_rows, 1)),
                                        self.col_idx: np.reshape(range(self.num_cols), (self.num_cols, 1))})
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

            print (
                "Testing //", "Epoch %d //" % itr, " Total cost = {:.2f}".format(numerator),
                " RMSE = {:.5f}".format(RMSE),
                "Elapsed time : %d sec" % (time.time() - start_time))
            print ("=" * 100)

    def make_records(self):  # record all the results' details into files
        _, Decoder, l1, l2 = self.sess.run([self.cost, self.Decoder, self.pre_cost3, self.pre_cost4],
                                   feed_dict={
                                        self.input_R_U: self.train_R,
                                        self.input_R_I: self.train_R,
                                        self.input_OH_U: self.U_OH_mat,
                                        self.input_OH_I: self.I_OH_mat,
                                        self.input_P_cor: [[0, 0]],
                                        self.input_N_cor: [[0, 0]],
                                        self.row_idx: np.reshape(range(self.num_rows), (self.num_rows, 1)),
                                        self.col_idx: np.reshape(range(self.num_cols), (self.num_cols, 1))})
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

    @staticmethod
    def l1_norm(tensor):
        return tf.reduce_sum(tf.abs(tensor))
