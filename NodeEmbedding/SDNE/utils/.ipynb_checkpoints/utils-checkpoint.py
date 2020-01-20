import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
import pdb
import bisect

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
'''
def getSimilarity(result, k):
    print ("getting similarity...")
   
    topk = np.zeros(k)
    topk_index = np.zeros((k,2), dtype=int)
    max_value = -999999
    max_index = -1
    cnt = 0
    prev_inserted = True
    print(result.shape)
    for i in range(result.shape[0]):
        for j in range(result.shape[0]):
            sim_ij = np.dot(result[i], result[j].T)
            if cnt<k:
                topk[cnt]=sim_ij
                topk_index[cnt][0] = i
                topk_index[cnt][1] = j
            else:
                if prev_inserted == True:
                    max_value = np.max(topk)
                    max_index = np.argmax(topk)
                    prev_inserted = False
                if sim_ij<max_value:
                    topk[max_index] = sim_ij
                    topk_index[max_index][0] = i
                    topk_index[max_index][1] = j
                    prev_inserted = True
            cnt+=1
        if i%100 == 0:
            print(i)
    
    return topk, topk_index
    #return np.dot(result, result.T)

def check_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print ("get precisionK...")
        topk, topk_index = getSimilarity(embedding, max_index)
        sortedInd = np.argsort(topk)
        print topk[sortedInd]
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = topk_index[ind][0]
            y = topk_index[ind][1]
            if count<30:
                print(x,y)
            
            count += 1
            if (data.adj_matrix[x].toarray()[0][y] == 1 or x == y):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK

    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret

'''
def getSimilarity(result):
    '''
    N = result.shape[0]
    a = np.zeros((N,N), dtype=np.float16)
    print "getting similarity..."
    for i in range(N):
        for j in range(N):
            a[i][j] = np.float16(np.dot(result[i], result[j].T))
    return a
    '''
    #result = np.float16(result)
    return np.dot(result, result.T)


def check_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print "get precisionK..."
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        #print similarity[sortedInd][:max_index]
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind / data.N
            y = ind % data.N
            count += 1
            if (data.adj_matrix[x].toarray()[0][y] == 1 or x == y):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK

    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print "precisonK[%d] %.2f" % (index, precisionK[index - 1])
        ret.append(precisionK[index - 1])
    return ret


def check_link_prediction(embedding, train_graph_data, origin_graph_data, check_index):
    def get_precisionK(embedding, train_graph_data, origin_graph_data, max_index):
        print "get precisionK..."
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        N = train_graph_data.N
        for ind in sortedInd:
            x = ind / N
            y = ind % N
            #if (x == y or train_graph_data.adj_matrix[x].toarray()[0][y] == 1):
            #    continue
            count += 1
            if (origin_graph_data.adj_matrix[x].toarray()[0][y] == 1):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK

    precisionK = get_precisionK(embedding, train_graph_data, origin_graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print "precisonK[%d] %.2f" % (index, precisionK[index - 1])
        ret.append(precisionK[index - 1])
    return ret


def check_multi_label_classification(X, Y, test_ratio=0.5):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape, np.bool)
        sort_index = np.flip(np.argsort(y_pred, axis=1), 1)
        for i in range(y_test.shape[0]):
            num = sum(y_test[i])
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)

    ## small trick : we assume that we know how many label to predict
    y_pred = small_trick(y_test, y_pred)

    #print(y_train, y_test)
    
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    return "micro_f1: %.4f macro_f1 : %.4f" % (micro, macro)
    #############################################

# by DM
def plot_embed_chart(X):
    return 0

