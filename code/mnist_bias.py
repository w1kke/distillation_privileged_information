from keras.layers.core import Dense, Dropout, Activation
from keras.objectives import categorical_crossentropy
from keras.models import Sequential, Graph
from scipy.misc import imresize
import numpy as np
import theano
import sys

def downsample(x,p_down):
    size = len(imresize(x[0].reshape(28,28),p_down,mode='F').ravel())
    s_tr = np.zeros((x.shape[0], size))
    for i in xrange(x.shape[0]):
      img = x[i].reshape(28,28)
      s_tr[i] = imresize(img,p_down,mode='F').ravel()
    return s_tr

def MLP(d,m,q):
    model = Sequential()
    model.add(Dense(m, input_dim=d, activation="relu"))
    model.add(Dense(m, activation="relu"))
    model.add(Dense(q))
    model.add(Activation('softmax'))
    model.compile('rmsprop','categorical_crossentropy')
    return model

def MLP_bias(d,m,q,b):
    model = Sequential()
    weights = (d, m)
    model.add(Dense(m, input_dim=d, activation="relu"))
    model.add(Dense(m, activation="relu"))
    model.add(Dense(q))
    model.add(Activation('softmax'))
    model.compile('rmsprop','categorical_crossentropy')
    return model

def softmax(w, t = 1.0):
    e = np.exp(w / t)
    return e/np.sum(e,1)[:,np.newaxis]

def weighted_loss(base_loss,l):
    def loss_function(y_true, y_pred):
        return l*base_loss(y_true,y_pred)
    return loss_function

def distillation(d,m,q,t,l):
    graph = Graph()
    graph.add_input(name='x', input_shape=(d,))
    graph.add_node(Dense(m), name='w1', input='x')
    graph.add_node(Activation('relu'), name='z1', input='w1')
    graph.add_node(Dense(m), name='w2', input='z1')
    graph.add_node(Activation('relu'), name='z2', input='w2')
    graph.add_node(Dense(q), name='w3', input='z2')
    graph.add_node(Activation('softmax'), name='hard_softmax', input='w3')
    graph.add_node(Activation('softmax'), name='soft_softmax', input='w3')
    graph.add_output(name='hard', input='hard_softmax')
    graph.add_output(name='soft', input='soft_softmax')
    loss_hard = weighted_loss(categorical_crossentropy,1.-l)
    loss_soft = weighted_loss(categorical_crossentropy,t*t*l)
    graph.compile('rmsprop', {'hard':loss_hard, 'soft':loss_soft})
    return graph

def load_data(dataset):
    import cPickle as pkl
    f = open('../data/' + dataset + '.npz', 'rb')
    d = pkl.load(f)
    x_tr = np.asarray(d['x_tr']).astype(np.float32)  
    x_te = np.asarray(d['x_te']).astype(np.float32)

    x_tr_result = {}
    #x_te_result = {}
    y_tr_result = {}
    #y_te_result = {}  

    for i in range(0, 10):
      x_tr_result[str(i)] = []
      #x_te_result[i] = []
      y_tr_result[str(i)] = []
      #y_te_result[i] = []

    y_tmp = np.asarray(d['y_tr']).astype(np.float32)
    counter = 0
    y_tr = []
    for y in y_tmp:
      vec = np.zeros(10)
      vec[int(y)] = 1
      y_tr = np.append(y_tr, vec)
      y_tr_result[str(int(y))] = np.append(y_tr_result[str(int(y))], vec)
      x_tr_result[str(int(y))] = np.append(x_tr_result[str(int(y))], x_tr[counter])
      counter = counter + 1
    for i in range(0, 10):
      x_tr_result[str(i)] = np.reshape(np.asarray(x_tr_result[str(i)]).astype(np.float32), (len(x_tr_result[str(i)]) / 784, 784))
      y_tr_result[str(i)] = np.reshape(np.asarray(y_tr_result[str(i)]).astype(np.float32), (len(y_tr_result[str(i)]) / 10, 10))

    y_tmp = np.asarray(d['y_te']).astype(np.float32)
    #counter = 0
    y_te = []
    for y in y_tmp:
      vec = np.zeros(10)
      vec[int(y)] = 1
      y_te = np.append(y_te, vec)
      #y_te_result[int(y)] = np.append(y_te_result[int(y)], vec)
      #x_te_result[int(y)] = x_te[counter]
      #counter = counter +1
    y_te = np.reshape(np.asarray(y_te).astype(np.float32), (len(y_te) / 10, 10))
    #for i in range(0, 10):
    #  y_te_result[i] = np.reshape(np.asarray(y_te_result[i]).astype(np.float32), (len(y_te_result[i]) / 10, 10))

    f.close()

    return x_tr_result, y_tr_result, x_te, y_te

np.random.seed(0)

ax_tr, ay_tr, x_te, y_te = load_data('mnist')
p_downsample = 25

# number of images used in training
N = int(sys.argv[1])

# number of neurons per hidden layer
M = int(sys.argv[2])

# students get less neurons
studLayerSize = M #// 4

outfile = open('result_mnist_test_' + str(N) + '_' + str(M), 'w')

#for i in range(0, 10):
#  xs_te[i] = downsample(x_te[i],p_downsample)
#  x_te[i]  = x_te[i]/255.0
#  xs_te[i] = xs_te[i]/255.0

xs_te = downsample(x_te,p_downsample)
x_te  = x_te/255.0
xs_te = xs_te/255.0

for rep in xrange(10):

  mlp_big = {}
  err_big = {}

  # train 10 teachers, each one trained on images of a single letter
  for i in range(0, 10):
    # random training split
    j     = np.random.permutation(ax_tr[str(i)].shape[0])[0:N]
    x_tr  = ax_tr[str(i)][j]
    y_tr  = ay_tr[str(i)][j]
    xs_tr = downsample(x_tr,p_downsample)
    x_tr  = x_tr/255.0
    xs_tr = xs_tr/255.0
    
    # big mlp
    mlp_big[i] = MLP(x_tr.shape[1],M,y_tr.shape[1])
    mlp_big[i].fit(x_tr, y_tr, nb_epoch=50, verbose=0)
    err_big[i] = np.mean(mlp_big[i].predict_classes(x_te,verbose=0)==np.argmax(y_te,1))

  err_lst = []
  for i in range(0,10):
    err_lst.append(err_big[i])
  err_big = err_lst

  # student mlp
  for t in [1,2,5,10,20,50,100,200,400]:
    for L in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
      ys_tr_dict = {}
      for i in range(0, 10):
        soften = theano.function([mlp_big[i].layers[0].input], mlp_big[i].layers[2].output)
        if i not in ys_tr_dict:
          ys_tr_dict[i] = []
        ys_tr_dict[i]  = np.append(ys_tr_dict[i], softmax(soften(x_tr),t))

      for i in range(0, 10):
        ys_tr_dict[i] = np.reshape(np.asarray(ys_tr_dict[i]).astype(np.float32), (len(ys_tr_dict[i]) / 10, 10))
      
      # average over the soft label
      ys_tr = []
      for i in range(0, len(ys_tr_dict[0])):
        ys_max = np.zeros(10)
        for j in range(0, 10):
          if np.max(ys_max) < np.max(ys_tr_dict[j][i]):
            ys_max = ys_tr_dict[j][i]

        #ys_mean /= 10
        ys_tr = np.append(ys_tr, ys_max)

      ys_tr = np.reshape(np.asarray(ys_tr).astype(np.float32), (len(ys_tr) / 10, 10))

      mlp_student = distillation(xs_tr.shape[1],studLayerSize,ys_tr.shape[1],t,L)
      mlp_student.fit({'x':xs_tr, 'hard':y_tr, 'soft':ys_tr}, nb_epoch=50, verbose=0)
      err_student = np.mean(np.argmax(mlp_student.predict({'x':xs_te})['hard'],1)==np.argmax(y_te,1))
      
      # save result
      line = [N, p_downsample, round(np.mean(err_big),3), t, L, round(err_student,3)]
      outfile.write(str(line)+'\n')

      # save the model

      filename = 'models/' + str(round(err_student,3)) + '_' + str(i) + '_' + str(t) + '_' + str(L)

      json_string = mlp_student.to_json()
      open(filename + '_model.json', 'w').write(json_string)
      mlp_student.save_weights(filename + '_weights.h5', overwrite=True)

outfile.close()
