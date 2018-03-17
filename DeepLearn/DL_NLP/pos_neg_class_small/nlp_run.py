# coding: utf-8

# In[1]:
import tensorflow as tf
#from Algorithm.DeepLearn.tf_nltk import create_featureset_and_labels
import numpy as np
import pickle

#train_x, train_y, test_x, test_y = create_featureset_and_labels("pos.txt","neg.txt")
train_x, train_y, test_x, test_y = pickle.load(open("C:\\Users\\15764\\Documents\\Datasets\\ft_bin_class\\sentement_set.pickle","rb"))

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum': n_nodes_hl3,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# Nothing changes
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =prediction,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)

# In[ ]:


# def nearal_network_model(data,layer_nodes_sizes,n_classes): datasize = int(data.shape[1]) numlayers = len(layer_nodes_sizes) hidden_layers = [] inputlayer = {'weights':tf.Variable(tf.random_normal([datasize,layer_nodes_sizes[0]])),'biases':tf.Variable(tf.random_normal([layer_nodes_sizes[0]]))} hidden_layers.append(inputlayer) for counter in range(1,numlayers) : hidden_layers.append({'weights':tf.Variable(tf.random_normal([layer_nodes_sizes[counter-1],layer_nodes_sizes[counter]])), 'biases':tf.Variable(tf.random_normal([layer_nodes_sizes[counter]]))}) hidden_layers.append({'weights':tf.Variable(tf.random_normal([layer_nodes_sizes[numlayers-1],n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}) ln=[]; tmp=tf.add(tf.matmul(data,hidden_layers[0]['weights']),hidden_layers[0]['biases']) ln.append(tf.nn.relu(tmp)) for counter1 in range(1,numlayers) : tmp = tf.add(tf.matmul(ln[counter1-1],hidden_layers[counter1]['weights']),hidden_layers[counter1]['biases']) tmp = tf.nn.relu(tmp) ln.append(tmp) output = tf.add(tf.matmul(ln[numlayers-1],hidden_layers[numlayers]['weights']),hidden_layers[numlayers]['biases']) return output def train_neural_network(x,layer_nodes,n_classes,nm_epochs,batch_size): prediction = nearal_network_model(x,layer_nodes,n_classes) cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) optimizer = tf.train.AdamOptimizer().minimize(cost) with tf.Session() as sess: sess.run(tf.global_variables_initializer()) for epoch in range(nm_epochs): epoch_loss=0 for _ in range(int(mnist.train.num_examples/batch_size)): epoch_x,epoch_y = mnist.train.next_batch(batch_size) _,c = sess.run([optimizer,cost],feed_dict = {x: epoch_x,y: epoch_y}) epoch_loss +=c print('Epoch',epoch,'completed out of', nm_epochs,'loss',epoch_loss) correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)) accuracy = tf.reduce_mean(tf.cast(correct,'float')) print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels})) batch_size= 100 nm_epochs =10 nm_classes =10 layer_nodes = [500,500,500]; train_neural_network(x,layer_nodes,10,nm_epochs,batch_size)﻿

