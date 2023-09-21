# GCNN_NodeClassification
Node classification using GCNN, RNN, LSTM and RL for optimizing the depth of hidden layers in NN


setting up proposed model: The model can be segregated into three parts, the first
part comprises a two-layered RNN (simpleRNN or LSTM or GRU), the second part
consist of a two convolutional layer GCN, and the last part is a Q-table optimizing the
size of the hidden layer for both RNN and GCN. The activation function used for the
inner layers is ReLU, for the final layer is SoftMax and ADAM is the optimizer for both
networks. The learning rate is provided as 0.1, the discount factor for Q-Learning is 0.9,
the dropout value for GCN is settled in [0.4, 0.6], the Q-table is initialized to zero matrix,
and the epoch count is set to 200. The model is approached in steps stated below.
Step 1: Node features are provided as input to RNN for aggregating the crucial node
features.
Step 2: Features attained in step 1 are provided to GCN as input. GCN works in two
folds. First, it aggregates node embedding by considering a subset of the neighbor
node, and then it uses the attained hidden embedding for a node classification task.
Step 3: Q-table samples the hidden layer with different sizes, and the matrix value gets
updated based on the accuracy of the network.
