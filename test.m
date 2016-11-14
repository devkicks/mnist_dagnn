netStruct = load('new_net.mat');

net = dagnn.DagNN.loadobj(netStruct);

data = im2single(imread('data/t10k-images/8_900.bmp'));
net.mode = 'test';

net.eval({'data', data});
idx = net.getVarIndex('ip2');
[i j] = max(net.vars(idx).value);

disp(j-1);