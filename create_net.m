function net = create_net()
% create init net by dagnn

net = dagnn.DagNN();
net.conserveMemory = false;
convBlock = dagnn.Conv('size', [5 5 1 20], 'hasBias', true);
net.addLayer('conv1', convBlock, {'data'}, {'conv1'}, {'filters1', 'biases1'});

poolBlock = dagnn.Pooling('poolSize', [2 2], 'stride', 2);
net.addLayer('pool1', poolBlock, {'conv1'}, {'pool1'}, {});

convBlock = dagnn.Conv('size', [5 5 20 50], 'hasBias', true);
net.addLayer('conv2', convBlock, {'pool1'}, {'conv2'}, {'filters2', 'biases2'});

poolBlock = dagnn.Pooling('poolSize', [2 2], 'stride', 2);
net.addLayer('pool2', poolBlock, {'conv2'}, {'pool2'}, {});

InnerProductBlock = dagnn.Conv('size', [4 4 50 500], 'hasBias', true);
net.addLayer('ip1', InnerProductBlock, {'pool2'}, {'ip1'}, {'filters3', 'biases3'});

reluBlock = dagnn.ReLU();
net.addLayer('relu1', reluBlock, {'ip1'}, {'relu1'}, {});

InnerProductBlock = dagnn.Conv('size', [1 1 500 10], 'hasBias', true);
net.addLayer('ip2', InnerProductBlock, {'relu1'}, {'ip2'}, {'filters4', 'biases4'});


softBlock = dagnn.Loss();
net.addLayer('loss', softBlock, {'ip2', 'label'}, {'objective'}, {});
net.initParams();

netStruct = net.saveobj();
save('train_init.mat', '-struct', 'netStruct');
end