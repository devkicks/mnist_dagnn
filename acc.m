imdb = load('imdb.mat');
netStruct = load('new_net.mat');

net = dagnn.DagNN.loadobj(netStruct);
net.move('gpu');
test = find(imdb.images.set == 3);

imdb.images.data = gpuArray(imdb.images.data);

ip2_idx = net.getVarIndex('ip2');
corrent = 0;
for i = 1:length(test)
    idx = test(i);
    net.eval({'data', imdb.images.data(:,:,:, idx)});
    [val prediction] = max(net.vars(ip2_idx).value);
    if prediction == imdb.images.labels(idx)
        corrent = corrent + 1;
    end
end

res = corrent / length(test);
disp(res);