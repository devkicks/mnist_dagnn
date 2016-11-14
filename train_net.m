function train_net()
    dbstop if error;
    
    if ~exist('train_init.mat', 'file'), create_net(); end
    netStruct = load('train_init.mat');
    init_net = dagnn.DagNN.loadobj(netStruct);

    if ~exist('imdb.mat', 'file'), create_imdb(); end
    imdb = load('imdb.mat');
    
    opts.gpus = [1] ;
    opts.batchsize = 256 ;
    opts.numEpochs = 100 ;
    
    opts.learningRate = 0.001 ;
    opts.weightDecay = 0.0005 ;
    opts.momentum = 0.9 ;
    opts.saveMomentum = true ;
    
    opts.derOutputs = {'objective', 1} ;
    
    [net stats] = cnn_train_dag(init_net, imdb, @getBatch, opts);
    
    net.mode = 'test';
    netStruct = net.saveobj();
    save('new_net.mat', '-struct', 'netStruct');
end