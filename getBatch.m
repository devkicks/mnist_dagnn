function inputs = getBatch(imdb, batch)
    images = imdb.images.data(:,:,:, batch);
    labels = imdb.images.labels(1, batch);
    
    images = gpuArray(images);
    inputs = {'data', images, 'label', labels};
end