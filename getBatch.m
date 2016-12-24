function inputs = getBatch(imdb, batch)

images = imdb.images.data(:,:,:, batch);
labels = imdb.images.labels(1, batch);

images = gpuArray(images); % use GPU
inputs = {'data', images, 'label', labels};

end