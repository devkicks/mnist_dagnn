function imdb = create_imdb()
% make data to imdb
trainDir = fullfile('data', 't10k-images', '*.bmp');
tmp = dir(trainDir);

data = [];
labels = [];
sets = [];
for i = 1:length(tmp)
    s_str = regexp(tmp(i).name, '.bmp', 'split');
    s_str = regexp(s_str{1}, '_', 'split');
    if(str2num(s_str{2}) < 800)
        % train data
        imgDir = fullfile('data', 't10k-images', tmp(i).name);
        data(:,:,:, end+1) = im2single(imread(imgDir));
        labels(end+1) = str2num(s_str{1}) + 1;
        sets(end+1) = 1;
    else
        % test data
        imgDir = fullfile('data', 't10k-images', tmp(i).name);
        data(:,:,:, end+1) = im2single(imread(imgDir));
        labels(end+1) = str2num(s_str{1}) + 1;
        sets(end+1) = 3;
    end
    
end

imdb.images.data = single(data);
imdb.images.labels = labels;
imdb.images.set = sets;

images = imdb.images;
save('imdb.mat', 'images');
end