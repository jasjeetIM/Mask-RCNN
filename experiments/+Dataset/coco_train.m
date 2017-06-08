function dataset = coco_train(dataset, usage, use_flip)
% MS COCO train set
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

devkit = coco_devkit(); 

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_coco(devkit, 'train','2014', use_flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false); 
    case {'test'}
        dataset.imdb_test     = imdb_from_coco(devkit, 'val', '2014', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    otherwise
        error('usage = ''train'' or ''test''');
end

end
