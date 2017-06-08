function save_model_path = mask_train(conf, imdb_train, roidb_train, varargin)
% save_model_path = fast_rcnn_train(conf, imdb_train, roidb_train, varargin)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb_train',                        @iscell);
    ip.addRequired('roidb_train',                       @iscell);
    ip.addParamValue('do_val',          false,          @isscalar);
    ip.addParamValue('imdb_val',        struct(),       @isstruct);
    ip.addParamValue('roidb_val',       struct(),       @isstruct);
    ip.addParamValue('val_iters',       100,            @isscalar); 
    ip.addParamValue('val_interval',    400,           @isscalar); 
    ip.addParamValue('snapshot_interval',...
                                        2000,          @isscalar);
    ip.addParamValue('solver_def_file', fullfile(pwd, 'models', 'mask_rcnn_prototxts/resnet_50layers_conv', 'solver_30k40k.prototxt'), ...
                                                        @isstr);
    ip.addParamValue('net_file',        fullfile(pwd, 'models', 'mask_rcnn_prototxts/resnet_50layers_conv', 'solver_30k40k.prototxt'), ...
                                                        @isstr);
    ip.addParamValue('cache_name',      'mask_RES50_VOC2007', ...
                                                        @isstr);
    
    ip.parse(conf, imdb_train, roidb_train, varargin{:});
    opts = ip.Results;
    
%% try to find trained model
    imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    cache_dir = fullfile(pwd, 'output', 'mask_rcnn_coco_cachedir', opts.cache_name, imdbs_name);
    save_model_path = fullfile(cache_dir, 'final'); 
    if exist(save_model_path, 'file')
        fprintf('Found existing model\n');
        return;
    end
%% init
    % init caffe solver
    mkdir_if_missing(cache_dir);
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);
    caffe_solver = caffe.Solver(opts.solver_def_file);
    caffe_solver.net.copy_from(opts.net_file);
    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['train_', timestamp, '.txt']);
    diary(log_file);
    
    % set random seed
    prev_rng = seed_rand(conf.rng_seed);
    caffe.set_random_seed(conf.rng_seed);
    
    % set gpu/cpu
    if conf.use_gpu
        caffe.set_mode_gpu();
    else
        caffe.set_mode_cpu();
    end
    disp(conf);
    disp(opts);
   fprintf('Preparing data\n');  
%% making tran/val data
    [image_roidb_train, bbox_means, bbox_stds]...
                            = mask_prepare_image_roidb_coco(conf, opts.imdb_train, opts.roidb_train);
    
    
  if opts.do_val
        [image_roidb_val]...
                                = mask_prepare_image_roidb_coco(conf, opts.imdb_val, opts.roidb_val, bbox_means, bbox_stds);
        % fix validation data
        shuffled_inds_val = generate_random_minibatch([], image_roidb_val, conf.ims_per_batch);
        shuffled_inds_val = shuffled_inds_val(randperm(length(shuffled_inds_val), opts.val_iters));
    end
   fprintf('Done\n'); 
%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough  
    num_classes = size(image_roidb_train(1).overlap, 2);
    fprintf('Checking GPU Memory\n'); 
    check_gpu_memory(conf, caffe_solver, num_classes, opts.do_val);
    fprintf('Done\n'); 
%% training
    shuffled_inds = [];
    train_results = [];  
    val_results = [];  
    iter_ = caffe_solver.iter();
    max_iter = caffe_solver.max_iter();
    net_input = {}; 
    while (iter_ < max_iter)
        %caffe_solver.net.set_phase('train');
        % generate minibatch training data
        [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, conf.ims_per_batch);
        [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob, mask_targets_blob] = ...
            mask_get_minibatch(conf, image_roidb_train(sub_db_inds));
        
        im_b = zeros(size(im_blob,1), size(im_blob, 2), size(im_blob,3), size(rois_blob, 4)); 
        for j=1:size(rois_blob,4)
          im_b(:,:,:,j) = im_blob; 
        end

         %net_inputs = {im_b, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob, mask_targets_blob};
        %net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob, mask_targets_blob};
         %Permute data into row major order
        
        %caffe_solver.net.reshape_as_input(net_inputs);

        % one iter SGD update
        %caffe_solver.net.set_input_data(net_inputs);
        %caffe_solver.net.forward(net_inputs);
        for j=1:size(rois_blob,4)
          im_in = im_b(:,:,:,j); 
          rois_in = rois_blob(:,:,:,j); 
          labels_in = labels_blob(:,:,:,j); 
          bb_t = bbox_targets_blob(:,:,:,j); 
          bb_w = bbox_loss_weights_blob(:,:,:,j); 
          mask_in = mask_targets_blob(:,:,:,j); 
          net_in = {im_in, rois_in, labels_in, bb_t, bb_w, mask_in}; 
          caffe_solver.net.reshape_as_input(net_in);
          caffe_solver.net.set_input_data(net_in);
          caffe_solver.step(1);
          rst = caffe_solver.net.get_output();
          loss_curr = caffe_solver.net.blobs('loss_mask').get_data() 
          mask_curr = caffe_solver.net.blobs('conv_mask6').get_data()
          %deconv_curr = caffe_solver.net.blobs('deconv_mask2').get_data()
          train_results = parse_rst(train_results, rst);
        end
        % do valdiation per val_interval iterations
        if ~mod(iter_, opts.val_interval) 
            if opts.do_val
                caffe_solver.net.set_phase('test');                
                for i = 1:length(shuffled_inds_val)
                    sub_db_inds = shuffled_inds_val{i};
                    [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob, mask_targets_blob] = ...
                        mask_get_minibatch(conf, image_roidb_val(sub_db_inds));
                    
                     im_b = zeros(size(im_blob,1), size(im_blob, 2), size(im_blob,3), size(rois_blob, 4)); 
                     for j=1:size(rois_blob,4)
                      im_b(:,:,:,j) = im_blob; 
                     end


                   for j=1:size(rois_blob,4)
                     im_in = im_b(:,:,:,j); 
                     rois_in = rois_blob(:,:,:,j); 
                     labels_in = labels_blob(:,:,:,j); 
                     bb_t = bbox_targets_blob(:,:,:,j); 
                     bb_w = bbox_loss_weights_blob(:,:,:,j); 
                     mask_in = mask_targets_blob(:,:,:,j); 
                     net_in = {im_in, rois_in, labels_in, bb_t, bb_w, mask_in}; 
                     caffe_solver.net.reshape_as_input(net_in);
                     caffe_solver.net.forward(net_in);
                     rst = caffe_solver.net.get_output();
                     val_results = parse_rst(train_results, rst);
                   end
                    % Reshape net's input blobs
                    %net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob, mask_targets_blob};
                    %caffe_solver.net.reshape_as_input(net_inputs);
                    
                    %caffe_solver.net.forward(net_inputs);
                    %rst = caffe_solver.net.get_output();
                    %val_results = parse_rst(val_results, rst);
                end
            end
            show_state(iter_, train_results, val_results);
            train_results = [];
            val_results = [];
            diary; diary; % flush diary
        end
       
        % snapshot
        if ~mod(iter_, opts.snapshot_interval)
            snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
        end
        iter_ = caffe_solver.iter();
    end
    
    % final snapshot
    snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
    save_model_path = snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, 'final');

    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
end

function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, ims_per_batch)

    % shuffle training data per batch
    if isempty(shuffled_inds)
        % make sure each minibatch, only has horizontal images or vertical
        % images, to save gpu memory
        
        hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb_train, 'UniformOutput', true);
        vert_image_inds = ~hori_image_inds;
        hori_image_inds = find(hori_image_inds);
        vert_image_inds = find(vert_image_inds);
        
        % random perm
        lim = floor(length(hori_image_inds) / ims_per_batch) * ims_per_batch;
        hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));
        lim = floor(length(vert_image_inds) / ims_per_batch) * ims_per_batch;
        vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));
        
        % combine sample for each ims_per_batch 
        hori_image_inds = reshape(hori_image_inds, ims_per_batch, []);
        vert_image_inds = reshape(vert_image_inds, ims_per_batch, []);
        shuffled_inds = [hori_image_inds, vert_image_inds];
        shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));
        
        shuffled_inds = num2cell(shuffled_inds, 1);
    end
    
    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end


function check_gpu_memory(conf, caffe_solver, num_classes, do_val)
%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough  
    % generate pseudo training data with max size
    im_blob = single(zeros(max(conf.scales), conf.max_size, 3, conf.ims_per_batch));
    rois_blob = single(repmat([0; 0; 0; max(conf.scales)-1; conf.max_size-1], 1, conf.batch_size));
    rois_blob = permute(rois_blob, [3, 4, 1, 2]);
    labels_blob = single(ones(conf.batch_size, 1));
    labels_blob = permute(labels_blob, [3, 4, 2, 1]);
    bbox_targets_blob = zeros(4 * (num_classes+1), conf.batch_size, 'single');
    bbox_targets_blob = single(permute(bbox_targets_blob, [3, 4, 1, 2])); 
    bbox_loss_weights_blob = bbox_targets_blob;
    
    mask_targets_blob = zeros(conf.batch_size, num_classes+1, 28, 28, 'single'); 
    mask_targets_blob(:) = -1; 
    mask_targets_blob = single(permute(mask_targets_blob, [4, 3, 2, 1]));  

    net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob, mask_targets_blob};
    % Reshape net's input blobs
    caffe_solver.net.reshape_as_input(net_inputs);
    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);
    if do_val
        % use the same net with train to save memory
        caffe_solver.net.set_phase('test');
        caffe_solver.net.forward(net_inputs);
        caffe_solver.net.set_phase('train');
    end
end

function model_path = snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, file_name)
    bbox_stds_flatten = reshape(bbox_stds', [], 1);
    bbox_means_flatten = reshape(bbox_means', [], 1);
    
    % merge bbox_means, bbox_stds into the model
    bbox_pred_layer_name = 'bbox_pred';
    weights = caffe_solver.net.params(bbox_pred_layer_name, 1).get_data();
    biase = caffe_solver.net.params(bbox_pred_layer_name, 2).get_data();
    weights_back = weights;
    biase_back = biase;
    
    weights = ...
        bsxfun(@times, weights, bbox_stds_flatten'); % weights = weights * stds; 
    biase = ...
        biase .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;
    
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase);

    model_path = fullfile(cache_dir, file_name);
    caffe_solver.net.save(model_path);
    fprintf('Saved as %s\n', model_path);
    
    % restore net to original state
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights_back);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase_back);
end

function show_state(iter, train_results, val_results)
    fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
    fprintf('Training : error %.3g, loss (cls %.3g, reg %.3g, mask %.3g)\n', ...
        1 - mean(train_results.accuarcy.data), ...
        mean(train_results.loss_cls.data), ...
        mean(train_results.loss_bbox.data), ...
        mean(train_results.loss_mask.data));
    if exist('val_results', 'var') && ~isempty(val_results)
        fprintf('Testing  : error %.3g, loss (cls %.3g, reg %.3g, mask %.3g)\n', ...
            1 - mean(val_results.accuarcy.data), ...
            mean(val_results.loss_cls.data), ...
            mean(val_results.loss_bbox.data), ...
            mean(val_results.loss_mask.data));
    end
end
