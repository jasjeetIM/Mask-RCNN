function mask_COCO_RES50()

% Adopted from code for Faster R-CNN by Shaoqing Ren
% Currently in development

warning('off', 'all');
if ~(isdeployed)
  opts.gpu_id = auto_select_gpu;
  active_caffe_mex(opts.gpu_id);
end

% model
model = Model.RES50_Mask_COCO;
opts.do_val = true;
opts.db = 'coco';
% cache base
cache_base_proposal = 'mask_RES50_COCO';
cache_base_mask = '';

% train/test data
dataset = [];
use_flipped = true;
dataset = Dataset.coco_train(dataset, 'train', use_flipped);
dataset = Dataset.coco_train(dataset, 'test', false);
%% -------------------- TRAIN --------------------
% conf
conf_proposal = proposal_config('image_means', model.mean_image, 'feat_stride', model.feat_stride);
conf_mask = mask_config('image_means', model.mean_image);

% set cache folder for each stage
model = Mask_Train.set_cache_folder(cache_base_proposal, cache_base_mask, model);
% generate anchors and pre-calculate output size of rpn network 
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
whos
%%  stage one proposal
fprintf('\n***************\nstage one proposal \n***************\n');
% train
model.stage1_rpn = Mask_Train.do_proposal_train(conf_proposal, dataset, model.stage1_rpn, opts.do_val);
%whos
% test
dataset.roidb_train = cellfun(@(x, y) Mask_Train.do_proposal_test(conf_proposal, model.stage1_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
%whos
dataset.roidb_test = Mask_Train.do_proposal_test(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);
%whos
%%  stage one mask
fprintf('\n***************\nstage one Mask RCNN\n***************\n');
% train
model.stage1_mask = Mask_Train.do_mask_train(conf_mask, dataset, model.stage1_mask, opts.do_val, opts.db);
% test
%opts.mAP = Mask_Train.do_mask_test(conf_mask, model.stage1_mask, dataset.imdb_test, dataset.roidb_test);

%%  stage two proposal
% net proposal
fprintf('\n***************\nstage two proposal\n***************\n');
% train
model.stage2_rpn.init_net_file = model.stage1_mask.output_model_file;
model.stage2_rpn = Mask_Train.do_proposal_train(conf_proposal, dataset, model.stage2_rpn, opts.do_val);
% test
dataset.roidb_train = cellfun(@(x, y) Mask_Train.do_proposal_test(conf_proposal, model.stage2_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test = Mask_Train.do_proposal_test(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);

%%  stage two mask
fprintf('\n***************\nstage two Mask RCNN \n***************\n');
% train
model.stage2_mask.init_net_file = model.stage1_mask.output_model_file;
model.stage2_mask = Mask_Train.do_mask_train(conf_mask, dataset, model.stage2_mask, opts.do_val, opts.db);

%% final test
fprintf('\n***************\nfinal test\n***************\n');
     
model.stage2_rpn.nms = model.final_test.nms;
%dataset.roidb_test = Mask_Train.do_proposal_test(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);
%opts.final_mAP = Mask_Train.do_mask_test(conf_mask, model.stage2_mask, dataset.imdb_test, dataset.roidb_test);

% save final models, for outside tester
Mask_Train.gather_rpn_mask_models(conf_proposal, conf_mask, model, dataset);
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file);
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  2.^[3:5]);
end
