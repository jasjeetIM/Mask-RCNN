function model = RES50_Mask_COCO(model)

model.mean_image                                = fullfile(pwd, 'models', 'pre_trained_models', 'vgg_16layers', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', 'pre_trained_models', 'resnet_50layers', 'ResNet-50-model.caffemodel');

% Stride in input image pixels at the last conv layer
model.feat_stride                               = 32;

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', 'rpn_prototxts', 'resnet_50layers_conv', 'solver_60k80k.prototxt');
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', 'rpn_prototxts', 'resnet_50layers_conv', 'test.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;

% rpn test setting
model.stage1_rpn.nms.per_nms_topN               = -1;
model.stage1_rpn.nms.nms_overlap_thres       	= 0.7;
model.stage1_rpn.nms.after_nms_topN         	= 200;

%% stage 1 fast rcnn, inited from pre-trained network
model.stage1_mask.solver_def_file          = fullfile(pwd, 'models', 'mask_rcnn_prototxts', 'resnet_50layers_conv', 'solver_30k40k.prototxt');
model.stage1_mask.test_net_def_file        = fullfile(pwd, 'models', 'mask_rcnn_prototxts', 'resnet_50layers_conv', 'test.prototxt');
model.stage1_mask.init_net_file            = model.pre_trained_net_file;

%% stage 2 rpn, only finetune fc layers
model.stage2_rpn.solver_def_file                = fullfile(pwd, 'models', 'rpn_prototxts', 'resnet_50layers_fc', 'solver_60k80k.prototxt');
model.stage2_rpn.test_net_def_file              = fullfile(pwd, 'models', 'rpn_prototxts', 'resnet_50layers_fc', 'test.prototxt');

% rpn test setting
model.stage2_rpn.nms.per_nms_topN              	= -1;
model.stage2_rpn.nms.nms_overlap_thres       	= 0.7;
model.stage2_rpn.nms.after_nms_topN           	= 200;

%% stage 2 fast rcnn, only finetune fc layers
model.stage2_mask.solver_def_file          = fullfile(pwd, 'models', 'mask_rcnn_prototxts', 'resnet_50layers_fc', 'solver_30k40k.prototxt');
model.stage2_mask.test_net_def_file        = fullfile(pwd, 'models', 'mask_rcnn_prototxts', 'resnet_50layers_fc', 'test.prototxt');

%% final test
model.final_test.nms.per_nms_topN              	= 600; % to speed up nms
model.final_test.nms.nms_overlap_thres       	= 0.7;
model.final_test.nms.after_nms_topN          	= 300;
end
