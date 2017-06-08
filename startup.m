function startup()
 fprintf('Entering startup\n'); 
  curdir = fileparts(mfilename('fullpath')); 
    if ~(isdeployed)
    fprintf('Adding paths\n'); 
      addpath(genpath(fullfile(curdir, 'utils')));
      addpath(genpath(fullfile(curdir, 'functions')));
      addpath(genpath(fullfile(curdir, 'bin')));
      addpath(genpath(fullfile(curdir, 'experiments')));
      addpath(genpath(fullfile(curdir, 'imdb')));
      addpath(genpath(fullfile(curdir, 'datasets/coco2014/')));
     fprintf('Starting caffe\n'); 
      caffe_path = fullfile(curdir, 'external', 'caffe', 'matlab'); 
      if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from external/caffe/matlab; See README.md');
      end
      addpath(genpath(caffe_path));
    fprintf('Done\n'); 
    end
    fprintf('mask_rcnn startup done\n');
end
