function active_caffe_mex(gpu_id)
% active_caffe_mex(gpu_id)
    % set gpu in matlab
    if ~(isdeployed) 
      fprintf('Setting caffe\n'); 
      gpuDevice(gpu_id);
      cur_dir = pwd
      caffe_dir = fullfile(pwd, 'external', 'caffe', 'matlab')
      cd(caffe_dir);
      caffe.set_device(gpu_id-1);
      fprintf('Done setting caffe gpu\n'); 
      cd(cur_dir);
    end
end
