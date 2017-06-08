function res = imdb_eval_coco(cls, boxes, imdb, cache_name, suffix)
% res = imdb_eval_voc(cls, boxes, imdb, suffix)
%   Use the VOCdevkit to evaluate detections specified in boxes
%   for class cls against the ground-truth boxes in the image
%   database imdb. Results files are saved with an optional
%   suffix.

% AUTORIGHTS
  fprintf('No eval done\n'); 
  res.recall = 0.0;
  res.prec = 0.0;
  res.ap = 0.0;
  res.ap_auc = 0.0;
end
