function roidb = roidb_from_coco(imdb, varargin)
% roidb = roidb_from_voc(imdb, rootDir)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addParamValue('rootDir',                         '.',    @ischar);
ip.addParamValue('extension',                       '',     @ischar);
ip.parse(imdb, varargin{:});
opts = ip.Results;
whos
roidb.name = imdb.name;
if ~isempty(opts.extension)
    opts.extension = ['_', opts.extension];
end

cache_file = fullfile(opts.rootDir, ['/imdb/cache/roidb_seg_' imdb.name opts.extension]);

if imdb.flip
    cache_file = [cache_file '_flip'];
end

cache_file = [cache_file, '.mat'];

try
  fprintf('Found the db cache file\n'); 
  load(cache_file);
catch
  fprintf('No cache file, creating db from scrach...\n'); 
  regions = [];
  regions.boxes = cell(length(imdb.image_ids), 1);
  if imdb.flip
    regions.images = imdb.image_ids(1:2:end);
  else
    regions.images = imdb.image_ids;
  end
  fprintf('Creating roidb...\n'); 
  if ~imdb.flip
      for i = 1:length(imdb.image_ids)
        %tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
        try
          fprintf('Getting image id\n'); 
          id = imdb.image_ids{i}; 
          id = str2num(id(end-11:end)); 
          anns = imdb.coco.getAnnIds('imgIds',id)
        catch
          anns = [];
        end
        if ~isempty(regions)
            [~, image_name1] = fileparts(imdb.image_ids{i});
            [~, image_name2] = fileparts(regions.images{i});
            assert(strcmp(image_name1, image_name2));
        end
        roidb.rois(i) = attach_proposals(anns, imdb.coco, imdb.class_to_id, false); 
      end
  else
      for i = 1:length(imdb.image_ids)/2
        if mod(i,500) == 0
          fprintf('Completed %d percent\n', i/(length(imdb.image_ids)/2)); 
        end
        %tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids)/2);
         try
          id = imdb.image_ids{i*2-1}; 
          id = str2num(id(end-11:end));
          fprintf('Checking %d for anns...', id); 
          anns = imdb.coco.getAnnIds('imgIds',id); 
         catch
          anns = [];
         end
        if size(anns,1) == 0
          fprintf('******************************Emtpy annotation\n'); 
        end
        if ~isempty(regions)
            [~, image_name1] = fileparts(imdb.image_ids{i*2-1});
            [~, image_name2] = fileparts(regions.images{i});
            assert(strcmp(image_name1, image_name2));
            assert(imdb.flip_from(i*2) == i*2-1);
        end
        roidb.rois(i*2-1) = attach_proposals(anns,imdb.coco, imdb.class_to_id, false); 
        roidb.rois(i*2) = attach_proposals(anns, imdb.coco, imdb.class_to_id, true);
      end
  end

  fprintf('Saving roidb to cache...');
  tic
  save(cache_file, 'roidb', '-v7.3');
  etme = toc; 
  fprintf('done in %f seconds\n', etme);

end
end

% ------------------------------------------------------------------------
function rec = attach_proposals(anns,coco, class_to_id,flip)
% ------------------------------------------------------------------------
  gt_boxes = [];  
  gt_classes = []; 
  gt_gt = []; 
  for i=1:length(anns)
    ann = coco.loadAnns(anns(i)); 
    bbox = ann.bbox; 
    bbox(1) = floor(bbox(1)); 
    bbox(2) = floor(bbox(2)); 
    bbox(3) = ceil(bbox(1) + bbox(3)); 
    bbox(4) = ceil(bbox(2) + bbox(4));
    if flip
      img_id = ann.image_id; 
      img = coco.loadImgs(img_id) ; 
      h = img.height; 
      w = img.width; 
      bbox([1,3]) = w + 1 - bbox([3,1]); 
    end
    cat = coco.loadCats(ann.category_id); 
    class = class_to_id(cat.name); 
    gt_boxes = [gt_boxes; bbox]; 
    gt_classes = [gt_classes; class];
    gt_gt = [gt_gt; 1];
  end 

  rec.gt = gt_gt;
  if size(rec.gt,1) == 0
    fprintf('***************************Empty GT\n'); 
  end
  rec.overlap = zeros(size(gt_boxes,1), class_to_id.Count, 'single');
  for i = 1:size(gt_boxes,1)
    rec.overlap(:, gt_classes(i)) = ...
        max(rec.overlap(:, gt_classes(i)), boxoverlap(gt_boxes, gt_boxes(i, :)));
  end
  rec.boxes = single(gt_boxes);
  rec.feat = []; 
  rec.class = uint8(gt_classes); 
end
