function imdb = imdb_from_coco(root_dir, image_set,year, flip)

if nargin < 4
    flip = false;
end

cache_file = ['./imdb/cache/imdb_coco_seg_' year '_' image_set];

if flip
    cache_file = [cache_file, '_flip'];
end

try
  fprintf('Loading cache file...\n'); 
  load(cache_file);
  imdb.eval_func = @imdb_eval_coco;
  imdb.roidb_func = @roidb_from_coco;
catch
  fprintf('Cache file not found, building from scratch\n'); 
  %% initialize COCO api (please specify dataType/annType below)
  annTypes = { 'instances', 'captions', 'person_keypoints' };
  dataType=[image_set year]; 
  annType=annTypes{1}; % specify dataType/annType
  ann_dir = [fullfile(root_dir,'annotations/') '%s_%s.json']; 
  annFile=sprintf(ann_dir,annType,dataType);
  coco=CocoApi(annFile);

  %Get imageids
  imgIds= sort(coco.getImgIds());
  imgIds = imgIds(1:10000); 
  coco_pre = ['COCO_' dataType]; 
  %Get annotation ids for each image
  annIds = coco.getAnnIds('imgIds',imgIds); 



  imdb.name = ['coco' year '_' image_set];
  imdb.image_dir = fullfile(root_dir,dataType);
  image_ids_tmp =  cellfun(@(y) sprintf('%s_%s', coco_pre, y), arrayfun(@(x) sprintf('%012d',x), imgIds, 'UniformOutput', false), 'UniformOutput', false); 
  imdb.extension = 'jpg';
  counter = 1;

  %Only use images with annotations for training 
  for i = 1:length(image_ids_tmp)
    id = image_ids_tmp{i}; 
    id = str2num(id(end-11:end)); 
    anns = coco.getAnnIds('imgIds',id);
    fprintf('Checking %d for anns ...', id); 
    if ~(size(anns,1) == 0)
      fprintf('Added %d to imdb\n', id); 
      imdb.image_ids{counter} = image_ids_tmp{i};
      counter = counter + 1; 
    end 
  end

  imdb.flip = flip;
  fprintf('Checking flipped images (creating them if they dont exist)...'); 
  if flip
      image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
      flip_image_at = @(i) sprintf('%s/%s_flip.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
      for i = 1:length(imdb.image_ids)
          if ~exist(flip_image_at(i), 'file')
             im = imread(image_at(i));
             imwrite(fliplr(im), flip_image_at(i));
          end
      end
      img_num = length(imdb.image_ids)*2;
      image_ids = imdb.image_ids;
      imdb.image_ids(1:2:img_num) = image_ids;
      imdb.image_ids(2:2:img_num) = cellfun(@(x) [x, '_flip'], image_ids, 'UniformOutput', false);
      imdb.flip_from = zeros(img_num, 1);
      imdb.flip_from(2:2:img_num) = 1:2:img_num;
  end
  fprintf('Done\n'); 
  cats = coco.loadCats(coco.getCatIds());
  imdb.classes = {cats.name};  
  imdb.num_classes = length(imdb.classes);
  imdb.class_to_id = ...
    containers.Map(imdb.classes, 1:imdb.num_classes);
  imdb.class_ids = 1:imdb.num_classes;

  imdb.coco = coco; 
  fprintf('Storing image sizes...\n'); 
  % VOC specific functions for evaluation and region of interest DB
  imdb.eval_func = @imdb_eval_coco;
  imdb.roidb_func = @roidb_from_coco;
  imdb.image_at = @(i) ...
      sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
  for i = 1:length(imdb.image_ids)
    if mod(i,2) == 0
      img_id = imdb.image_ids{i-1}; 
      img_id = str2num(img_id(end-11:end)); 
      fprintf('Img ID: %d\n', img_id); 
      img = coco.loadImgs( img_id ); 
      imdb.sizes(i, :) = [img.height img.width];
      fprintf('Saved %d with h=%d , w=%d\n', img_id, img.height, img.width); 
    else
      img_id = imdb.image_ids{i}; 
      img_id = str2num(img_id(end-11:end)); 
      fprintf('Img ID: %d\n', img_id); 
      img = coco.loadImgs( img_id ); 
      imdb.sizes(i, :) = [img.height img.width];
      fprintf('Saved %d with h=%d , w=%d\n', img_id, img.height, img.width); 
    end
  end
  whos  
  fprintf('Done\nSaving imdb to cache...');
  tic 
  save(cache_file, 'imdb', '-v7.3');
  etme = toc; 
  fprintf('Done in %f seconds\n', etme);
end
