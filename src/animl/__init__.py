from animl import animl_to_md
from animl import classifiers
from animl import detect
from animl import file_management
from animl import generator
from animl import inference
from animl import matchypatchy
from animl import megadetector
from animl import models
from animl import multi_species
from animl import plot_boxes
from animl import reid
from animl import sequence_classification
from animl import split
from animl import symlink
from animl import test
from animl import timelapse
from animl import train
from animl import utils
from animl import video_processing
from animl import zooniverse

from animl.animl_to_md import (animl_results_to_md_results,
                               detection_category_id_to_name, main,)
from animl.classifiers import (EfficientNetV2M, EfficientNetV2S, load_model, save_model,)
from animl.detect import (detect_MD_batch, parse_MD, process_image,)
from animl.file_management import (VALID_EXTENSIONS, WorkingDirectory,
                                   active_times, build_file_manifest,
                                   check_file, load_data, save_data,)
from animl.generator import (ImageGenerator, MiewGenerator, ResizeWithPadding,
                             TrainGenerator, manifest_dataloader,
                             reid_dataloader, resize_with_padding,
                             train_dataloader,)
from animl.inference import (get_device, predict_species, softmax,
                             tensor_to_onnx,)
from animl.matchypatchy import (detect, extract_frames, miew_embedding,
                                viewpoint_estimator,)
from animl.megadetector import (CONF_DIGITS, COORD_DIGITS, MegaDetector,
                                convert_yolo_to_xywh, truncate_float,
                                truncate_float_array,)
from animl.models import (AutoShape, Bottleneck, BottleneckCSP, C3, C3Ghost,
                          C3SPP, C3TR, C3x, Classify, Concat, Contract, Conv,
                          CrossConv, DWConv, DWConvTranspose2d, Detect,
                          DetectMultiBackend, Detections, Expand, FILE, Focus,
                          GhostBottleneck, GhostConv, Model, ROOT, SPP, SPPF,
                          TransformerBlock, TransformerLayer, autopad,
                          check_anchor_order, common, parse_model, yolo,)
from animl.multi_species import (multi_species_detection,)
from animl.plot_boxes import (demo_boxes, draw_bounding_boxes, main,
                              plot_all_bounding_boxes,)
from animl.reid import (ArcFaceLossAdaptiveMargin, ArcFaceSubCenterDynamic,
                        ArcMarginProduct, ArcMarginProduct_subcenter,
                        ElasticArcFace, GeM, IMAGE_HEIGHT, IMAGE_WIDTH,
                        MiewIdNet, ViewpointModel, heads, l2_norm, load,
                        miewid, viewpoint,
                        weights_init_classifier, weights_init_kaiming,)
from animl.sequence_classification import (sequence_classification,)
from animl.split import (get_animals, get_empty, train_val_test,)
from animl.symlink import (remove_symlink, symlink_MD, symlink_species,
                           update_labels,)
from animl.test import (main, test,)
from animl.timelapse import (csv_converter,)
from animl.train import (init_seed, main, train, validate,)
from animl.utils import (AUTOINSTALL, Albumentations, BAR_FORMAT, CONFIG_DIR,
                         DATASETS_DIR, EarlyStopping, FILE, FONT, HELP_URL,
                         IMG_FORMATS, InfiniteDataLoader, LOCAL_RANK, LOGGER,
                         LoadImages, LoadImagesAndLabels, LoadStreams,
                         LoadWebcam, ModelEMA, NCOLS, NUM_THREADS, Profile,
                         RANK, ROOT, Timeout, VERBOSE, VID_FORMATS,
                         WorkingDirectory, apply_classifier, augment_hsv,
                         augmentations, autosplit, bbox_iou, box_area,
                         box_candidates, box_iou, check_amp, check_dataset,
                         check_file, check_font, check_git_status,
                         check_img_size, check_imshow, check_online,
                         check_python, check_requirements, check_suffix,
                         check_version, check_yaml, clean_str, clip_coords,
                         coco80_to_coco91_class, colorstr, copy_attr,
                         copy_paste, create_dataloader, create_folder, cutout,
                         dataloaders, dataset_stats, de_parallel, device_count,
                         download, emojis, exif_size, exif_transpose,
                         extract_boxes, file_age, file_date, file_size,
                         find_modules, fitness, flatten_recursive,
                         fuse_conv_and_bn, general, get_hash, get_latest_run,
                         git_describe, gsutil_getsize, hist_equalize,
                         img2label_paths, imread, imshow, imshow_, imwrite,
                         increment_path, init_seeds, initialize_weights,
                         intersect_dicts, is_ascii, is_chinese, is_colab,
                         is_docker, is_kaggle, is_parallel, is_pip,
                         is_writeable, labels_to_class_weights,
                         labels_to_image_weights, letterbox, make_divisible,
                         methods, mixup, model_info, non_max_suppression,
                         one_cycle, print_args, print_mutation, profile, prune,
                         random_perspective, replicate, resample_segments,
                         scale_coords, scale_img, segment2box, segments2boxes,
                         select_device, set_logging, sparsity, strip_optimizer,
                         threaded, time_sync, torch_distributed_zero_first,
                         torch_utils, try_except, url2file, user_config_dir,
                         verify_image_label, xyn2xy, xywh2xyxy, xywhn2xyxy,
                         xyxy2xywh, xyxy2xywhn,)
from animl.video_processing import (extract_frame_single, extract_frames,)
from animl.zooniverse import (connect_to_Panoptes, copy_image,
                              create_SubjectSet, upload_to_Zooniverse,
                              upload_to_Zooniverse_Simple,)

__all__ = ['AUTOINSTALL', 'Albumentations', 'ArcFaceLossAdaptiveMargin',
           'ArcFaceSubCenterDynamic', 'ArcMarginProduct',
           'ArcMarginProduct_subcenter', 'AutoShape', 'BAR_FORMAT',
           'Bottleneck', 'BottleneckCSP', 'C3', 'C3Ghost', 'C3SPP', 'C3TR',
           'C3x', 'CONFIG_DIR', 'CONF_DIGITS', 'COORD_DIGITS', 'Classify',
           'Concat', 'Contract', 'Conv', 'CrossConv', 'DATASETS_DIR', 'DWConv',
           'DWConvTranspose2d', 'Detect', 'DetectMultiBackend', 'Detections',
           'EarlyStopping', 'EfficientNetV2M', 'ElasticArcFace', 'Expand', 'FILE',
           'FONT', 'Focus', 'GeM', 'GhostBottleneck', 'GhostConv', 'HELP_URL',
           'IMAGE_HEIGHT', 'IMAGE_WIDTH', 'IMG_FORMATS', 'ImageGenerator',
           'InfiniteDataLoader', 'LOCAL_RANK', 'LOGGER', 'LoadImages',
           'LoadImagesAndLabels', 'LoadStreams', 'LoadWebcam', 'MegaDetector',
           'MiewGenerator', 'MiewIdNet', 'Model', 'ModelEMA', 'NCOLS',
           'NUM_THREADS', 'Profile', 'RANK', 'ROOT', 'ResizeWithPadding',
           'SPP', 'SPPF', 'Timeout', 'TrainGenerator', 'TransformerBlock',
           'TransformerLayer', 'VALID_EXTENSIONS', 'VERBOSE', 'VID_FORMATS',
           'ViewpointModel', 'WorkingDirectory', 'active_times',
           'animl_results_to_md_results', 'animl_to_md', 'apply_classifier',
           'augment_hsv', 'augmentations', 'autopad', 'autosplit', 'bbox_iou',
           'box_area', 'box_candidates', 'box_iou', 'build_file_manifest',
           'check_amp', 'check_anchor_order', 'check_dataset', 'check_file',
           'check_font', 'check_git_status', 'check_img_size', 'check_imshow',
           'check_online', 'check_python', 'check_requirements',
           'check_suffix', 'check_version', 'check_yaml', 'classifiers',
           'clean_str', 'clip_coords', 'coco80_to_coco91_class', 'colorstr',
           'common', 'connect_to_Panoptes', 'convert_yolo_to_xywh',
           'copy_attr', 'copy_image', 'copy_paste', 'create_SubjectSet',
           'create_dataloader', 'create_folder', 'csv_converter', 'cutout',
           'dataloaders', 'dataset_stats', 'de_parallel', 'demo_boxes',
           'detect', 'detect_MD_batch', 'detection_category_id_to_name',
           'device_count', 'download', 'draw_bounding_boxes', 'emojis',
           'exif_size', 'exif_transpose', 'extract_boxes',
           'extract_frame_single', 'extract_frames', 'file_age', 'file_date',
           'file_management', 'file_size', 'find_modules', 'fitness',
           'flatten_recursive', 'fuse_conv_and_bn', 'general', 'generator',
           'get_animals', 'get_device', 'get_empty', 'get_hash',
           'get_latest_run', 'git_describe', 'gsutil_getsize', 'heads',
           'hist_equalize', 'img2label_paths', 'imread', 'imshow', 'imshow_',
           'imwrite', 'increment_path', 'inference', 'init_seed', 'init_seeds',
           'initialize_weights', 'intersect_dicts', 'is_ascii', 'is_chinese',
           'is_colab', 'is_docker', 'is_kaggle', 'is_parallel', 'is_pip',
           'is_writeable', 'l2_norm', 'labels_to_class_weights',
           'labels_to_image_weights', 'letterbox', 'load', 'load_data',
           'load_model', 'main', 'make_divisible', 'manifest_dataloader',
           'matchypatchy', 'megadetector', 'methods', 'miew_embedding',
           'miewid', 'mixup', 'model_info', 'models', 'multi_species',
           'multi_species_detection', 'non_max_suppression', 'one_cycle',
           'parse_MD', 'parse_model', 'plot_all_bounding_boxes', 'plot_boxes',
           'predict_species', 'print_args', 'print_mutation', 'process_image',
           'profile', 'prune', 'random_perspective', 'reid', 'reid_dataloader',
           'remove_symlink', 'replicate', 'resample_segments',
           'resize_with_padding', 'save_data', 'save_model', 'scale_coords',
           'scale_img', 'segment2box', 'segments2boxes', 'select_device',
           'sequence_classification', 'set_logging', 'softmax', 'sparsity',
           'split', 'strip_optimizer', 'symlink', 'symlink_MD',
           'symlink_species', 'tensor_to_onnx', 'test', 'threaded',
           'time_sync', 'timelapse', 'torch_distributed_zero_first',
           'torch_utils', 'train', 'train_dataloader', 'train_val_test',
           'truncate_float', 'truncate_float_array', 'try_except',
           'update_labels', 'upload_to_Zooniverse',
           'upload_to_Zooniverse_Simple', 'url2file', 'user_config_dir',
           'utils', 'validate', 'verify_image_label', 'video_processing',
           'viewpoint', 'viewpoint_estimator', 'weights_init_classifier',
           'weights_init_kaiming', 'xyn2xy', 'xywh2xyxy', 'xywhn2xyxy',
           'xyxy2xywh', 'xyxy2xywhn', 'yolo', 'zooniverse', 'EfficientNetV2S']
