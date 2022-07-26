from collections import OrderedDict

cfg = OrderedDict()
cfg['use_gpu'] = False
cfg['scut_fbp_dir'] = '/home/manhduy/Documents/dataset/SCUT-FBP/Crop'
cfg['hotornot_dir'] = '/home/manhduy/Documents/dataset/eccv2010_HotOrNot/eccv2010_beauty_data/hotornot_face'
cfg['cv_index'] = 1
cfg['scutfbp5500_base'] = '/home/manhduy/Documents/dataset/SCUT-FBP5500_v2.1/SCUT-FBP5500_v2'

cfg['batch_size'] = 4
cfg['epoch'] = 2
cfg['random_seed'] = 40
cfg['scut-fbp-attractive-label'] = '/home/manhduy/Documents/dataset/SCUT-FBP/Rating_Collection/Attractiveness_label.xlsx'
