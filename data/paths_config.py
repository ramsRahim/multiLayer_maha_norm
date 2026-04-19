import os
ninco_folder = '/home/rahim/exp/Data/NINCO/NINCO'
repo_path = '/home/rahim/exp/multiLayer_maha_norm'

dset_location_dict = {
    'NINCO': ninco_folder,
    'ImageNet-O': '/home/rahim/exp/Data/imagenet-o',
}

root_folder_0 = '/home/rahim/exp/Data'
dset_location_suffix_dict_0 = {
    'ImageNet1K': 'imagenet',
}

additional_dset_location_dict_0 = {k: os.path.join(root_folder_0, v) for k,v in dset_location_suffix_dict_0.items()}
dset_location_dict.update(additional_dset_location_dict_0)

dataset_csvs = {
    'NINCO_OOD_classes': 'data/NINCO_OOD_classes.csv',
    'NINCO_OOD_unit_tests': 'data/NINCO_OOD_unit_tests.csv',
    'NINCO_popular_datasets_subsamples': 'data/NINCO_popular_datasets_subsamples.csv',
}

naming_csvs = {
    'NINCO_class_names': 'data/NINCO_class_names.csv',
}