import os
ninco_folder = '/data/home/mislambhuian/exp/ood_project/multiLayer_maha_norm/NINCO/NINCO'
repo_path = '/data/home/mislambhuian/exp/ood_project/multiLayer_maha_norm'

dset_location_dict = {
    'NINCO': ninco_folder,
    'ImageNet-O': '/data/home/mislambhuian/exp/ood_project/multiLayer_maha_norm/OpenOOD/data/images_largescale/imagenet-o',
    'ImageNet1K': '/data/home/mislambhuian/data/imagenet-1k',
    'ImageNet-LT': '/data/home/mislambhuian/data/imagenetlt',
}

# root_folder_0 = '/data/home/mislambhuian/exp/ood_project/multiLayer_maha_norm/OpenOOD/data/images_largescale'
# dset_location_suffix_dict_0 = {
#     'ImageNet1K': 'imagenet_1k',
# }

# additional_dset_location_dict_0 = {k: os.path.join(root_folder_0, v) for k,v in dset_location_suffix_dict_0.items()}
# dset_location_dict.update(additional_dset_location_dict_0)

dataset_csvs = {
    'NINCO_OOD_classes': 'data/NINCO_OOD_classes.csv',
    'NINCO_OOD_unit_tests': 'data/NINCO_OOD_unit_tests.csv',
    'NINCO_popular_datasets_subsamples': 'data/NINCO_popular_datasets_subsamples.csv',
}

naming_csvs = {
    'NINCO_class_names': 'data/NINCO_class_names.csv',
}