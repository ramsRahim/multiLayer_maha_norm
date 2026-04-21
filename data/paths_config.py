import os
ninco_folder = '/data/home/mislambhuian/exp/ood_project/multiLayer_maha_norm/NINCO/NINCO'
repo_path = '/data/home/mislambhuian/exp/ood_project/multiLayer_maha_norm'

dset_location_dict = {
    'NINCO': ninco_folder,
    'ImageNet-O': '/data/home/mislambhuian/data/imagenet-o',
    'ImageNet1K': '/data/home/mislambhuian/data/imagenet-1k',
}

dataset_csvs = {
    'NINCO_OOD_classes': 'data/NINCO_OOD_classes.csv',
    'NINCO_OOD_unit_tests': 'data/NINCO_OOD_unit_tests.csv',
    'NINCO_popular_datasets_subsamples': 'data/NINCO_popular_datasets_subsamples.csv',
}

naming_csvs = {
    'NINCO_class_names': 'data/NINCO_class_names.csv',
}