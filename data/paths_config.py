import os

_DATA_ROOT = os.environ.get('DATA_ROOT', '/path/to/data')

ninco_folder = os.path.join(_DATA_ROOT, 'NINCO/NINCO')
repo_path    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dset_location_dict = {
    'NINCO':      ninco_folder,
    'ImageNet-O': os.path.join(_DATA_ROOT, 'imagenet-o'),
}

root_folder_0 = _DATA_ROOT
dset_location_suffix_dict_0 = {
    'ImageNet1K': 'imagenet',
}

additional_dset_location_dict_0 = {k: os.path.join(root_folder_0, v) for k, v in dset_location_suffix_dict_0.items()}
dset_location_dict.update(additional_dset_location_dict_0)

dataset_csvs = {
    'NINCO_OOD_classes':               'data/NINCO_OOD_classes.csv',
    'NINCO_OOD_unit_tests':            'data/NINCO_OOD_unit_tests.csv',
    'NINCO_popular_datasets_subsamples': 'data/NINCO_popular_datasets_subsamples.csv',
}

naming_csvs = {
    'NINCO_class_names': 'data/NINCO_class_names.csv',
}
