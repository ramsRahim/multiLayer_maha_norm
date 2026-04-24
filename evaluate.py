import os
import csv
import argparse
import datetime
import json
import traceback
import timm
import torchvision.datasets as dset
from scipy.special import softmax
from torch.utils.data.dataset import Dataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils import (extract_features, extract_clip_embeddings, timm_models,
                   fpr_at_tpr, auroc_ood, set_seed,
                   extract_intermediate_features, load_intermediate_features)
import utils
from resnet50 import ResNetNNGUIDE, ResNetSupCon
import data.paths_config
from detection_methods import *
import datasets
import pandas as pd

os.umask(0)  # all created files and directories have full 777 permissions


class OODScore:
    def __init__(self, path_to_imagenet=data.paths_config.dset_location_dict['ImageNet1K'],
                 path_to_cache='model_outputs/cache'):
        self.path_to_cache = path_to_cache
        self.path_to_imagenet = path_to_imagenet
        self.id_dataset_name = self._resolve_id_dataset_name(path_to_imagenet)
        self.clip_quantile = 0.99
        # self.methods = [
        #     'MSP', 'Energy', 'Energy+React', 'ODIN',
        #     'Mahalanobis', 'Mahalanobis_norm',
        #     'Relative_Mahalanobis', 'Relative_Mahalanobis_norm',
        #     'knn', 'MM_plus_plus', 'MM_plus_plus_topk',
        #     'MM_plus_plus_topk_k3',
        #     'MM_plus_plus_topk_cat', 'MM_plus_plus_topk_cat_k3',
        #     'MM_plus_plus_topk_rel',
        #     'MM_plus_plus_topk_erb', 'MM_plus_plus_topk_erb_rel',
        #     'MM_plus_plus_zscore', 'MM_plus_plus_topk2_zscore',
        # ]
        self.methods = [
            'MSP', 'Energy', 'Energy+React', 'ODIN',
            'Mahalanobis', 'Mahalanobis_norm',
            'Relative_Mahalanobis', 'Relative_Mahalanobis_norm',
            'knn', 'MM_plus_plus', 'MM_plus_plus_topk_cat',
        ]
        self.clip_transform = None
        self.val_acc = -99
        self.train_acc = -99

    @staticmethod
    def _resolve_id_dataset_name(path_to_imagenet):
        normalized_path = os.path.abspath(os.path.normpath(path_to_imagenet))
        for dataset_name, dataset_path in data.paths_config.dset_location_dict.items():
            if os.path.abspath(os.path.normpath(dataset_path)) == normalized_path:
                return dataset_name
        return os.path.basename(normalized_path) or normalized_path

    @property
    def metrics_json_path(self):
        return os.path.join(self.path_to_cache, 'scores', 'metrics.json')

    def _load_metrics_index(self):
        if not os.path.exists(self.metrics_json_path):
            return {}
        try:
            with open(self.metrics_json_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f'[Warning] Could not decode metrics JSON: {self.metrics_json_path}')
            return {}

    def _write_metrics_index(self, metrics_index):
        os.makedirs(os.path.dirname(self.metrics_json_path), exist_ok=True)
        tmp_path = f'{self.metrics_json_path}.{os.getpid()}.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(metrics_index, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.metrics_json_path)

    def _metrics_entry(self, metrics_index, model_name, method):
        return (
            metrics_index
            .get(self.id_dataset_name, {})
            .get(self.dataset_out.__name__, {})
            .get(model_name, {})
            .get(method)
        )

    @staticmethod
    def _metrics_entry_complete(entry):
        return entry is not None and entry.get('auroc') is not None and entry.get('fpr_at_95') is not None

    def pending_methods(self, model_name, methods):
        metrics_index = self._load_metrics_index()
        pending = []
        for method in methods:
            entry = self._metrics_entry(metrics_index, model_name, method)
            if self._metrics_entry_complete(entry):
                print(
                    f'[Skip] {method} already evaluated for '
                    f'id={self.id_dataset_name}, ood={self.dataset_out.__name__}, model={model_name}.'
                )
            else:
                pending.append(method)
        return pending

    def _store_method_metrics(self, model, method, method_results, bootstrap_ci):
        metrics_index = self._load_metrics_index()
        id_metrics = metrics_index.setdefault(self.id_dataset_name, {})
        ood_metrics = id_metrics.setdefault(self.dataset_out.__name__, {})
        model_metrics = ood_metrics.setdefault(model.model_name, {})
        model_metrics[method] = {
            'id_dataset': self.id_dataset_name,
            'id_dataset_path': self.path_to_imagenet,
            'ood_dataset': self.dataset_out.__name__,
            'ood_dataset_arg': self.dataset,
            'model': model.model_name,
            'method': method,
            'auroc': float(method_results['ood_classes_mean_auroc']),
            'fpr_at_95': float(method_results['ood_classes_mean_fpr_at_95']),
            'samples_mean_auroc': float(method_results['samples_mean_auroc']),
            'samples_mean_fpr_at_95': float(method_results['samples_mean_fpr_at_95']),
            'bootstrap_ci': {
                'auroc_mean': float(bootstrap_ci['auroc_mean']),
                'auroc_ci': [float(v) for v in bootstrap_ci['auroc_ci']],
                'fpr_mean': float(bootstrap_ci['fpr_mean']),
                'fpr_ci': [float(v) for v in bootstrap_ci['fpr_ci']],
            },
            'val_acc': float(self.val_acc),
            'train_acc': float(self.train_acc),
            'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        self._write_metrics_index(metrics_index)

    def setup(self, dataset, model, ood_dataset_paths_prefix=None, clip_model=False):
        """Load and prepare the data."""
        self.dataset = dataset

        # transform from timm cfg
        config = resolve_data_config({}, model=model, use_test_size=True)
        if clip_model:
            test_transform = self.clip_transform
        else:
            test_transform = create_transform(**config)

        available_OOD_datasets = {
            'NINCO': datasets.NINCO,
            'NINCO_OOD_unit_tests': datasets.NINCOOODUnitTests,
            'NINCO_popular_datasets_subsamples': datasets.NINCOPopularDatasetsSubsamples,
            'ImageNet-O': datasets.ImageNetO,
        }

        train_dir = os.path.join(self.path_to_imagenet, 'train')
        val_dir = os.path.join(self.path_to_imagenet, 'val')
        if not os.path.isdir(train_dir):
            print(f"[Warning] ImageNet train not found at {train_dir}; using val as train fallback.")
            train_dir = val_dir
        self.dataset_in_train = dset.ImageFolder(train_dir, transform=test_transform)
        self.dataset_in_val = dset.ImageFolder(val_dir, transform=test_transform)
        if dataset.endswith('.csv'):
            if ood_dataset_paths_prefix == None:
                self.dataset_out = datasets.ImageCSVDataset(image_table_csv=dataset, transform=test_transform, )
            else:
                self.dataset_out = datasets.ImageCSVDataset(image_table_csv=dataset, paths_prefix=ood_dataset_paths_prefix, transform=test_transform)
        else:
            self.dataset_out = available_OOD_datasets[dataset](transform=test_transform)

    @staticmethod
    def check_complete(path, expected_samples, sources=['features', 'labels_true', 'logits']):
        predictions = {}
        if os.path.exists(path):
            for source in sources:
                print('checking ', source)
                names = sorted([f for f in os.listdir(path) if f.startswith(source + '_') and f.endswith('.npy')
                                and f[len(source + '_'):-len('.npy')].isdigit()])
                if len(names) == 0:
                    print('No samples in {}'.format(path))
                    return False
                predictions[source] = np.concatenate([np.load(os.path.join(path, f)) for f in names])
                print('# samples: ',len(predictions[source]))
                if source == 'text_encoded' and len(predictions[source]) == 1000:
                    continue
                else:
                    if len(predictions[source]) != expected_samples:
                        print(f'There should be {expected_samples} samples of {source} in {path},'
                              f'but there are {len(predictions[source])}')
                        return False
        return predictions

    def get_features_and_logits(self, model, train=True, val=True, ood=True, overwrite='no'):
        if train:
            save_path_train = os.path.join(os.path.join(self.path_to_cache, 'cache_train', model.model_name))
            if overwrite in {'no', 'ood', 'notrain'}:
                predictions_train = self.check_complete(save_path_train, expected_samples=len(self.dataset_in_train))
            else:
                predictions_train = None
            if not predictions_train:
                print('Train features not complete, extracting...')
                extract_features(model, self.dataset_in_train, wo_head=False, savepath=save_path_train)
                predictions_train = self.check_complete(save_path_train, expected_samples=len(self.dataset_in_train))
            self.train_labels = predictions_train['labels_true']
            self.feature_id_train = predictions_train['features'].astype(np.float64)  # [:,:,0,0]
            self.logits_id_train = predictions_train['logits'].astype(np.float64)
            print('Computing softmax...')
            self.softmax_id_train = softmax(self.logits_id_train, axis=-1)
            predicted_classes_train = np.argmax(self.logits_id_train, axis=-1)
            self.train_acc = np.equal(predicted_classes_train, predictions_train['labels_true']).mean()
            print('Accuracy train: ', self.train_acc)
            print('Done')
        if val:
            save_path_val = os.path.join(os.path.join(self.path_to_cache, 'cache_val', model.model_name))
            if overwrite in {'no', 'ood'}:
                predictions_val = self.check_complete(save_path_val, expected_samples=len(self.dataset_in_val))
            else:
                predictions_val = None
            if not predictions_val:
                print('Val features not complete, extracting...')
                extract_features(model, self.dataset_in_val, wo_head=False, savepath=save_path_val)
                predictions_val = self.check_complete(save_path_val, expected_samples=len(self.dataset_in_val))
            self.feature_id_val = predictions_val['features'].astype(np.float64)
            self.logits_id_val = predictions_val['logits'].astype(np.float64)
            self.labels_id_val = predictions_val['labels_true']
            print('Computing softmax...')
            self.softmax_id_val = softmax(self.logits_id_val, axis=-1)
            self.predicted_classes = np.argmax(self.logits_id_val, axis=-1)
            self.val_acc = np.equal(self.predicted_classes, self.labels_id_val).mean()
            print('Accuracy val: ', self.val_acc)
            print('Done')
        if ood:
            save_path_ood = os.path.join(os.path.join(self.path_to_cache, 'cache_ood', model.model_name, self.dataset_out.__name__))
            if overwrite in {'no'}:
                predictions_ood = self.check_complete(save_path_ood, expected_samples=len(self.dataset_out))
            else:
                predictions_ood = None
            if not predictions_ood:
                print('OOD features ({}) not complete, extracting...'.format(self.dataset))
                extract_features(model, self.dataset_out, wo_head=False, savepath=save_path_ood)
                predictions_ood = self.check_complete(save_path_ood, expected_samples=len(self.dataset_out))
            self.feature_ood = predictions_ood['features'].astype(np.float64)
            self.logits_ood = predictions_ood['logits'].astype(np.float64)
            self.labels_ood = predictions_ood['labels_true']
            print('Computing softmax...')
            self.softmax_ood = softmax(self.logits_ood, axis=-1)
            print('Done')

    def get_intermediate_features(self, model, train=True, val=True, ood=True, overwrite='no'):
        """Extract and cache intermediate (per-layer) features for MM++."""
        n_layers = len(utils.get_layer_config(model, model.model_name))

        if train:
            save_path = os.path.join(self.path_to_cache, 'cache_train_inter', model.model_name)
            inter = load_intermediate_features(save_path, n_expected=len(self.dataset_in_train))
            if inter is None or len(inter) != n_layers or overwrite not in {'no', 'ood', 'notrain'}:
                print('[MM++] Train intermediate features not complete, extracting...')
                extract_intermediate_features(model, self.dataset_in_train, save_path)
                inter = load_intermediate_features(save_path, n_expected=len(self.dataset_in_train))
            self.inter_train_path   = save_path
            self.inter_feats_train  = inter  # list of [N_train, D_l] — kept for reference

        if val:
            save_path = os.path.join(self.path_to_cache, 'cache_val_inter', model.model_name)
            inter = load_intermediate_features(save_path, n_expected=len(self.dataset_in_val))
            if inter is None or len(inter) != n_layers or overwrite not in {'no', 'ood'}:
                print('[MM++] Val intermediate features not complete, extracting...')
                extract_intermediate_features(model, self.dataset_in_val, save_path)
                inter = load_intermediate_features(save_path, n_expected=len(self.dataset_in_val))
            self.inter_feats_val = inter  # list of [N_val, D_l]

        if ood:
            save_path = os.path.join(self.path_to_cache, 'cache_ood_inter',
                                     model.model_name, self.dataset_out.__name__)
            inter = load_intermediate_features(save_path, n_expected=len(self.dataset_out))
            if inter is None or len(inter) != n_layers or overwrite not in {'no'}:
                print(f'[MM++] OOD intermediate features ({self.dataset}) not complete, extracting...')
                extract_intermediate_features(model, self.dataset_out, save_path)
                inter = load_intermediate_features(save_path, n_expected=len(self.dataset_out))
            self.inter_feats_ood = inter  # list of [N_ood, D_l]

    def get_features_clip(self, model, train=False, val=True, ood=True, overwrite='no',openclip=False):
        if train:
            save_path_train = os.path.join(os.path.join(self.path_to_cache, 'cache_train', model.model_name))
            if overwrite in {'no', 'ood'}:
                predictions_train = self.check_complete(save_path_train, expected_samples=len(self.dataset_in_train),
                                                      sources=['features', 'labels_true'])
            else:
                predictions_train = None
            if not predictions_train:
                print('train features not complete, extracting...')
                if openclip:
                    extract_openclip_embeddings(model, self.dataset_in_train, text=None, savepath=save_path_train)
                else:
                    extract_clip_embeddings(model, self.dataset_in_train, text=None, savepath=save_path_train)
                predictions_train = self.check_complete(save_path_train, expected_samples=len(self.dataset_in_train),
                                                      sources=['features', 'labels_true'])
            self.feature_id_train = predictions_train['features'].astype(np.float64)
            self.labels_id_train = predictions_train['labels_true']
            self.clip_labels_true = predictions_train['labels_true']
            print('train done.')           
        if val:
            save_path_val = os.path.join(os.path.join(self.path_to_cache, 'cache_val', model.model_name))
            if overwrite in {'no', 'ood'}:
                predictions_val = self.check_complete(save_path_val, expected_samples=len(self.dataset_in_val),
                                                      sources=['features', 'labels_true', 'text_encoded'])
            else:
                predictions_val = None
            if not predictions_val:
                print('Val features not complete, extracting...')
                text_labels = np.load('model_outputs/im_class_clean.npy')
                if openclip:
                    extract_openclip_embeddings(model, self.dataset_in_val, text=text_labels, savepath=save_path_val)
                else:
                    extract_clip_embeddings(model, self.dataset_in_val, text=text_labels, savepath=save_path_val)
                predictions_val = self.check_complete(save_path_val, expected_samples=len(self.dataset_in_val),
                                                      sources=['features', 'labels_true', 'text_encoded'])
            self.feature_id_val = predictions_val['features'].astype(np.float64)
            self.labels_id_val = predictions_val['labels_true']
            self.labels_encoded_clip = predictions_val['text_encoded']
            self.clip_labels_true = predictions_val['labels_true']
            print('Val done.')
        if ood:
            save_path_ood = os.path.join(
                os.path.join(self.path_to_cache, 'cache_ood', model.model_name, self.dataset_out.__name__))
            if overwrite in {'no', }:
                predictions_ood = self.check_complete(save_path_ood, expected_samples=len(self.dataset_out),
                                                      sources=['features', 'labels_true'])
            else:
                predictions_ood = None
            if not predictions_ood:
                print('OOD features ({}) not complete, extracting...'.format(self.dataset_out.__name__))
                if openclip:
                    print('Using openclip...')
                    extract_openclip_embeddings(model, self.dataset_out, savepath=save_path_ood)
                else:
                    extract_clip_embeddings(model, self.dataset_out, savepath=save_path_ood)
                predictions_ood = self.check_complete(save_path_ood, expected_samples=len(self.dataset_out),
                                                      sources=['features', 'labels_true'])
            self.feature_ood = predictions_ood['features'].astype(np.float64)
            self.labels_ood = predictions_ood['labels_true']
            print('OOD done.')

    def evaluate(self, model, OOD_classes, methods=['MSP'], n_bootstrap_seeds=3):
        # patly adapted from https://github.com/haoqiwang/vim/blob/master/benchmark.py
        methods = self.pending_methods(model.model_name, methods)
        if not methods:
            print(
                f'No pending methods for id={self.id_dataset_name}, '
                f'ood={self.dataset_out.__name__}, model={model.model_name}.'
            )
            return {}

        path = os.path.join(self.path_to_cache, 'cache_methods', model.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Extract classifier head weights for Energy+React
        w, b = None, None
        if 'Energy+React' in methods:
            mn = model.model_name
            try:
                if hasattr(model, 'fc'):                     # ResNet
                    w = model.fc.weight.cpu().detach().numpy()
                    b = model.fc.bias.cpu().detach().numpy()
                elif hasattr(model.head, 'fc'):              # ConvNeXt, Swin, SwinV2
                    w = model.head.fc.weight.cpu().detach().numpy()
                    b = model.head.fc.bias.cpu().detach().numpy()
                else:                                        # ViT, DeiT
                    w = model.head.weight.cpu().detach().numpy()
                    b = model.head.bias.cpu().detach().numpy()
            except Exception as e:
                print(f'[Warning] Could not extract w/b for Energy+React: {e}')
                w, b = None, None

        methods_results = {}
        for method in methods:
            try:
                if method == 'MSP':
                    scores_id, scores_ood = evaluate_MSP(self.softmax_id_val, self.softmax_ood)
                elif method == 'Energy':
                    scores_id, scores_ood = evaluate_Energy(self.logits_id_val, self.logits_ood)
                elif method == 'Energy+React':
                    scores_id, scores_ood = evaluate_Energy_React(
                        self.feature_id_train, self.feature_id_val, self.feature_ood, w, b, path)
                elif method == 'ODIN':
                    scores_id, scores_ood = evaluate_ODIN(
                        model, self.dataset_in_val, self.dataset_out, path,
                        T=1000, epsilon=0.0014, batch_size=model.batch_size)
                elif method == 'Mahalanobis':
                    scores_id, scores_ood = evaluate_Mahalanobis(
                        self.feature_id_train, self.feature_id_val, self.feature_ood,
                        self.train_labels, path)
                elif method == 'Mahalanobis_norm':
                    scores_id, scores_ood = evaluate_Mahalanobis_norm(
                        self.feature_id_train, self.feature_id_val, self.feature_ood,
                        self.train_labels, path)
                elif method == 'Relative_Mahalanobis':
                    scores_id, scores_ood = evaluate_Relative_Mahalanobis(
                        self.feature_id_train, self.feature_id_val, self.feature_ood,
                        self.train_labels, path)
                elif method == 'Relative_Mahalanobis_norm':
                    scores_id, scores_ood = evaluate_Relative_Mahalanobis_norm(
                        self.feature_id_train, self.feature_id_val, self.feature_ood,
                        self.train_labels, path)
                elif method == 'knn':
                    scores_id, scores_ood = evaluate_KNN(
                        self.feature_id_train, self.feature_id_val, self.feature_ood, path)
                elif method == 'MM_plus_plus':
                    scores_id, scores_ood = evaluate_MM_plus_plus(
                        train_inter_path=self.inter_train_path,
                        layer_feats_val=self.inter_feats_val,
                        layer_feats_ood=self.inter_feats_ood,
                        train_labels=self.train_labels,
                        path=path,
                    )
                elif method == 'MM_plus_plus_topk':
                    scores_id, scores_ood = evaluate_MM_plus_plus_topk_gating(
                        train_inter_path=self.inter_train_path,
                        layer_feats_val=self.inter_feats_val,
                        layer_feats_ood=self.inter_feats_ood,
                        train_labels=self.train_labels,
                        path=path,
                        K=2,
                    )
                elif method == 'MM_plus_plus_topk_cat':
                    scores_id, scores_ood = evaluate_MM_plus_plus_topk_gating(
                        train_inter_path=self.inter_train_path,
                        layer_feats_val=self.inter_feats_val,
                        layer_feats_ood=self.inter_feats_ood,
                        train_labels=self.train_labels,
                        path=path,
                        K=2,
                        concat=True,
                    )
                elif method == 'MM_plus_plus_topk_cat_k3':
                    scores_id, scores_ood = evaluate_MM_plus_plus_topk_gating(
                        train_inter_path=self.inter_train_path,
                        layer_feats_val=self.inter_feats_val,
                        layer_feats_ood=self.inter_feats_ood,
                        train_labels=self.train_labels,
                        path=path,
                        K=3,
                        concat=True,
                    )
                elif method == 'MM_plus_plus_topk_k3':
                    scores_id, scores_ood = evaluate_MM_plus_plus_topk_gating(
                        train_inter_path=self.inter_train_path,
                        layer_feats_val=self.inter_feats_val,
                        layer_feats_ood=self.inter_feats_ood,
                        train_labels=self.train_labels,
                        path=path,
                        K=3,
                    )
                elif method == 'MM_plus_plus_topk_rel':
                    scores_id, scores_ood = evaluate_MM_plus_plus_topk_gating(
                        train_inter_path=self.inter_train_path,
                        layer_feats_val=self.inter_feats_val,
                        layer_feats_ood=self.inter_feats_ood,
                        train_labels=self.train_labels,
                        path=path,
                        K=2,
                        use_erb=False,
                        relative=True,
                    )
                elif method == 'MM_plus_plus_topk_erb':
                    scores_id, scores_ood = evaluate_MM_plus_plus_topk_gating(
                        train_inter_path=self.inter_train_path,
                        layer_feats_val=self.inter_feats_val,
                        layer_feats_ood=self.inter_feats_ood,
                        train_labels=self.train_labels,
                        path=path,
                        K=2,
                        use_erb=True,
                    )
                elif method == 'MM_plus_plus_zscore':
                    scores_id, scores_ood = evaluate_MM_plus_plus(
                        train_inter_path=self.inter_train_path,
                        layer_feats_val=self.inter_feats_val,
                        layer_feats_ood=self.inter_feats_ood,
                        train_labels=self.train_labels,
                        path=path,
                        zscore=True,
                    )
                elif method == 'MM_plus_plus_topk2_zscore':
                    scores_id, scores_ood = evaluate_MM_plus_plus(
                        train_inter_path=self.inter_train_path,
                        layer_feats_val=self.inter_feats_val,
                        layer_feats_ood=self.inter_feats_ood,
                        train_labels=self.train_labels,
                        path=path,
                        top_k=2,
                        zscore=True,
                    )
                elif method == 'MM_plus_plus_topk_erb_rel':
                    scores_id, scores_ood = evaluate_MM_plus_plus_topk_gating(
                        train_inter_path=self.inter_train_path,
                        layer_feats_val=self.inter_feats_val,
                        layer_feats_ood=self.inter_feats_ood,
                        train_labels=self.train_labels,
                        path=path,
                        K=2,
                        use_erb=True,
                        relative=True,
                    )
                else:
                    raise NotImplementedError(f'Method {method} not implemented.')

                print('s-id finite:',np.isfinite(scores_id).all())
                print('s-ood finite:',np.isfinite(scores_ood).all())
                methods_results[method] = {'scores_id': scores_id,
                                           'scores_ood': scores_ood}

                for c in OOD_classes:
                    class_indices = np.where(self.labels_ood == self.dataset_out.class_to_idx[c])
                    scores_on_ood_class = scores_ood[class_indices]
                    methods_results[method][c] = {'auroc': auroc_ood(scores_id, scores_on_ood_class),
                                                  'fpr_at_95': fpr_at_tpr(scores_id, scores_on_ood_class, 0.95)}
                methods_results[method]['samples_mean_auroc'] = auroc_ood(scores_id, scores_ood)
                methods_results[method]['samples_mean_fpr_at_95'] = fpr_at_tpr(scores_id, scores_ood, 0.95)
                methods_results[method]['ood_classes_mean_auroc'] = np.mean(
                    np.array([methods_results[method][c]['auroc'] for c in OOD_classes]))
                methods_results[method]['ood_classes_mean_fpr_at_95'] = np.mean(
                    np.array([methods_results[method][c]['fpr_at_95'] for c in OOD_classes]))

                auroc_pt  = methods_results[method]['ood_classes_mean_auroc']
                fpr_pt    = methods_results[method]['ood_classes_mean_fpr_at_95']
                ci = bootstrap_ci(scores_id, scores_ood,
                                  seeds=tuple(range(n_bootstrap_seeds)), n_bootstrap=1000)
                methods_results[method]['bootstrap_ci'] = ci
                self._store_method_metrics(model, method, methods_results[method], ci)
                print(
                    '{} on {} evaluated with {}.\n'
                    'Auroc: {:.4f}  95% CI [{:.4f}, {:.4f}]\n'
                    'fpr at 95: {:.4f}  95% CI [{:.4f}, {:.4f}]\n'
                    'accuracy val: {}\n accuracy train: {}'.format(
                        method, self.dataset, model.model_name,
                        auroc_pt,  ci['auroc_ci'][0], ci['auroc_ci'][1],
                        fpr_pt,    ci['fpr_ci'][0],   ci['fpr_ci'][1],
                        self.val_acc, self.train_acc,
                    )
                )
            except Exception as e:
                print(f'[ERROR] Method {method} failed on {model.model_name} / {self.dataset}: {e}')
                traceback.print_exc()
                methods_results.pop(method, None)
                continue
        # save results
        savepath = os.path.join(self.path_to_cache, 'scores', model.model_name, self.dataset_out.__name__)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        eval_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        np.savez(os.path.join(savepath, f'E{eval_time}.npz'), methods_results=methods_results,
                 id_labels=self.labels_id_val, ood_labels=self.labels_ood, ood_classes=OOD_classes,
                 val_acc=self.val_acc, train_acc=self.train_acc)


def bootstrap_ci(scores_id, scores_ood, seeds=(0, 1, 2), n_bootstrap=1000):
    """
    Bootstrap 95% CI for AUROC and FPR@95 across multiple random seeds.

    Each seed draws n_bootstrap independent resamples (with replacement) of the
    ID and OOD score vectors, producing a total of len(seeds)*n_bootstrap estimates.
    The 2.5th / 97.5th percentiles of all estimates form the 95% CI.

    Args:
        scores_id:   [N_id] ID validation scores
        scores_ood:  [N_ood] OOD test scores
        seeds:       iterable of integer RNG seeds (paper requires ≥ 3)
        n_bootstrap: number of resamples per seed

    Returns:
        dict with keys auroc_mean, auroc_ci, fpr_mean, fpr_ci
    """
    aurocs, fprs = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstrap):
            sid = scores_id[rng.integers(0, len(scores_id), len(scores_id))]
            sod = scores_ood[rng.integers(0, len(scores_ood), len(scores_ood))]
            aurocs.append(auroc_ood(sid, sod))
            fprs.append(fpr_at_tpr(sid, sod, 0.95))
    return {
        'auroc_mean': float(np.mean(aurocs)),
        'auroc_ci':   (float(np.percentile(aurocs, 2.5)),
                       float(np.percentile(aurocs, 97.5))),
        'fpr_mean':   float(np.mean(fprs)),
        'fpr_ci':     (float(np.percentile(fprs, 2.5)),
                       float(np.percentile(fprs, 97.5))),
    }


methods_train_usage = {
    'MSP':                      False,
    'Energy':                   False,
    'Energy+React':             True,
    'ODIN':                     False,
    'Mahalanobis':              True,
    'Mahalanobis_norm':         True,
    'Relative_Mahalanobis':     True,
    'Relative_Mahalanobis_norm': True,
    'knn':                      True,
    'MM_plus_plus':             True,
    'MM_plus_plus_topk':        True,
    'MM_plus_plus_topk_k3':     True,
    'MM_plus_plus_topk_cat':    True,
    'MM_plus_plus_topk_cat_k3': True,
    'MM_plus_plus_zscore':      True,
    'MM_plus_plus_topk2_zscore': True,
    'MM_plus_plus_topk_rel':    True,
    'MM_plus_plus_topk_erb':    True,
    'MM_plus_plus_topk_erb_rel': True,
}

parser = argparse.ArgumentParser(description='OOD Evaluation on NINCO')
parser.add_argument('--path_to_weights', default='model_weights', )
parser.add_argument('--model_name', default='convnext_base_in22ft1k')
parser.add_argument('--dataset', type=str, default='NINCO') #choices=['NINCO', 'NINCO_OOD_unit_tests', 'NINCO_popular_datasets_subsamples' ...csv],
parser.add_argument('--dataset_paths_prefix', type=str) #choices=['NINCO', 'NINCO_OOD_unit_tests', 'NINCO_popular_datasets_subsamples' ...csv],
parser.add_argument('--overwrite_model_outputs', type=str, choices=['no', 'all', 'notrain', 'ood'], default='no')
parser.add_argument('--method', default='MM_plus_plus_topk')
parser.add_argument('--path_to_imagenet', default=data.paths_config.dset_location_dict['ImageNet1K'])
parser.add_argument('--path_to_cache', default='./cache')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=99)
parser.add_argument('--n_bootstrap_seeds', type=int, default=3,
                    help='Number of RNG seeds for bootstrap CI (paper requires ≥ 3)')


def main():
    args = parser.parse_args()
    set_seed(args.seed)
    torch.hub.set_dir(args.path_to_weights)
    task = OODScore(path_to_cache=args.path_to_cache, path_to_imagenet=args.path_to_imagenet)
    methods = task.methods if args.method == 'all' else [args.method]
    if args.model_name=='all':
        model_names=list(timm_models.keys())
    else:
        model_names = [args.model_name]
    ood_datasets=['./data/ssb_hard.csv', './data/places365.csv', './data/texture.csv', './data/openimages_o.csv', './data/inaturalist.csv'] if args.dataset=='openood-datasets' else [args.dataset]
    ood_datasets=['NINCO', 'NINCO_OOD_unit_tests', 'NINCO_popular_datasets_subsamples'] if args.dataset=='ninco-datasets' else ood_datasets
    for method in methods:
        _ = methods_train_usage[method]  # raises KeyError if a method is not available
    for ood_dataset_name in ood_datasets:
        current_dataset=ood_dataset_name
        for model_name in model_names:
            if model_name in timm_models.keys():
                model = timm.create_model(**timm_models[model_name]['config']).cuda().eval()
                model.model_name = model_name
                model.batch_size = args.batch_size
                model.project_features = False
                print('Created model {}.'.format(model.model_name))
                task.setup(current_dataset, model, ood_dataset_paths_prefix=args.dataset_paths_prefix, clip_model=False)
                print('Task is set up.')
                pending_methods = task.pending_methods(model.model_name, methods)
                if not pending_methods:
                    print(
                        f'All requested methods already evaluated for '
                        f'id={task.id_dataset_name}, ood={task.dataset_out.__name__}, model={model.model_name}.'
                    )
                    continue
                need_train_outputs = any([methods_train_usage[m] for m in pending_methods])
                task.get_features_and_logits(model, ood=True, train=need_train_outputs,
                                            overwrite=args.overwrite_model_outputs)
                if any(m.startswith('MM_plus_plus') for m in pending_methods):
                    task.get_intermediate_features(model, ood=True, train=True,
                                                   overwrite=args.overwrite_model_outputs)
                OOD_classes = task.dataset_out.classes
                task.evaluate(model, OOD_classes=OOD_classes, methods=pending_methods,
                              n_bootstrap_seeds=args.n_bootstrap_seeds)
                print(f'# ood classes: {len(OOD_classes)}')
            elif model_name=='rn50supcon':
                model = ResNetSupCon()
                # sd = torch.load('/mnt/qb/hein/mmueller67/vkd/resnet50-supcon.pt')
                print('Loading SD')
                sd = torch.load('/mnt/qb/hein/mmueller67/vkd/resnet50-supcon.pt')
                print('Loaded SD')
                model.load_state_dict(sd['model_state_dict'])
                model = model.cuda().eval()
                model.model_name = model_name
                model.batch_size = args.batch_size
                model.project_features = False
                print('Created model {}.'.format(model.model_name))
                task.setup(current_dataset, model, ood_dataset_paths_prefix=args.dataset_paths_prefix, clip_model=False)
                print('Task is set up.')
                pending_methods = task.pending_methods(model.model_name, methods)
                if not pending_methods:
                    print(
                        f'All requested methods already evaluated for '
                        f'id={task.id_dataset_name}, ood={task.dataset_out.__name__}, model={model.model_name}.'
                    )
                    continue
                need_train_outputs = any([methods_train_usage[m] for m in pending_methods])
                task.get_features_and_logits(model, ood=True, train=need_train_outputs,
                                            overwrite=args.overwrite_model_outputs)
                if any(m.startswith('MM_plus_plus') for m in pending_methods):
                    task.get_intermediate_features(model, ood=True, train=True,
                                                   overwrite=args.overwrite_model_outputs)
                #if current_dataset.endswith('.csv'):
                OOD_classes = task.dataset_out.classes
                task.evaluate(model, OOD_classes=OOD_classes, methods=pending_methods,
                              n_bootstrap_seeds=args.n_bootstrap_seeds)
                print(f'# ood classes: {len(OOD_classes)}')
            else:
                raise NotImplementedError(
                    '{} is not implemented. Please add it to the model-dictionary.'.format(model_name))


if __name__ == "__main__":
    with torch.no_grad():
        main()
