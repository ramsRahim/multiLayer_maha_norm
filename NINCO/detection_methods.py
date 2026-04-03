import os
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from scipy.linalg import cholesky
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from numpy.linalg import norm, pinv
from tqdm import tqdm
import torch
import torch.nn.functional as F
from itertools import groupby
import faiss
from sklearn.neighbors import NearestNeighbors
import scipy
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
import copy 

def get_cholesky(mat, epsilon=1e-10, max_attempts=10):
    attempts = 0
    while attempts < max_attempts:
        try:
            L = cholesky(mat, lower=True)
            return L
        except np.linalg.LinAlgError:
            print(f'Not PD, adding epsilon={epsilon:.1e} to the diagonal')
            mat = mat + epsilon * np.eye(mat.shape[0])
            epsilon *= 10
            attempts += 1
    
    raise ValueError('Matrix is not positive definite even after multiple attempts.')


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


# much of the following is built upon code from https://github.com/haoqiwang/vim/blob/master/benchmark.py

def evaluate_MSP(softmax_id_val, softmax_ood):
    """
    Evaluate Maximum Softmax Probability (MSP) for given softmax values of in-distribution (id) and out-of-distribution (ood) data.

    Inputs:
        softmax_id_val: Numpy array of shape (m, n), representing the softmax probabilities of m data points from in-distribution.
        softmax_ood: Numpy array of shape (p, n), representing the softmax probabilities of p data points from out-of-distribution.
        
    Outputs:
        score_id: Numpy array of shape (m,), representing the maximum softmax probability of m data points from in-distribution.
        score_ood: Numpy array of shape (p,), representing the maximum softmax probability of p data points from out-of-distribution.
        """
    score_id = softmax_id_val.max(axis=-1)
    score_ood = softmax_ood.max(axis=-1)
    return score_id, score_ood


def evaluate_MaxLogit(logits_in_distribution, logits_out_of_distribution):
    """Compute the maximum logit value for both in- and out-of-distribution data.

    Args:
        logits_in_distribution (ndarray): Logits for the in-distribution data.
        logits_out_of_distribution (ndarray): Logits for the out-of-distribution data.

    Returns:
        tuple: Tuple of the maximum logit value for both in- and out-of-distribution data.
    """
    score_in_distribution = logits_in_distribution.max(axis=-1)
    score_out_of_distribution = logits_out_of_distribution.max(axis=-1)
    return score_in_distribution, score_out_of_distribution


def evaluate_nan(features_in_distribution, features_out_of_distribution, use_pos=False):
    """Compute the maximum logit value for both in- and out-of-distribution data.

    Args:
        logits_in_distribution (ndarray): Logits for the in-distribution data.
        logits_out_of_distribution (ndarray): Logits for the out-of-distribution data.

    Returns:
        tuple: Tuple of the maximum logit value for both in- and out-of-distribution data.
        
    """

    l1_norm_id = np.sum(np.abs(features_in_distribution), axis=1)
    l0_norm_id = np.count_nonzero(features_in_distribution, axis=1)
    n_pos_id = np.sum(features_in_distribution>0,axis=1)

    l1_norm_ood = np.sum(np.abs(features_out_of_distribution), axis=1)
    l0_norm_ood = np.count_nonzero(features_out_of_distribution, axis=1)
    n_pos_ood = np.sum(features_out_of_distribution>0,axis=1)

    score_in_distribution = (l1_norm_id/n_pos_id) if use_pos else (l1_norm_id/l0_norm_id)
    score_out_of_distribution = (l1_norm_ood/n_pos_ood) if use_pos else (l1_norm_ood/l0_norm_ood)
    return score_in_distribution, score_out_of_distribution


def evaluate_Energy(logits_in_distribution, logits_out_of_distribution):
    """Compute the energy value for both in- and out-of-distribution data.

    Args:
        logits_in_distribution (ndarray): Logits for the in-distribution data.
        logits_out_of_distribution (ndarray): Logits for the out-of-distribution data.

    Returns:
        tuple: Tuple of the energy value for both in- and out-of-distribution data.
    """
    score_in_distribution = logsumexp(logits_in_distribution, axis=1)
    score_out_of_distribution = logsumexp(logits_out_of_distribution, axis=1)
    return score_in_distribution, score_out_of_distribution

def softmax_scaled_cosine(features, w,scale=1):
    out = torch.from_numpy(features).float()
    x_norm = F.normalize(out)
    w_norm = F.normalize(w)
    w_norm_transposed = torch.transpose(w_norm, 0, 1)
    
    # cos_sim = torch.mm(out,w.T)+b#
    cos_sim = torch.mm(x_norm, w_norm_transposed) # cos_theta
    scaled_cosine = cos_sim * scale
    softmax = F.softmax(scaled_cosine, -1)
    return softmax.numpy(), scaled_cosine.numpy()

def evaluate_softmax_scaled_cosine(feature_id_val, feature_ood, w, s=1):
    with torch.no_grad():
        softmax_sc_val, cos_val = softmax_scaled_cosine(feature_id_val,torch.from_numpy(w),scale=s)
        softmax_sc_ninco, cos_ninco = softmax_scaled_cosine(feature_ood,torch.from_numpy(w),scale=s)
    return softmax_sc_val.max(-1),softmax_sc_ninco.max(-1)

def evaluate_ViM(feature_id_train, feature_id_val, feature_ood, logits_id_train, logits_id_val, logits_ood, u, path):
    """
    This function evaluates the performance of the ViM out-of-distribution detection method.

    Inputs:

        feature_id_train: numpy array of shape (n, d), the training set features for the in-distribution data.
        feature_id_val: numpy array of shape (m, d), the validation set features for the in-distribution data.
        feature_ood: numpy array of shape (p, d), the features for the out-of-distribution data.
        logits_id_train: numpy array of shape (n, k), the logits for the in-distribution training set.
        logits_id_val: numpy array of shape (m, k), the logits for the in-distribution validation set.
        logits_ood: numpy array of shape (p, k), the logits for the out-of-distribution data.
        u: numpy array of shape (d,), the mean feature vector.
        path: string, the path to store intermediate results.

    Outputs:

        score_id: numpy array of shape (m,), the ViM scores for the in-distribution validation set.
        score_ood: numpy array of shape (p,), the ViM scores for the out-of-distribution data.
        """
    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else (
        512 if feature_id_val.shape[-1] >= 768 else int(feature_id_val.shape[-1] / 2))
    print(f'{DIM=}')
    print('Reading alpha and NS')
    alpha_path = os.path.join(path, 'alpha.npy')
    NS_path = os.path.join(path, 'NS.npy')
    if os.path.exists(NS_path):
        NS = np.load(NS_path)
    else:
        print('NS not stored, computing principal space...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
        np.save(NS_path, NS)
    if os.path.exists(alpha_path):
        alpha = np.load(alpha_path)
    else:
        print('alpha not stored, computing alpha...')
        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        alpha = logits_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        np.save(alpha_path, alpha)
    print(f'{alpha=:.4f}')

    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logits_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val

    energy_ood = logsumexp(logits_ood, axis=-1)
    vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
    score_ood = -vlogit_ood + energy_ood
    return score_id, score_ood


def evaluate_gen(sf_id, sf_ood, gamma=0.1, M=100):
    def generalized_entropy(softmax_id_val, gamma, M):
            probs =  softmax_id_val 
            probs_sorted = np.sort(probs, axis=1)[:,-M:]
            scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
            return -scores 
    score_id = generalized_entropy(sf_id, gamma=gamma, M=M)
    score_ood = generalized_entropy(sf_ood, gamma=gamma, M=M)

    return score_id,score_ood

def evaluate_fdbd(features_train, features_val, features_ninco,logits_val,logits_ninco,w,):
    device='cuda:0'
    
    feat_log_train=torch.from_numpy(features_train).to(device)
    feat_log_val=torch.from_numpy(features_val).to(device)
    feat_log=torch.from_numpy(features_ninco).to(device)
    score_log_val=torch.from_numpy(logits_val).to(device)
    score_log=torch.from_numpy(logits_ninco).to(device)
    w=torch.from_numpy(w)

    train_mean = torch.mean(feat_log_train, dim= 0).to(device)

    denominator_matrix = torch.zeros((1000,1000)).to(device)
    for p in range(1000):
        w_p = w - w[p,:]
        denominator = torch.norm(w_p, dim=1)
        denominator[p] = 1
        denominator_matrix[p, :] = denominator

    #################### fDBD score OOD detection #################

    values, nn_idx = score_log_val.max(1)
    logits_sub = torch.abs(score_log_val - values.repeat(1000, 1).T)
    #pdb.set_trace()
    score_in = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log_val - train_mean , dim = 1)
    score_in = score_in.float().cpu().numpy()

    values, nn_idx = score_log.max(1)
    logits_sub = torch.abs(score_log - values.repeat(1000, 1).T)
    scores_out_test = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log - train_mean , dim = 1)
    scores_out_test = scores_out_test.float().cpu().numpy()

    return score_in, scores_out_test

def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def gmm_fit(embeddings, labels, num_classes,JITTERS):
    with torch.no_grad():
        classwise_mean_features=[]
        classwise_cov_features=[]
        for c in range(num_classes):
            mean_f = torch.mean(embeddings[labels == c], dim=0)
            f = embeddings[labels == c]
            classwise_mean_features.append(mean_f)
            classwise_cov_features.append(centered_cov_torch(f-mean_f))
    classwise_mean_features=torch.stack(classwise_mean_features)
    classwise_cov_features=torch.stack(classwise_cov_features)
    with torch.no_grad():
        jitter_eps=0
        attempts=1
        # while attempts < max_attempts:
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1], device=classwise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
                )
                return gmm, jitter_eps        
            except:
                continue

    raise ValueError(f'Matrix is not positive definite even after multiple attempts, jitter={jitter_eps}')

def get_lob_probs(features,gmm,batch_size=256):
    num_batches = (features.shape[0] + batch_size - 1) // batch_size  # To cover all elements in features_val
    print(num_batches)
    log_probs = []
    with torch.no_grad():
        for i in range(num_batches):
            batch = (features[i * batch_size: (i + 1) * batch_size])[:, None, :]
            log_probs_batch = gmm.log_prob(batch)
            log_probs.append(log_probs_batch)
            if i%50==0:
                print(f"{i}/{num_batches} batches processed.")
    log_probs = torch.cat(log_probs, dim=0)
    return log_probs

def evaluate_gmm(feature_id_train, feature_id_val, feature_ood,train_labels,path,normalize=False):
    DOUBLE_INFO = torch.finfo(torch.double)
    JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-15, 0, 1)]
    normalizer = lambda x: x/np.linalg.norm(x,axis=-1,keepdims=True)

    f_val = normalizer(feature_id_val) if normalize else feature_id_val
    f_ood = normalizer(feature_ood) if normalize else feature_ood

    gmm_path = os.path.join(path, f'gmm{"_norm" if normalize else ""}.pkl')
    if os.path.exists(gmm_path):
        print('Loading GMM')
        with open(gmm_path, 'rb') as file:
            gmm = pickle.load(file)
    else:    
        print('Fitting GMM...')
        f_train = normalizer(feature_id_train) if normalize else feature_id_train
        gmm, jitter_eps = gmm_fit(torch.from_numpy(f_train),train_labels,1000,JITTERS)
        print(f"Done. Used jitter={jitter_eps}")

        gmmPickle = open(gmm_path, 'wb') 
        pickle.dump(gmm, gmmPickle)  
        gmmPickle.close()


    score_id_gmm_path = os.path.join(path, f'scores_id_gmm{"_norm" if normalize else ""}.npy')
    if os.path.exists(score_id_gmm_path):
        print('Loading ID scores')
        score_id = np.load(score_id_gmm_path)
    else:
        log_probs_val=get_lob_probs(torch.from_numpy(f_val),gmm)
        score_id=log_probs_val.max(-1)[0].numpy()
        np.save(score_id_gmm_path, score_id)

    log_probs_ood=get_lob_probs(torch.from_numpy(f_ood),gmm)
    score_ood = log_probs_ood.max(-1)[0].numpy()

    return score_id, score_ood

def evaluate_neco(feature_id_train, feature_id_val, feature_ood, logit_id_val, logit_ood, path,use_logit=True):
    '''
    Prints the auc/fpr result for NECO method, with adaptive selection of neco_dim to explain 90% of the variance.

            Parameters:
                    feature_id_train (array): An array of training samples features
                    feature_id_val (array): An array of evaluation samples features
                    feature_ood (array): An array of OOD samples features
                    logit_id_val (array): An array of evaluation samples logits
                    logit_ood (array): An array of OOD samples logits
                    model_architecture_type (string): Module architecture used

            Returns:
                    score_id (array): Scores for in-distribution samples
                    score_ood (array): Scores for out-of-distribution samples
    '''
    ss_path = os.path.join(path, 'ss_neco.pkl')
    if os.path.exists(ss_path):
        with open(ss_path, 'rb') as file:
            ss = pickle.load(file)
            complete_vectors_train = ss.transform(feature_id_train)
    else:
        ss = StandardScaler()
        complete_vectors_train = ss.fit_transform(feature_id_train)

        ssPickle = open(ss_path, 'wb') 
        pickle.dump(ss, ssPickle)  
        ssPickle.close()
        
    complete_vectors_test = ss.transform(feature_id_val)
    complete_vectors_ood = ss.transform(feature_ood)

    # Fit PCA to the training data
    pca_path = os.path.join(path, 'pca_neco.pkl')
    if os.path.exists(pca_path):
        print('Loading PCA...')
        with open(pca_path, 'rb') as file:
            pca_estimator = pickle.load(file)
    else:
        print('Fitting PCA...')
        print(complete_vectors_train.shape)
        print(complete_vectors_ood.shape)
        print(complete_vectors_test.shape)
        pca_estimator = PCA()
        pca_estimator.fit(complete_vectors_train)

        pcaPickle = open(pca_path, 'wb') 
        pickle.dump(pca_estimator, pcaPickle)  
        pcaPickle.close()

    # Calculate the cumulative explained variance ratio
    cumulative_variance = np.cumsum(pca_estimator.explained_variance_ratio_)

    # Find the number of components that explain at least 90% of the variance
    neco_dim = np.where(cumulative_variance >= 0.90)[0][0] + 1

    # Perform PCA transformation
    cls_test_reduced_all = pca_estimator.transform(complete_vectors_test)
    cls_ood_reduced_all = pca_estimator.transform(complete_vectors_ood)

    score_id_maxlogit = logit_id_val.max(axis=-1)
    score_ood_maxlogit = logit_ood.max(axis=-1)

    # # If model architecture is one of 'deit' or 'swin', skip PCA
    # if model_architecture_type in ['deit', 'swin']:
    #     complete_vectors_train = feature_id_train
    #     complete_vectors_test = feature_id_val
    #     complete_vectors_ood = feature_ood

    # Select the reduced dimensions for in-distribution and OOD data
    cls_test_reduced = cls_test_reduced_all[:, :neco_dim]
    cls_ood_reduced = cls_ood_reduced_all[:, :neco_dim]

    l_ID = []
    l_OOD = []
    print('Computing scores...')
    # Compute ID scores based on the norm of reduced vectors
    for i in range(cls_test_reduced.shape[0]):
        sc_complet = LA.norm(complete_vectors_test[i, :])
        sc = LA.norm(cls_test_reduced[i, :])
        sc_finale = sc / sc_complet
        l_ID.append(sc_finale)

    # Compute OOD scores based on the norm of reduced vectors
    for i in range(cls_ood_reduced.shape[0]):
        sc_complet = LA.norm(complete_vectors_ood[i, :])
        sc = LA.norm(cls_ood_reduced[i, :])
        sc_finale = sc / sc_complet
        l_OOD.append(sc_finale)

    l_OOD = np.array(l_OOD)
    l_ID = np.array(l_ID)

    # Return the final scores
    score_id = l_ID* score_id_maxlogit if use_logit else l_ID
    score_ood = l_OOD*score_ood_maxlogit if use_logit else l_OOD

    return score_id, score_ood


def evaluate_Mahalanobis_norm(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    """
    This function computes Mahalanobis scores for in-distribution and out-of-distribution samples.

    Parameters:
    feature_id_train (numpy array): The in-distribution training samples.
    feature_id_val (numpy array): The in-distribution validation samples.
    feature_ood (numpy array): The out-of-distribution samples.
    train_labels (numpy array): The labels of the in-distribution training samples.
    path (str): The path to save and load the mean and precision matrix.

    Returns:
    tuple: The Mahalanobis scores for in-distribution validation and out-of-distribution samples.

    """
    feature_id_val = feature_id_val/np.linalg.norm(feature_id_val,axis=-1,keepdims=True)
    feature_ood = feature_ood/np.linalg.norm(feature_ood,axis=-1,keepdims=True)
    # load mean and prec
    mean_path = os.path.join(path, 'mean_norm.npy')
    prec_path = os.path.join(path, 'prec_norm.npy')
    complete = True
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        complete = False
    if os.path.exists(prec_path):
        prec = np.load(prec_path)
    else:
        complete = False
    if not complete:
        print('not complete, computing classwise mean feature...')
        feature_id_train = feature_id_train/np.linalg.norm(feature_id_train,axis=-1,keepdims=True)
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = np.array(train_means)
        prec = (ec.precision_)
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    print('go to gpu...')
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()
    print('Computing scores...')
    score_id_path = os.path.join(path, 'maha_id_scores_norm.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
        print('Loaded ID scores.')
    else:
        score_id = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                              tqdm(torch.from_numpy(feature_id_val).cuda().double())])
        np.save(score_id_path, score_id)
        print('Computed ID scores.')
    score_ood = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                           tqdm(torch.from_numpy(feature_ood).cuda().double())])
    return score_id, score_ood

def evaluate_Mahalanobis(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    """
    This function computes Mahalanobis scores for in-distribution and out-of-distribution samples.

    Parameters:
    feature_id_train (numpy array): The in-distribution training samples.
    feature_id_val (numpy array): The in-distribution validation samples.
    feature_ood (numpy array): The out-of-distribution samples.
    train_labels (numpy array): The labels of the in-distribution training samples.
    path (str): The path to save and load the mean and precision matrix.

    Returns:
    tuple: The Mahalanobis scores for in-distribution validation and out-of-distribution samples.

    """
    # load mean and prec
    mean_path = os.path.join(path, 'mean.npy')
    prec_path = os.path.join(path, 'prec.npy')
    complete = True
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        complete = False
    if os.path.exists(prec_path):
        prec = np.load(prec_path)
    else:
        complete = False
    if not complete:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = np.array(train_means)
        prec = (ec.precision_)
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    print('go to gpu...')
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()
    print('Computing scores...')
    score_id_path = os.path.join(path, 'maha_id_scores.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        score_id = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                              tqdm(torch.from_numpy(feature_id_val).cuda().double())])
        np.save(score_id_path, score_id)

    score_ood = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                           tqdm(torch.from_numpy(feature_ood).cuda().double())])
    return score_id, score_ood

def evaluate_Relative_Mahalanobis_norm(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    """
      This function computes the relative Mahalanobis scores for in-distribution and out-of-distribution samples.

      Parameters:
      feature_id_train (numpy array): The in-distribution training samples.
      feature_id_val (numpy array): The in-distribution validation samples.
      feature_ood (numpy array): The out-of-distribution samples.
      train_labels (numpy array): The labels of the in-distribution training samples.
      path (str): The path to save and load the mean and precision matrix.

      Returns:
      tuple: The relative Mahalanobis scores for in-distribution validation and out-of-distribution samples.

      Steps:
      - Load class-wise mean and precision from disk if they exist, otherwise compute them from the ID training samples and save to disk.
      - Load global mean and precision from disk if they exist, otherwise compute them from all the ID training samples and save to disk.
      - Compute the relative Mahalanobis scores for ID validation samples and save to disk if they don't exist.
      - Compute the relative Mahalanobis scores for OOD samples and save to disk.

      """
    feature_id_val = feature_id_val/np.linalg.norm(feature_id_val,axis=-1,keepdims=True)
    feature_ood = feature_ood/np.linalg.norm(feature_ood,axis=-1,keepdims=True)
    # load class-wise mean and prec
    mean_path = os.path.join(path, 'mean_norm.npy')
    prec_path = os.path.join(path, 'prec_norm.npy')
    complete = True
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        complete = False
    if os.path.exists(prec_path):
        prec = np.load(prec_path)
    else:
        complete = False
    if not complete:
        print('not complete, computing classwise mean feature...')
        feature_id_train = feature_id_train/np.linalg.norm(feature_id_train,axis=-1,keepdims=True)
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = np.array(train_means)
        prec = (ec.precision_)
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    # print('go to gpu with class-wise...')
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()

    # load global mean and prec - stay on cpu and use numpy for better precision
    mean_path_global = os.path.join(path, 'mean-global_norm.npy')
    prec_path_global = os.path.join(path, 'prec-global_norm.npy')
    complete = True
    if os.path.exists(mean_path_global):
        mean_global = np.load(mean_path_global)
    else:
        complete = False
    if os.path.exists(prec_path_global):
        prec_global = np.load(prec_path_global)
    else:
        complete = False
    if not complete:
        print('not complete, computing global mean feature...')
        train_means_global = []
        train_feat_centered_global = []

        _m_global = feature_id_train.mean(axis=0)
        train_means_global.append(_m_global)
        train_feat_centered_global.extend(feature_id_train - _m_global)

        print('computing precision matrix...')
        ec_global = EmpiricalCovariance(assume_centered=True)
        ec_global.fit(np.array(train_feat_centered_global).astype(np.float64))

        mean_global = np.array(train_means_global)
        prec_global = (ec_global.precision_)
        np.save(mean_path_global, mean_global)
        np.save(prec_path_global, prec_global)

    print('Computing scores...')
    score_id_path = os.path.join(path, 'rel_maha_id_scores_norm.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        score_id_path_classwise = os.path.join(path, 'maha_id_scores_norm.npy')
        if os.path.exists(score_id_path_classwise):
            score_id_classwise = np.load(score_id_path_classwise)
        else:
            score_id_classwise = -np.array(
                [(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                 tqdm(torch.from_numpy(feature_id_val).cuda().double())])
            np.save(score_id_path_classwise, score_id_classwise)
        #
        score_id_global = -np.array(
            [((((f - mean_global) @ prec_global) * (f - mean_global)).sum(axis=-1)).item() for f in
             tqdm((feature_id_val))])  # tqdm(torch.from_numpy(feature_id_val).cuda().float())])

        score_id = score_id_classwise - score_id_global
        np.save(score_id_path, score_id)

    score_ood_classwise = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                                     tqdm(torch.from_numpy(feature_ood).cuda().double())])
    score_ood_global = -np.array(
        [((((f - mean_global) @ prec_global) * (f - mean_global)).sum(axis=-1)).item() for f in tqdm(feature_ood)])
    score_ood = score_ood_classwise - score_ood_global
    return score_id, score_ood

def evaluate_Relative_Mahalanobis(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    """
      This function computes the relative Mahalanobis scores for in-distribution and out-of-distribution samples.

      Parameters:
      feature_id_train (numpy array): The in-distribution training samples.
      feature_id_val (numpy array): The in-distribution validation samples.
      feature_ood (numpy array): The out-of-distribution samples.
      train_labels (numpy array): The labels of the in-distribution training samples.
      path (str): The path to save and load the mean and precision matrix.

      Returns:
      tuple: The relative Mahalanobis scores for in-distribution validation and out-of-distribution samples.

      Steps:
      - Load class-wise mean and precision from disk if they exist, otherwise compute them from the ID training samples and save to disk.
      - Load global mean and precision from disk if they exist, otherwise compute them from all the ID training samples and save to disk.
      - Compute the relative Mahalanobis scores for ID validation samples and save to disk if they don't exist.
      - Compute the relative Mahalanobis scores for OOD samples and save to disk.

      """
    # load class-wise mean and prec
    mean_path = os.path.join(path, 'mean.npy')
    prec_path = os.path.join(path, 'prec.npy')
    complete = True
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        complete = False
    if os.path.exists(prec_path):
        prec = np.load(prec_path)
    else:
        complete = False
    if not complete:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = np.array(train_means)
        prec = (ec.precision_)
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    # print('go to gpu with class-wise...')
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()

    # load global mean and prec - stay on cpu and use numpy for better precision
    mean_path_global = os.path.join(path, 'mean-global.npy')
    prec_path_global = os.path.join(path, 'prec-global.npy')
    complete = True
    if os.path.exists(mean_path_global):
        mean_global = np.load(mean_path_global)
    else:
        complete = False
    if os.path.exists(prec_path_global):
        prec_global = np.load(prec_path_global)
    else:
        complete = False
    if not complete:
        print('not complete, computing global mean feature...')
        train_means_global = []
        train_feat_centered_global = []

        _m_global = feature_id_train.mean(axis=0)
        train_means_global.append(_m_global)
        train_feat_centered_global.extend(feature_id_train - _m_global)

        print('computing precision matrix...')
        ec_global = EmpiricalCovariance(assume_centered=True)
        ec_global.fit(np.array(train_feat_centered_global).astype(np.float64))

        mean_global = np.array(train_means_global)
        prec_global = (ec_global.precision_)
        np.save(mean_path_global, mean_global)
        np.save(prec_path_global, prec_global)

    print('Computing scores...')
    score_id_path = os.path.join(path, 'rel_maha_id_scores.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        score_id_path_classwise = os.path.join(path, 'maha_id_scores.npy')
        if os.path.exists(score_id_path_classwise):
            score_id_classwise = np.load(score_id_path_classwise)
        else:
            score_id_classwise = -np.array(
                [(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                 tqdm(torch.from_numpy(feature_id_val).cuda().double())])
            np.save(score_id_path_classwise, score_id_classwise)
        #
        score_id_global = -np.array(
            [((((f - mean_global) @ prec_global) * (f - mean_global)).sum(axis=-1)).item() for f in
             tqdm((feature_id_val))])  # tqdm(torch.from_numpy(feature_id_val).cuda().float())])

        score_id = score_id_classwise - score_id_global
        np.save(score_id_path, score_id)

    score_ood_classwise = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                                     tqdm(torch.from_numpy(feature_ood).cuda().double())])
    score_ood_global = -np.array(
        [((((f - mean_global) @ prec_global) * (f - mean_global)).sum(axis=-1)).item() for f in tqdm(feature_ood)])
    score_ood = score_ood_classwise - score_ood_global
    return score_id, score_ood

def evaluate_KL_Matching(softmax_id_train, softmax_id_val, softmax_ood, path):
    """
    Evaluate KL Matching between softmax output of trained classifier and validation/out-of-distribution data.

    Inputs:
    softmax_id_train (ndarray): Softmax output of classifier on training data. Shape: (num_training_samples, num_classes)
    softmax_id_val (ndarray): Softmax output of classifier on validation data. Shape: (num_validation_samples, num_classes)
    softmax_ood (ndarray): Softmax output of classifier on out-of-distribution data. Shape: (num_ood_samples, num_classes)
    path (str): Path to directory where mean_softmax_train.npy and score_id_KL.npy should be stored/loaded from

    Outputs:
    score_id (ndarray): KL Matching score between softmax_id_val and mean_softmax_train. Shape: (num_validation_samples,)
    score_ood (ndarray): KL Matching score between softmax_ood and mean_softmax_train. Shape: (num_ood_samples,)
    """
    mean_softmax_train_path = os.path.join(path, 'mean_softmax_train.npy')
    score_id_KL_path = os.path.join(path, 'score_id_KL.npy')
    if os.path.exists(mean_softmax_train_path):
        mean_softmax_train = np.load(mean_softmax_train_path)
    else:
        print('not complete, computing classwise mean softmax...')
        pred_labels_train = np.argmax(softmax_id_train, axis=-1)
        mean_softmax_train = np.array(
            [softmax_id_train[pred_labels_train == i].mean(axis=0) for i in tqdm(range(1000))])
        np.save(mean_softmax_train_path, mean_softmax_train)
    if os.path.exists(score_id_KL_path):
        score_id = np.load(score_id_KL_path)
    else:
        print('not complete, Computing id score...')
        score_id = -pairwise_distances_argmin_min(softmax_id_val, (mean_softmax_train), metric=kl)[1]
        print('score_id is nan: ', np.isnan(score_id).any())
        np.save(score_id_KL_path, score_id)
    print('Computing OOD score...')
    score_ood = -pairwise_distances_argmin_min(softmax_ood, (mean_softmax_train), metric=kl)[1]
    return score_id, score_ood


def evaluate_Energy_React(feature_id_train, feature_id_val, feature_ood, w, b, path, clip_quantile=0.99):
    """Evaluate Energy React Score

       The function evaluates Energy React Score by computing score_id and score_ood.

       Parameters
       ----------
       feature_id_train: np.ndarray
           Input features of the training set.
       feature_id_val: np.ndarray
           Input features of the validation set.
       feature_ood: np.ndarray
           Input features of the out-of-distribution set.
       w: np.ndarray
           Weight matrix of classifiers last layer
       b: np.ndarray
           Bias vector of classifiers last layer
       path: str
           Path to store intermediate values.
       clip_quantile: float, optional, default 0.99
           Quantile used for clipping the input features.

       Returns
       -------
       score_id: np.ndarray
           Energy Reactivity Score for validation set.
       score_ood: np.ndarray
           Energy Reactivity Score for out-of-distribution set.
       """
    clip_react_path = os.path.join(path, 'clip_react.npy')
    if os.path.exists(clip_react_path):
        clip = np.load(clip_react_path)
    else:
        clip = np.quantile(feature_id_train, clip_quantile)
        np.save(clip_react_path, clip)
    print(f'clip quantile {clip_quantile}, clip {clip:.4f}')
    score_id_energy_react_path = os.path.join(path, 'score_id_energy_react.npy')
    if os.path.exists(score_id_energy_react_path):
        score_id = np.load(score_id_energy_react_path)
    else:
        print('not complete, Computing id score...')
        logit_id_val_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T + b
        score_id = logsumexp(logit_id_val_clip, axis=-1)
        np.save(score_id_energy_react_path, score_id)
    logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
    score_ood = logsumexp(logit_ood_clip, axis=-1)
    return score_id, score_ood


def evaluate_KNN(feature_id_train, feature_id_val, feature_ood, path):
    """
    Evaluate KNN classification for in-distribution (ID) and out-of-distribution (OOD) samples.

    This function computes KNN scores for ID and OOD samples. The KNN scores are computed as the distance to the K nearest neighbour of the ID samples in a preprocessed feature space.

    Args:
    feature_id_train_prepos (numpy.ndarray): Preprocessed features of ID training samples.
    feature_id_val (numpy.ndarray): Features of ID validation samples.
    feature_ood (numpy.ndarray): Features of OOD samples.
    path (str): File path to save intermediate computations.

    Returns:
    Tuple of numpy.ndarray:
    score_id (numpy.ndarray): KNN scores of ID validation samples.
    score_ood (numpy.ndarray): KNN scores of OOD samples.

    """
    normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

    scores_id_path_knn = os.path.join(path, 'scores_id_knn.npy')
    index_path = os.path.join(path, 'trained.index')

    # compute neighbours
    K = 1000
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        print('Index not stored, creating index...')
        feature_id_train_prepos = prepos_feat(feature_id_train)
        index = faiss.IndexFlatL2(feature_id_train_prepos.shape[1])
        index.add(feature_id_train_prepos)
        faiss.write_index(index, index_path)

    if os.path.exists(scores_id_path_knn):
        score_id = np.load(scores_id_path_knn)
    else:
        print('Computing id knn scores...')
        ftest = prepos_feat(feature_id_val).astype(np.float32)
        D, _ = index.search(ftest, K, )
        score_id = -D[:, -1]
        np.save(scores_id_path_knn, score_id)

    print('Computing ood knn scores...')
    food = prepos_feat(feature_ood)
    D, _ = index.search(food, K)
    score_ood = -D[:, -1]
    return score_id, score_ood

def evaluate_nnguide(feature_id_train, feature_id_val, feature_ood, logits_id_train, logits_id_val, logits_ood, path, alpha=0.01, K=1000):
    normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
    scores_id_path_knn = os.path.join(path, 'scores_id_nnguide.npy')
    index_path = os.path.join(path, 'trained_nnguide.index')

    # compute neighbours
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        print('Index not stored, creating index...')
        n_bank_samples = int(len(feature_id_train)*alpha)
        idxs=np.random.choice(range(0,len(feature_id_train)),n_bank_samples,replace=False)
        bank_feas = normalizer(feature_id_train[idxs,:])
        bank_confs = logsumexp(logits_id_train[idxs,:],axis=-1)
        bank_guide = bank_feas * bank_confs[:, None]
        index = faiss.IndexFlatIP(bank_guide.shape[-1])
        index.add(bank_guide)
        faiss.write_index(index, index_path)

    if False: #os.path.exists(scores_id_path_knn): 
        score_id = np.load(scores_id_path_knn)
    else:
        print('Computing id knn scores...')
        feas_val_norm = normalizer(feature_id_val)
        energy_val = logsumexp(logits_id_val,axis=-1)
        D, _ = index.search(feas_val_norm, K)
        conf_val = np.array(D.mean(axis=1))
        score_id = conf_val * energy_val
        # np.save(scores_id_path_knn,score_id)

    feas_ood_norm = normalizer(feature_ood)
    energy_ood = logsumexp(logits_ood,axis=-1)
    D, _ = index.search(feas_ood_norm, K)
    conf_ood = np.array(D.mean(axis=1))
    score_ood = conf_ood * energy_ood

    return score_id, score_ood    

def evaluate_she(feature_id_train, feature_id_val, feature_ood, logits_id_train, logits_id_val, logits_ood, labels_train, path, metric='inner_product'):
        
    activation_log_path = os.path.join(path, 'activation_log.pt')
    if os.path.exists(activation_log_path):
        activation_log = torch.load(activation_log_path)
    else:
        train_preds=torch.max(torch.from_numpy(logits_id_train),-1)[1]
        all_activation_log = torch.from_numpy(feature_id_train)
        activation_log = []
        for i in range(1000):
            mask = torch.logical_and(torch.from_numpy(labels_train) == i, train_preds == i)
            class_correct_activations = all_activation_log[mask]
            activation_log.append(
                class_correct_activations.mean(0, keepdim=True))

        activation_log = torch.cat(activation_log)
        torch.save(activation_log, activation_log_path)

    def distance(penultimate, target, metric='inner_product'):
        if metric == 'inner_product':
            return torch.sum(torch.mul(penultimate, target), dim=1)
        elif metric == 'euclidean':
            return -torch.sqrt(torch.sum((penultimate - target)**2, dim=1))
        elif metric == 'cosine':
            return torch.cosine_similarity(penultimate, target, dim=1)
        else:
            raise ValueError('Unknown metric: {}'.format(metric))
    pred_val=torch.max(torch.from_numpy(logits_id_val),-1)[1]
    score_id = distance(feature_id_val, activation_log[pred_val], metric).numpy()

    pred_ood=torch.max(torch.from_numpy(logits_ood),-1)[1]
    score_ood = distance(feature_ood, activation_log[pred_ood], metric).numpy()

    return score_id, score_ood

def evaluate_scale(feature_id_val, feature_ood,  w, b, percentile=85):
    def scale(x, percentile=85):
        input_ = x.clone()
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1, 2, 3])

        # apply sharpening
        scale = s1 / s2

        return input_ * torch.exp(scale[:, None, None, None])

    def eval_scale(feature,w,b,percentile=65):
        feature=torch.from_numpy(feature.copy())
        feature = scale(feature.view(feature.size(0), -1, 1, 1), percentile)
        feature = feature.view(feature.size(0), -1)
        logits_cls = (feature@w.T) + b
        return torch.logsumexp(logits_cls.data.cpu(), dim=1).numpy()
    
    s_id = eval_scale(feature_id_val,w,b,percentile=percentile)
    s_ood = eval_scale(feature_ood,w,b,percentile=percentile)

    return s_id, s_ood

def evaluate_ash_s(feature_id_val, feature_ood,  w, b, percentile=90):
    def ash_s(x, percentile=percentile):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1, 2, 3])

        # apply sharpening
        scale = s1 / s2
        x = x * torch.exp(scale[:, None, None, None])

        return x

    def ash(f,w,b):
        with torch.no_grad():
            to_forward = torch.from_numpy(f.copy())
            feature = ash_s(to_forward.view(to_forward.size(0), -1, 1, 1))
            feature = feature.view(feature.size(0), -1)
            output=feature@ w.T + b
            print(output.shape)
            energyconf = torch.logsumexp(output.data.cpu(), dim=-1)
            return energyconf.numpy()
    s_id = ash(feature_id_val,w,b)
    s_ood = ash(feature_ood,w,b)
    return s_id, s_ood

def evaluate_ash_b(feature_id_val, feature_ood,  w, b, percentile=65):
    def ash_b(x, percentile=percentile):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])

        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        fill = s1 / k
        fill = fill.unsqueeze(dim=1).expand(v.shape)
        t.zero_().scatter_(dim=1, index=i, src=fill)
        return x

    def ash(f,w,b):
        with torch.no_grad():
            to_forward = torch.from_numpy(f.copy())
            feature = ash_b(to_forward.view(to_forward.size(0), -1, 1, 1))
            feature = feature.view(feature.size(0), -1)
            output=feature@ w.T + b
            print(output.shape)
            energyconf = torch.logsumexp(output.data.cpu(), dim=-1)
            return energyconf.numpy()
    s_id = ash(feature_id_val,w,b)
    s_ood = ash(feature_ood,w,b)
    return s_id, s_ood

def evaluate_cosine(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    '''
    Like Cosine for CLIP, but with class-wise mean-features as encoded text:

    This function loads the mean of the in-distribution features, or computes and saves it if not found,
    and computes the cosine similarity scores between the in-distribution and out-of-distribution inputs and the mean.

    Parameters:
    feature_id_train (np.array): In-distribution training features.
    feature_id_val (np.array): In-distribution validation features.
    feature_ood (np.array): Out-of-distribution features.
    train_labels (np.array): Labels for in-distribution training data.
    path (str): Path to save and load mean.

    Returns:
    tuple: In-distribution and out-of-distribution cosine similarity scores.
    '''
    # load mean
    mean_path = os.path.join(path, 'mean.npy')
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)
        mean = np.array(train_means)
        np.save(mean_path, mean)
    means_n = np.array([m / np.linalg.norm(m) for m in mean])
    features_id_normalized = np.array([m / np.linalg.norm(m) for m in feature_id_val])
    score_id = (features_id_normalized @ means_n.T).max(axis=-1)
    features_ood_normalized = np.array([m / np.linalg.norm(m) for m in feature_ood])
    score_ood = (features_ood_normalized @ means_n.T).max(axis=-1)
    return score_id, score_ood


def evaluate_rcos(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    '''
    Like MCM, but with class-wise mean-features as encoded text:
    This function loads the mean of the in-distribution data, or computes and saves it if not found,
    and computes the softmax of the cosine similarity scores between the in-distribution and out-of-distribution
    inputs and the mean.

    Parameters:
    feature_id_train (np.array): In-distribution training features.
    feature_id_val (np.array): In-distribution validation features.
    feature_ood (np.array): Out-of-distribution features.
    train_labels (np.array): Labels for in-distribution training data.
    path (str): Path to save and load mean.

    Returns:
    tuple: In-distribution and out-of-distribution re-scaled cosine similarity scores.
    '''
    T = 1.
    mean_path = os.path.join(path, 'mean.npy')
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)
        mean = np.array(train_means)
        np.save(mean_path, mean)

    # use train means as encoded text pairs
    text_encoded = torch.from_numpy(mean).float()
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    scores_id_path_clip = os.path.join(path, 'mcm_scores_id.npy')
    if os.path.exists(scores_id_path_clip):
        score_id = np.load(scores_id_path_clip)
    else:
        print('Computing ID scores...')

        features_id = torch.from_numpy(feature_id_val).float()
        features_id /= features_id.norm(dim=-1, keepdim=True)

        out_id = features_id @ text_encoded.T
        smax_id = F.softmax(out_id / T, dim=1).data.cpu().numpy()
        score_id = np.max(smax_id, axis=1)

        np.save(scores_id_path_clip, score_id)
    print('Computing OOD scores...')

    features_ood = torch.from_numpy(feature_ood).float()
    features_ood /= features_ood.norm(dim=-1, keepdim=True)

    out_ood = features_ood @ text_encoded.T
    smax_ood = F.softmax(out_ood / T, dim=1).data.cpu().numpy()
    score_ood = np.max(smax_ood, axis=1)
    return score_id, score_ood


def evaluate_cosine_clip(feature_id_val, feature_ood, clip_labels, labels_encoded_clip, path):
    """
       Evaluates cosine similarity scores for in-distribution and out-of-distribution samples and returns the scores,
       along with the in-distribution accuracy for CLIP features.

       Parameters:
       feature_id_val (np.ndarray): In-distribution validation feature tensor.
       feature_ood (np.ndarray): Out-of-distribution feature tensor.
       clip_labels (np.ndarray): Ground truth labels for the in-distribution validation samples.
       labels_encoded_clip (np.ndarray): Encoded ground truth labels for the in-distribution samples.
       path (str): Path to the directory to save the scores.

       Returns:
       tuple:
           score_id (np.ndarray): Cosine similarity scores for the in-distribution samples.
           score_ood (np.ndarray): Cosine similarity scores for the out-of-distribution samples.
           val_acc (float): In-distribution accuracy.
       """
    text_encoded = np.array([m / np.linalg.norm(m) for m in
                             labels_encoded_clip])  # labels_encoded_clip / labels_encoded_clip.norm(dim = -1, keepdim = True)
    scores_id_path_clip = os.path.join(path, 'cosine-clip_scores_id.npy')
    acc_path = os.path.join(path, 'accuracy.npy')
    if os.path.exists(scores_id_path_clip):
        score_id = np.load(scores_id_path_clip)
        val_acc = np.load(acc_path)
    else:
        print('Computing ID scores...')
        x_val_id_encoded = np.array([m / np.linalg.norm(m) for m in feature_id_val])
        # feature_id_val / feature_id_val.norm(dim = -1, keepdim = True)
        similarity_id = (x_val_id_encoded @ text_encoded.T)
        preds = np.argmax(similarity_id, axis=-1)
        val_acc = np.equal(preds, clip_labels).mean()
        np.save(acc_path, val_acc)
        score_id = np.max(similarity_id, axis=-1)
        np.save(scores_id_path_clip, score_id)
    print('Computing OOD scores...')
    x_ood_encoded = np.array([m / np.linalg.norm(m) for m in feature_ood])
    # feature_ood / feature_ood.norm(dim = -1, keepdim = True)

    similarity_ood = (x_ood_encoded @ text_encoded.T)
    score_ood = np.max(similarity_ood, axis=-1)
    return score_id, score_ood, val_acc


def evaluate_mcm_clip(feature_id_val, feature_ood, clip_labels, labels_encoded_clip, path):
    """
    This function computes the MCM score for a given set of ID data (feature_id_val) and OOD data (feature_ood)
    by first normalizing the features and then computing the dot product between the features and the encoded
    text representations (labels_encoded_clip). The resulting scores are then passed through a softmax function
    to obtain the final MCM scores. The ID scores are saved to disk (mcm-clip_scores_id.npy) along with the
    accuracy (accuracy.npy) if they have not already been computed.

    Inputs:

        feature_id_val: numpy array, shape (num_ID_data, num_features)
        The in-distribution data to evaluate the MCM scores for.
        feature_ood: numpy array, shape (num_OOD_data, num_features)
        The out-of-distribution data to evaluate the MCM scores for.
        clip_labels: numpy array, shape (num_ID_data,)
        The labels for the in-distribution data.
        labels_encoded_clip: numpy array, shape (num_texts, num_features)
        The encoded text representations.
        path: str
        The path to save the ID scores and accuracy if they have not already been computed.

    Returns:

        score_id: numpy array, shape (num_ID_data,)
        The MCM scores for the in-distribution data.
        score_ood: numpy array, shape (num_OOD_data,)
        The MCM scores for the out-of-distribution data.
        val_acc: float
        The accuracy of the prediction on the in-distribution validation data.
        """
    T = 1.
    text_encoded = torch.from_numpy(labels_encoded_clip).float()
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    scores_id_path_clip = os.path.join(path, 'mcm-clip_scores_id.npy')
    acc_path = os.path.join(path, 'accuracy.npy')
    if os.path.exists(scores_id_path_clip):
        score_id = np.load(scores_id_path_clip)
        val_acc = np.load(acc_path)
    else:
        print('Computing ID scores...')

        features_id = torch.from_numpy(feature_id_val).float()
        features_id /= features_id.norm(dim=-1, keepdim=True)

        out_id = features_id @ text_encoded.T
        smax_id = F.softmax(out_id / T, dim=1).data.cpu().numpy()
        score_id = np.max(smax_id, axis=1)

        preds = np.argmax(out_id.data.cpu().numpy(), axis=-1)
        val_acc = np.equal(preds, clip_labels).mean()
        np.save(acc_path, val_acc)
        np.save(scores_id_path_clip, score_id)
    print('Computing OOD scores...')

    features_ood = torch.from_numpy(feature_ood).float()
    features_ood /= features_ood.norm(dim=-1, keepdim=True)

    out_ood = features_ood @ text_encoded.T
    smax_ood = F.softmax(out_ood / T, dim=1).data.cpu().numpy()
    score_ood = np.max(smax_ood, axis=1)

    return score_id, score_ood, val_acc


# ──────────────────────────────────────────────────────────────────────────────
# MM++: Multilayer Mahalanobis++ Distance (NeurIPS 2025)
# ──────────────────────────────────────────────────────────────────────────────

def _fit_logreg_weights(X, y, w_init, lr=0.1, n_iter=200):
    """
    Optimize logistic regression weights via gradient descent with custom init.

    Binary cross-entropy:  L(w) = -1/N Σ [y_i log p_i + (1-y_i) log(1-p_i)]
    where p_i = sigmoid(X_i @ w).

    Args:
        X:      [N, L] score matrix
        y:      [N] binary labels  (0 = ID, 1 = OOD)
        w_init: [L] initial weights (inverse-std from Eq. 12)
        lr:     learning rate
        n_iter: gradient-descent steps

    Returns:
        w: [L] optimized weights
    """
    w = w_init.copy().astype(np.float64)
    y = y.astype(np.float64)
    N = len(y)
    for _ in range(n_iter):
        logits = X @ w
        p = 1.0 / (1.0 + np.exp(-logits.clip(-500, 500)))  # sigmoid, numerically stable
        grad = (X.T @ (p - y)) / N
        w -= lr * grad
    return w


def evaluate_MM_plus_plus(train_inter_path, layer_feats_val, layer_feats_ood,
                          train_labels, path, n_classes=1000,
                          top_k=None, zscore=False):
    """
    MM++: Multilayer Mahalanobis++ Distance for OOD detection.

    Algorithm (Section 3.2–3.3 of the paper):
      For each intermediate layer l:
        1. L2-normalize features onto unit hypersphere  (Eq. 4)
        2. Compute class-conditional means              (Eq. 5)
        3. Estimate tied covariance with Ledoit-Wolf    (Eq. 13)
        4. Layer score S_l(x) = -min_c M++_{c,l}(x)   (Eq. 7)
      Aggregate:
        5. Initialize w_l = eps / std(S_l on ID val)   (Eq. 12)
        6. Learn w_l via logistic regression on a
           small OOD val split + all ID val             (Eq. 10-11)
        7. S_total(x) = Σ_l w_l S_l(x)                (Eq. 8)

    Training features are loaded one layer at a time from disk to avoid
    holding all L × N_train × D in memory simultaneously.

    Args:
        train_inter_path: directory of per-layer training features
                          (output of extract_intermediate_features on train set)
        layer_feats_val:  list of L arrays [N_val, D_l]
        layer_feats_ood:  list of L arrays [N_ood, D_l]
        train_labels:     [N_train] integer class labels
        path:             cache directory for computed statistics / ID scores
        n_classes:        number of ID classes (1000 for ImageNet-1k)
        top_k:            if set, only aggregate the top-K layers by ER_B
        zscore:           if True, z-normalize each layer's scores using ID val
                          statistics before aggregation (fixes scale mismatch)

    Returns:
        score_id:  [N_val] MM++ scores for ID validation data
        score_ood: [N_ood] MM++ scores for OOD data
    """
    from sklearn.covariance import LedoitWolf
    import json

    names_path = os.path.join(train_inter_path, 'layer_names.json')
    if not os.path.exists(names_path):
        raise FileNotFoundError(
            f'layer_names.json not found in {train_inter_path}. '
            f'Run extract_intermediate_features() first.'
        )
    with open(names_path) as f:
        layer_names = json.load(f)

    L = len(layer_names)
    assert len(layer_feats_val) == L, \
        f'Expected {L} val-layer arrays, got {len(layer_feats_val)}'
    assert len(layer_feats_ood) == L, \
        f'Expected {L} OOD-layer arrays, got {len(layer_feats_ood)}'

    layer_scores_val   = []
    layer_scores_ood   = []
    effective_ranks    = []   # within-class ER  (kept for diagnostics)
    effective_ranks_b  = []   # between-class ER (used for aggregation)

    for l, layer_name in enumerate(layer_names):
        tag       = f'mm_pp_{layer_name}'
        mean_path = os.path.join(path, f'{tag}_mean.npy')
        prec_path = os.path.join(path, f'{tag}_prec.npy')
        er_path   = os.path.join(path, f'{tag}_er.npy')

        # ── Fit class means + Ledoit-Wolf precision + effective rank (cached) ─
        if os.path.exists(mean_path) and os.path.exists(prec_path) and os.path.exists(er_path):
            mean_l = np.load(mean_path)
            prec_l = np.load(prec_path)
            er_l   = float(np.load(er_path))
        else:
            print(f'[MM++] Layer {l+1}/{L} ({layer_name}): loading training features...')
            parts = sorted(
                [fn for fn in os.listdir(train_inter_path)
                 if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
                key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
            )
            if not parts:
                raise FileNotFoundError(
                    f'No training features found for layer {layer_name} in {train_inter_path}'
                )
            f_train = np.concatenate(
                [np.load(os.path.join(train_inter_path, p)) for p in parts], axis=0
            ).astype(np.float64)

            # L2-normalize (Eq. 4)
            norms = np.linalg.norm(f_train, axis=-1, keepdims=True).clip(min=1e-10)
            f_train /= norms

            print(f'[MM++] Layer {l+1}/{L} ({layer_name}): computing class means...')
            train_means = []
            centered    = []
            for c in tqdm(range(n_classes), desc=f'  Layer {layer_name}', leave=False):
                fs = f_train[train_labels == c]
                m  = fs.mean(axis=0) if len(fs) > 0 else np.zeros(f_train.shape[1])
                train_means.append(m)
                if len(fs) > 0:
                    centered.extend(fs - m)

            centered_arr = np.array(centered, dtype=np.float64)  # [N, D]

            print(f'[MM++] Layer {l+1}/{L} ({layer_name}): Ledoit-Wolf shrinkage...')
            lw = LedoitWolf(assume_centered=True)
            lw.fit(centered_arr)

            # ── Effective rank (Eqs. 7–8) via eigenspectrum of tied covariance ─
            # Dual-SVD: Gram matrix [N×N] if N<D, else direct D×D covariance.
            # Tied covariance = class-mean-centered, L2-normalized features.
            N_total, D = centered_arr.shape
            if N_total < D:
                G = centered_arr @ centered_arr.T / N_total       # [N, N]
                eigvals = np.linalg.eigvalsh(G)
            else:
                cov = centered_arr.T @ centered_arr / N_total     # [D, D]
                eigvals = np.linalg.eigvalsh(cov)
            eigvals = eigvals[eigvals > 0]
            eigvals /= eigvals.sum()
            er_l = float(np.exp(-np.dot(eigvals, np.log(eigvals))))

            mean_l = np.array(train_means)
            prec_l = lw.precision_
            np.save(mean_path, mean_l)
            np.save(prec_path, prec_l)
            np.save(er_path, er_l)
            del f_train, centered, centered_arr  # free memory

        effective_ranks.append(er_l)

        # ── Between-class ER: ER of class-means covariance (cached) ──────────
        # Measures how well-spread class means are on the hypersphere.
        # Increases with depth for both ViT and CNN → fixes the inverted-ER problem.
        er_b_path = os.path.join(path, f'{tag}_er_b.npy')
        if os.path.exists(er_b_path):
            er_b_l = float(np.load(er_b_path))
        else:
            mu_global = mean_l.mean(axis=0)                        # [D]
            M = (mean_l - mu_global).astype(np.float64)            # [C, D]
            C_cls = M.shape[0]
            sigma_b = M.T @ M / C_cls                              # [D, D]
            ev = np.linalg.eigvalsh(sigma_b)
            ev = ev[ev > 0]
            ev /= ev.sum()
            er_b_l = float(np.exp(-np.dot(ev, np.log(ev))))
            np.save(er_b_path, er_b_l)
        effective_ranks_b.append(er_b_l)

        # ── Move to GPU ───────────────────────────────────────────────────────
        mean_t = torch.from_numpy(mean_l).cuda().double()
        prec_t = torch.from_numpy(prec_l).cuda().double()

        def _maha_scores(feats_np):
            """Returns [N] array of -min_c M++_{c,l}(x) for each sample."""
            feats_np = feats_np.astype(np.float64)
            norms = np.linalg.norm(feats_np, axis=-1, keepdims=True).clip(min=1e-10)
            feats_np = feats_np / norms
            scores = []
            for f in torch.from_numpy(feats_np).cuda().double():
                diff = f - mean_t                         # [C, D]
                d    = ((diff @ prec_t) * diff).sum(-1)   # [C]
                scores.append(-d.min().cpu().item())
            return np.array(scores)

        # ── ID val scores (cached per layer) ─────────────────────────────────
        id_score_path = os.path.join(path, f'{tag}_id_scores.npy')
        if os.path.exists(id_score_path):
            s_val_l = np.load(id_score_path)
        else:
            print(f'[MM++] Layer {l+1}/{L} ({layer_name}): computing ID scores...')
            s_val_l = _maha_scores(layer_feats_val[l])
            np.save(id_score_path, s_val_l)

        # ── OOD scores ────────────────────────────────────────────────────────
        print(f'[MM++] Layer {l+1}/{L} ({layer_name}): computing OOD scores...')
        s_ood_l = _maha_scores(layer_feats_ood[l])

        layer_scores_val.append(s_val_l)
        layer_scores_ood.append(s_ood_l)

    # ── Boltzmann rank-weighted sparsification (Eqs. 7–11, ER_B variant) ─────
    # Use between-class ER (ER_B) instead of total/within ER.
    # ER_B = ER of class-means covariance; increases with depth for both ViT and CNN
    # because class means become more spread as the network builds semantic structure.
    # This fixes the inverted-ER problem in isotropic ViTs, where total ER increases
    # with depth (early blocks uninformative, not collapsed), while remaining valid
    # for CNNs (ER_B also increases with depth as classes separate).
    er   = np.array(effective_ranks)    # [L]  within-class ER  (diagnostic only)
    er_b = np.array(effective_ranks_b)  # [L]  between-class ER (used for weighting)

    S_val = np.stack(layer_scores_val)   # [L, N_val]
    S_ood = np.stack(layer_scores_ood)   # [L, N_ood]

    # ── Optional top-K layer selection ───────────────────────────────────────
    # Discard early layers whose ER_B is too low to carry class-discriminative signal.
    if top_k is not None and top_k < len(er_b):
        keep  = np.argsort(er_b)[-top_k:]          # indices of top-K by ER_B
        er_b  = er_b[keep]
        er    = er[keep]
        S_val = S_val[keep]
        S_ood = S_ood[keep]
        print(f'[MM++] top_k={top_k}: keeping layers {keep.tolist()}')

    # ── Adaptive temperature: largest adjacent log-gain in ER_B ──────────────
    tau = float(np.max(np.log(er_b[1:] / er_b[:-1].clip(min=1e-8)))) if len(er_b) > 1 else 1.0

    # Boltzmann with positive sign: higher ER_B → more weight
    log_w = +tau * er_b
    log_w -= log_w.max()   # numerical stability
    w = np.exp(log_w)
    w /= w.sum()

    print(f'[MM++] Within-class ER:        {er.round(2)}')
    print(f'[MM++] Between-class ER (ER_B):{er_b.round(2)}')
    print(f'[MM++] Adaptive temperature:   τ = {tau:.4f}')
    print(f'[MM++] Aggregation weights:    {w.round(4)}')

    # ── Optional z-score normalization per layer (fixes scale mismatch) ──────
    # Standardize using ID val statistics; apply same shift/scale to OOD.
    if zscore:
        mean_l = S_val.mean(axis=1, keepdims=True)          # [L, 1]
        std_l  = S_val.std(axis=1, keepdims=True).clip(min=1e-8)
        S_val  = (S_val - mean_l) / std_l
        S_ood  = (S_ood - mean_l) / std_l

    # S_total(x) = Σ_l w_l · S_l(x)
    score_id  = w @ S_val   # [N_val]
    score_ood = w @ S_ood   # [N_ood]

    return score_id, score_ood


def evaluate_MM_plus_plus_feat(train_inter_path, layer_feats_val, layer_feats_ood,
                               train_labels, path, n_classes=1000,
                               boltzmann_temp=None, top_k=None):
    """
    MM++ Feature Mixing: L2-normalized feature fusion + single Mahalanobis++.

    Combines M++'s L2-normalization (Eq. 4 of MM++ paper) with X-Mahalanobis-style
    feature mixing.  Fusion weights are based on between-class ER (ER_B).

    For homogeneous-width backbones (e.g. ViTs), this function performs additive
    feature mixing within a selected dimension group.

    For heterogeneous-width backbones (e.g. ResNet stages with 256/512/1024/2048),
    additive mixing is not well-defined. In that case, this function falls back to
    concatenation: L2-normalize each selected stage separately, concatenate them,
    and fit a single Mahalanobis model on the joint feature vector.

    Algorithm:
      1. Load cached per-layer class means + ER_B (from evaluate_MM_plus_plus run).
      2. Group layers by feature dimension D.
      3. Rank layers by ER_B only within each dimension group.
      4. Select the fusion group as the dimension group whose best layer has the
         highest ER_B (ties broken in favour of the deeper layer).
      5. Compute fusion weights only inside the selected group:
           - boltzmann_temp=None: α_l = ER_B_l / Σ ER_B  (linear, selected group)
           - boltzmann_temp=τ:    α_l = softmax(τ * ER_B_l)  (Boltzmann, selected group)
           - top_k=k:             keep the top-k ER_B layers inside the selected
                                  group before normalising the weights
      6. For each active layer l: load features, L2-normalize, accumulate
             Φ(x) += α_l * h_l(x)   (one layer at a time, O(N×D) memory)
      7. Compute fused class means:  μ_c = Σ_l α_l * μ_{c,l}
      8. Fit Ledoit-Wolf on within-class residuals of fused features.
      9. Score: S(x) = −min_c (Φ(x) − μ_c)ᵀ Σ⁻¹ (Φ(x) − μ_c)

    Pre-requisite: evaluate_MM_plus_plus() must have been run first so that
    per-layer class-mean (.._mean.npy) and ER_B (.._er_b.npy) caches exist.

    Args:
        train_inter_path: directory of per-layer training feature shards
        layer_feats_val:  list of L arrays [N_val, D_l]
        layer_feats_ood:  list of L arrays [N_ood, D_l]
        train_labels:     [N_train] integer class labels
        path:             cache directory (same as used by evaluate_MM_plus_plus)
        n_classes:        number of ID classes

    Returns:
        score_id:  [N_val]  higher = more ID-like
        score_ood: [N_ood]  higher = more ID-like
    """
    from sklearn.covariance import LedoitWolf
    import json

    names_path = os.path.join(train_inter_path, 'layer_names.json')
    if not os.path.exists(names_path):
        raise FileNotFoundError(
            f'layer_names.json not found in {train_inter_path}. '
            f'Run extract_intermediate_features() first.'
        )
    with open(names_path) as f:
        layer_names = json.load(f)
    L = len(layer_names)

    # ── Pass 1: load cached ER_B and class means ──────────────────────────
    # All of these are produced (and cached) by evaluate_MM_plus_plus().
    er_b_vals  = []
    mean_list  = []   # [L]  each entry: [C, D]
    for l, layer_name in enumerate(layer_names):
        tag       = f'mm_pp_{layer_name}'
        er_b_path = os.path.join(path, f'{tag}_er_b.npy')
        mean_path = os.path.join(path, f'{tag}_mean.npy')
        if not os.path.exists(er_b_path) or not os.path.exists(mean_path):
            raise FileNotFoundError(
                f'Cached stats for layer {layer_name} not found in {path}. '
                f'Run MM_plus_plus first to populate the cache.'
            )
        er_b_vals.append(float(np.load(er_b_path)))
        mean_list.append(np.load(mean_path).astype(np.float64))  # [C, D]

    er_b_full = np.array(er_b_vals, dtype=np.float64)   # [L]  all layers
    dims_full = np.array([mean.shape[1] for mean in mean_list], dtype=int)

    # Heterogeneous-width backbones cannot be additively fused without an
    # explicit learned alignment. Use concatenation instead, defaulting to
    # the top-2 ER_B layers so early weak stages do not dominate the joint fit.
    if len(np.unique(dims_full)) > 1:
        concat_top_k = top_k if top_k is not None else 2
        if boltzmann_temp is not None:
            print('[MM++Feat] Heterogeneous feature dims detected; '
                  f'falling back to concatenation with top_k={concat_top_k}. '
                  'boltzmann_temp is ignored.')
        else:
            print('[MM++Feat] Heterogeneous feature dims detected; '
                  f'falling back to concatenation with top_k={concat_top_k}.')
        return evaluate_MM_plus_plus_concat(
            train_inter_path=train_inter_path,
            layer_feats_val=layer_feats_val,
            layer_feats_ood=layer_feats_ood,
            train_labels=train_labels,
            path=path,
            n_classes=n_classes,
            top_k=concat_top_k,
        )

    # ── Group layers by feature dimension and rank only within group ──────
    dim_groups = {}
    for idx, dim in enumerate(dims_full):
        dim_groups.setdefault(int(dim), []).append(idx)

    group_infos = []
    for dim, idxs in sorted(dim_groups.items(), key=lambda item: (item[1][-1], item[0])):
        idxs_arr = np.array(idxs, dtype=int)
        order = np.lexsort((idxs_arr, er_b_full[idxs_arr]))[::-1]
        ranked_idx = idxs_arr[order]
        ranked_erb = er_b_full[ranked_idx]
        group_infos.append({
            'dim': int(dim),
            'indices': idxs_arr,
            'ranked_idx': ranked_idx,
            'ranked_erb': ranked_erb,
            'best_erb': float(ranked_erb[0]),
            'best_depth': int(ranked_idx[0]),
        })
        print(
            f"[MM++Feat] Dim group {int(dim)}: "
            f"layers={[layer_names[i] for i in idxs_arr]} | "
            f"ER_B={np.round(er_b_full[idxs_arr], 2).tolist()} | "
            f"ranked={[layer_names[i] for i in ranked_idx]}"
        )

    # Choose the dimension group whose top ER_B layer is strongest.
    # This generalises the old "use final-layer dimension" rule without
    # comparing incompatible dimensions during fusion.
    selected_group = max(group_infos, key=lambda g: (g['best_erb'], g['best_depth']))
    ranked_idx = selected_group['ranked_idx']
    if top_k is not None and top_k < len(ranked_idx):
        active_idx = np.sort(ranked_idx[:top_k])
    else:
        active_idx = np.sort(ranked_idx)

    er_b_active = er_b_full[active_idx]                # [k]
    selected_dim = selected_group['dim']

    print(
        f"[MM++Feat] Selected dim group {selected_dim} "
        f"(best ER_B={selected_group['best_erb']:.3f} at "
        f"{layer_names[selected_group['best_depth']]})"
    )

    # ── Compute fusion weights ────────────────────────────────────────────
    if boltzmann_temp is not None:
        log_w  = boltzmann_temp * er_b_active
        log_w -= log_w.max()
        alpha  = np.exp(log_w)
        alpha /= alpha.sum()
    else:
        erb_sum = er_b_active.sum()
        if erb_sum <= 0:
            alpha = np.full(len(er_b_active), 1.0 / len(er_b_active), dtype=np.float64)
        else:
            alpha = er_b_active / erb_sum              # linear ER_B normalisation

    print(f'[MM++Feat] ER_B (active layers): {er_b_active.round(2)}')
    print(f'[MM++Feat] Active layer names:   {[layer_names[i] for i in active_idx]}')
    print(f'[MM++Feat] Fusion weights (α):   {alpha.round(4)}')

    # ── Cache tag: unique per (selection rule, dim group, boltzmann_temp, top_k)
    # Bump the prefix so old fused caches from the previous cross-dimension
    # selection rule are never reused by accident.
    tag_suffix  = f'_tau{boltzmann_temp:.4f}' if boltzmann_temp is not None else '_linear'
    tag_suffix += f'_top{top_k}' if top_k is not None else '_all'
    fused_tag       = f'mm_pp_feat_dimgroup_v2_dim{selected_dim}{tag_suffix}'
    fused_mean_path = os.path.join(path, f'{fused_tag}_means.npy')
    fused_prec_path = os.path.join(path, f'{fused_tag}_prec.npy')
    id_score_path   = os.path.join(path, f'{fused_tag}_id_scores.npy')

    D = mean_list[active_idx[0]].shape[1]  # feature dimension of active layers

    if os.path.exists(fused_mean_path) and os.path.exists(fused_prec_path):
        print('[MM++Feat] Loading cached fused class means + precision...')
        fused_means = np.load(fused_mean_path)    # [C, D]
        prec_fused  = np.load(fused_prec_path)    # [D, D]
    else:
        # ── Pass 2: accumulate fused training features ────────────────────
        N_train     = len(train_labels)
        fused_train = np.zeros((N_train, D), dtype=np.float64)
        fused_means = np.zeros((n_classes, D), dtype=np.float64)

        for a_pos, l in enumerate(active_idx):
            layer_name = layer_names[l]
            print(f'[MM++Feat] Layer {l+1}/{L} ({layer_name}): fusing train features...')
            parts = sorted(
                [fn for fn in os.listdir(train_inter_path)
                 if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
                key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
            )
            if not parts:
                raise FileNotFoundError(
                    f'No training features for layer {layer_name} in {train_inter_path}'
                )
            f_l = np.concatenate(
                [np.load(os.path.join(train_inter_path, p)) for p in parts], axis=0
            ).astype(np.float64)

            # L2-normalize (Eq. 4 of MM++ paper)
            norms = np.linalg.norm(f_l, axis=-1, keepdims=True).clip(min=1e-10)
            f_l  /= norms

            fused_train += alpha[a_pos] * f_l
            fused_means += alpha[a_pos] * mean_list[l]
            del f_l

        # Center fused training features within each class
        print('[MM++Feat] Centering fused training features...')
        fused_centered = fused_train - fused_means[train_labels]
        del fused_train

        # Fit Ledoit-Wolf shrinkage on fused within-class residuals
        print('[MM++Feat] Fitting Ledoit-Wolf on fused features...')
        lw = LedoitWolf(assume_centered=True)
        lw.fit(fused_centered)
        prec_fused = lw.precision_
        del fused_centered

        np.save(fused_mean_path, fused_means)
        np.save(fused_prec_path, prec_fused)

    # ── Helper: fuse + score a set of per-layer feature arrays ───────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fuse_and_score(layer_feats_list):
        N     = layer_feats_list[active_idx[0]].shape[0]
        fused = np.zeros((N, D), dtype=np.float64)
        for a_pos, l in enumerate(active_idx):
            feats = layer_feats_list[l].astype(np.float64)
            norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-10)
            feats /= norms
            fused += alpha[a_pos] * feats

        mean_t = torch.from_numpy(fused_means).to(device=device, dtype=torch.float64)   # [C, D]
        prec_t = torch.from_numpy(prec_fused).to(device=device, dtype=torch.float64)    # [D, D]
        mean_prec_t = mean_t @ prec_t                                                    # [C, D]
        mean_quad_t = (mean_prec_t * mean_t).sum(dim=1)                                  # [C]

        batch_size = 1024 if device.type == 'cuda' else 256
        scores = np.empty(N, dtype=np.float64)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_t = torch.from_numpy(fused[start:end]).to(device=device, dtype=torch.float64)
            x_prec_t = x_t @ prec_t                                                      # [B, D]
            x_quad_t = (x_prec_t * x_t).sum(dim=1, keepdim=True)                         # [B, 1]
            cross_t = x_t @ mean_prec_t.T                                                # [B, C]
            dist_t = x_quad_t - (2.0 * cross_t) + mean_quad_t.unsqueeze(0)               # [B, C]
            scores[start:end] = (-dist_t.min(dim=1).values).cpu().numpy()
        return scores

    # ── ID val scores (cached) ────────────────────────────────────────────
    if os.path.exists(id_score_path):
        score_id = np.load(id_score_path)
    else:
        print('[MM++Feat] Computing ID val scores...')
        score_id = _fuse_and_score(layer_feats_val)
        np.save(id_score_path, score_id)

    # ── OOD scores ────────────────────────────────────────────────────────
    print('[MM++Feat] Computing OOD scores...')
    score_ood = _fuse_and_score(layer_feats_ood)

    return score_id, score_ood


def evaluate_MM_plus_plus_concat(train_inter_path, layer_feats_val, layer_feats_ood,
                                 train_labels, path, n_classes=1000, top_k=None):
    """
    MM++ Concatenation: stack L2-normalized features from selected layers, then
    compute a single Mahalanobis++ score in the joint feature space.

    Unlike feature mixing (Σ α_l h_l), concatenation [h_{l1}; h_{l2}; ...]
    preserves all per-layer information and learns the full joint covariance
    across layers.  An OOD sample that looks normal in each individual layer
    can be detected via anomalous cross-layer correlations.

    Layers are selected by highest between-class ER (ER_B), which requires
    evaluate_MM_plus_plus() to have been run first (mean + ER_B cached).

    Args:
        top_k:
            if None, concatenate all layers in their original order;
            otherwise concatenate the top-k layers by ER_B.
    """
    from sklearn.covariance import LedoitWolf
    import json

    names_path = os.path.join(train_inter_path, 'layer_names.json')
    if not os.path.exists(names_path):
        raise FileNotFoundError(f'layer_names.json not found in {train_inter_path}.')
    with open(names_path) as f:
        layer_names = json.load(f)
    L = len(layer_names)

    # ── Load cached ER_B and class means ──────────────────────────────────
    er_b_vals = []
    mean_list = []
    for l, layer_name in enumerate(layer_names):
        tag       = f'mm_pp_{layer_name}'
        er_b_path = os.path.join(path, f'{tag}_er_b.npy')
        mean_path = os.path.join(path, f'{tag}_mean.npy')
        if not os.path.exists(er_b_path) or not os.path.exists(mean_path):
            raise FileNotFoundError(
                f'Cached stats for {layer_name} not found. Run MM_plus_plus first.'
            )
        er_b_vals.append(float(np.load(er_b_path)))
        mean_list.append(np.load(mean_path).astype(np.float64))  # [C, D]

    er_b_full = np.array(er_b_vals, dtype=np.float64)
    if top_k is None or top_k >= L:
        active_idx = np.arange(L)
        active_label = 'all'
    else:
        top_k = min(top_k, L)
        active_idx = np.sort(np.argsort(er_b_full)[-top_k:])   # top-k by ER_B, in order
        active_label = f'top{top_k}'
    active_dims = [int(mean_list[i].shape[1]) for i in active_idx]
    D_joint = int(sum(active_dims))

    print(f'[MM++Cat] Concatenating {active_label} layers: '
          f'{[layer_names[i] for i in active_idx]}')
    print(f'[MM++Cat] ER_B of active layers: {er_b_full[active_idx].round(2)}')
    print(f'[MM++Cat] Active dims: {active_dims}')
    print(f'[MM++Cat] Joint feature dim: {D_joint}')

    # ── Cache tag ─────────────────────────────────────────────────────────
    fused_tag       = f'mm_pp_cat_{active_label}'
    fused_mean_path = os.path.join(path, f'{fused_tag}_means.npy')
    fused_prec_path = os.path.join(path, f'{fused_tag}_prec.npy')
    id_score_path   = os.path.join(path, f'{fused_tag}_id_scores.npy')

    if os.path.exists(fused_mean_path) and os.path.exists(fused_prec_path):
        print('[MM++Cat] Loading cached joint class means + precision...')
        cat_means  = np.load(fused_mean_path)   # [C, D_joint]
        prec_joint = np.load(fused_prec_path)   # [D_joint, D_joint]
    else:
        # ── Load and concatenate training features ─────────────────────────
        N_train           = len(train_labels)
        layer_feats_train = []   # list of [N_train, D] L2-normalised arrays

        for a_pos, l in enumerate(active_idx):
            layer_name = layer_names[l]
            print(f'[MM++Cat] Layer {l+1}/{L} ({layer_name}): loading train features...')
            parts = sorted(
                [fn for fn in os.listdir(train_inter_path)
                 if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
                key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
            )
            f_l = np.concatenate(
                [np.load(os.path.join(train_inter_path, p)) for p in parts], axis=0
            ).astype(np.float64)
            norms = np.linalg.norm(f_l, axis=-1, keepdims=True).clip(min=1e-10)
            f_l  /= norms
            layer_feats_train.append(f_l)

        # Concatenated joint features [N_train, D_joint] and class means [C, D_joint]
        cat_train = np.concatenate(layer_feats_train, axis=1)
        del layer_feats_train
        cat_means = np.concatenate([mean_list[l] for l in active_idx], axis=1)

        # Center within classes and fit Ledoit-Wolf
        print('[MM++Cat] Fitting Ledoit-Wolf on joint features...')
        cat_centered = cat_train - cat_means[train_labels]
        del cat_train
        lw = LedoitWolf(assume_centered=True)
        lw.fit(cat_centered)
        prec_joint = lw.precision_
        del cat_centered

        np.save(fused_mean_path, cat_means)
        np.save(fused_prec_path, prec_joint)

    # ── Scoring helper ────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean_t = torch.from_numpy(cat_means).to(device=device, dtype=torch.float64)    # [C, D_joint]
    prec_t = torch.from_numpy(prec_joint).to(device=device, dtype=torch.float64)   # [D_joint, D_joint]
    mean_prec_t = mean_t @ prec_t                                                   # [C, D_joint]
    mean_quad_t = (mean_prec_t * mean_t).sum(dim=1)                                 # [C]

    def _cat_and_score(layer_feats_list):
        parts = []
        for l in active_idx:
            feats = layer_feats_list[l].astype(np.float64)
            norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-10)
            parts.append(feats / norms)
        cat = np.concatenate(parts, axis=1)   # [N, D_joint]
        batch_size = 1024 if device.type == 'cuda' else 256
        scores = np.empty(cat.shape[0], dtype=np.float64)
        for start in range(0, cat.shape[0], batch_size):
            end = min(start + batch_size, cat.shape[0])
            x_t = torch.from_numpy(cat[start:end]).to(device=device, dtype=torch.float64)
            x_prec_t = x_t @ prec_t                                                     # [B, D_joint]
            x_quad_t = (x_prec_t * x_t).sum(dim=1, keepdim=True)                        # [B, 1]
            cross_t = x_t @ mean_prec_t.T                                               # [B, C]
            dist_t = x_quad_t - (2.0 * cross_t) + mean_quad_t.unsqueeze(0)              # [B, C]
            scores[start:end] = (-dist_t.min(dim=1).values).cpu().numpy()
        return scores

    # ── ID val scores (cached) ────────────────────────────────────────────
    if os.path.exists(id_score_path):
        score_id = np.load(id_score_path)
    else:
        print('[MM++Cat] Computing ID val scores...')
        score_id = _cat_and_score(layer_feats_val)
        np.save(id_score_path, score_id)

    print('[MM++Cat] Computing OOD scores...')
    score_ood = _cat_and_score(layer_feats_ood)

    return score_id, score_ood


def evaluate_MM_plus_plus_anchored(train_inter_path, layer_feats_val, layer_feats_ood,
                                   train_labels, path, n_classes=1000, alpha=0.5):
    """
    MM++ Final-Layer Anchor: fully unsupervised score-level aggregation.

    S(x) = α · S_final(x) + (1-α) · Σ_{l≠final} w_l · S_l(x)

    where:
      - S_final  is the MM++ score from the last layer (highest ER_B, strongest signal)
      - w_l      = ER_B_l / Σ_{j≠final} ER_B_j   (proportional to between-class ER)
      - α        fixes the minimum contribution of the final layer (the "anchor")

    Key properties:
      * Completely unsupervised — weights are derived solely from between-class ER
        (ER_B) which is computed on training features with no OOD data.
      * Final-layer guarantee — the final layer always carries exactly weight α,
        so the combined score can never drift far below the single-layer baseline.
      * α=1 → pure final-layer MM++ (the baseline); α=0 → ER_B-proportional over
        all layers (similar to MM++ but without Boltzmann sharpening).

    Pre-requisite: evaluate_MM_plus_plus() must have been run first to populate
    per-layer caches: mean, prec, er_b, id_scores.

    Args:
        train_inter_path: directory of per-layer training feature shards
        layer_feats_val:  list of L arrays [N_val, D_l]
        layer_feats_ood:  list of L arrays [N_ood, D_l]
        train_labels:     [N_train] integer class labels (unused; kept for API parity)
        path:             cache directory (same as used by evaluate_MM_plus_plus)
        n_classes:        number of ID classes
        alpha:            anchor weight for the final layer (default 0.5)

    Returns:
        score_id:  [N_val]  higher = more ID-like
        score_ood: [N_ood]  higher = more ID-like
    """
    import json

    names_path = os.path.join(train_inter_path, 'layer_names.json')
    if not os.path.exists(names_path):
        raise FileNotFoundError(
            f'layer_names.json not found in {train_inter_path}. '
            f'Run extract_intermediate_features() first.'
        )
    with open(names_path) as f:
        layer_names = json.load(f)
    L = len(layer_names)

    # ── Load cached per-layer statistics (all produced by evaluate_MM_plus_plus) ──
    er_b_vals     = []
    mean_list     = []
    prec_list     = []
    id_score_list = []

    for l, layer_name in enumerate(layer_names):
        tag = f'mm_pp_{layer_name}'
        er_b_path     = os.path.join(path, f'{tag}_er_b.npy')
        mean_path     = os.path.join(path, f'{tag}_mean.npy')
        prec_path     = os.path.join(path, f'{tag}_prec.npy')
        id_score_path = os.path.join(path, f'{tag}_id_scores.npy')

        for fpath, label in [(er_b_path, 'er_b'), (mean_path, 'mean'),
                             (prec_path, 'prec'), (id_score_path, 'id_scores')]:
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f'Cache [{label}] for layer {layer_name} not found in {path}. '
                    f'Run evaluate_MM_plus_plus() first.'
                )

        er_b_vals.append(float(np.load(er_b_path)))
        mean_list.append(np.load(mean_path).astype(np.float64))   # [C, D]
        prec_list.append(np.load(prec_path).astype(np.float64))   # [D, D]
        id_score_list.append(np.load(id_score_path))              # [N_val]

    er_b = np.array(er_b_vals)   # [L]

    # ── Identify final vs. intermediate layers ────────────────────────────────
    final_idx = L - 1                     # last layer = anchor
    inter_idx = list(range(L - 1))        # all others share (1-alpha)

    er_b_inter = er_b[inter_idx]          # [L-1]
    if er_b_inter.sum() > 0:
        w_inter = (1.0 - alpha) * er_b_inter / er_b_inter.sum()
    else:
        w_inter = np.zeros(len(inter_idx))

    print(f'[MM++Anchor] Final layer (anchor):     {layer_names[final_idx]}')
    print(f'[MM++Anchor] alpha (final weight):     {alpha}')
    print(f'[MM++Anchor] ER_B all layers:          {er_b.round(2)}')
    print(f'[MM++Anchor] Intermediate layer names: {[layer_names[i] for i in inter_idx]}')
    print(f'[MM++Anchor] Intermediate weights:     {w_inter.round(4)}')

    # ── Helper: L2-normalize + MM++ score at a single layer ──────────────────
    def _score_layer(l, feats_np):
        mean_t = torch.from_numpy(mean_list[l]).cuda().double()
        prec_t = torch.from_numpy(prec_list[l]).cuda().double()
        feats  = feats_np.astype(np.float64)
        norms  = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-10)
        feats /= norms
        scores = []
        for f in torch.from_numpy(feats).cuda().double():
            diff = f - mean_t                          # [C, D]
            d    = ((diff @ prec_t) * diff).sum(-1)    # [C]
            scores.append(-d.min().cpu().item())
        return np.array(scores)

    # ── Aggregate ID val scores (final layer already cached) ─────────────────
    s_final_id = id_score_list[final_idx]
    score_id   = alpha * s_final_id
    for a_pos, l in enumerate(inter_idx):
        if w_inter[a_pos] == 0.0:
            continue
        score_id = score_id + w_inter[a_pos] * id_score_list[l]

    # Per-sample floor: S(x) >= S_final(x)  ← guarantees >= single-layer baseline
    score_id = np.maximum(score_id, s_final_id)

    # ── Compute and aggregate OOD scores ─────────────────────────────────────
    print(f'[MM++Anchor] Scoring OOD — final layer ({layer_names[final_idx]})...')
    s_final_ood = _score_layer(final_idx, layer_feats_ood[final_idx])
    score_ood   = alpha * s_final_ood

    for a_pos, l in enumerate(inter_idx):
        if w_inter[a_pos] == 0.0:
            continue
        print(f'[MM++Anchor] Scoring OOD — layer {l} ({layer_names[l]})...')
        score_ood = score_ood + w_inter[a_pos] * _score_layer(l, layer_feats_ood[l])

    # Per-sample floor: S(x) >= S_final(x)  ← guarantees >= single-layer baseline
    score_ood = np.maximum(score_ood, s_final_ood)

    return score_id, score_ood


def evaluate_MM_plus_plus_paper(train_inter_path, layer_feats_val, layer_feats_ood,
                                train_labels, path, n_classes=1000, K=3, zscore=False):
    """
    MM++ (paper method): Top-K Information Bottleneck Gating using ER_B.

    Adaptation of Algorithm 1 & 2 using between-class ER (ER_B) instead of
    within-class ER. ER_B = ER of class-means covariance; it *increases* with
    depth as class structure sharpens. The log-rank gap formula is flipped to:

        Δ_l = ln(ER_B_l) - ln(ER_B_{l-1})   for l = 2..L

    A larger Δ_l means class structure emerged most sharply at layer l.
    Top-K layers with the largest Δ_l are selected (hard gate); all others
    receive zero weight. Weights: w_l = Δ_l / Σ_{j∈K} Δ_j.

    For ViT-B these correspond to the final 2-3 blocks where ER_B spikes.
    For ResNet50 they align with macro-block transitions.

    Requires evaluate_MM_plus_plus to have been run first to populate
    {tag}_er_b.npy, {tag}_mean.npy, {tag}_prec.npy, {tag}_id_scores.npy.

    Args:
        K: number of gated layers (paper uses K=3 for CNNs, K=1-2 for ViTs)
    """
    import json

    names_path = os.path.join(train_inter_path, 'layer_names.json')
    with open(names_path) as f:
        layer_names = json.load(f)

    L = len(layer_names)
    assert len(layer_feats_val) == L and len(layer_feats_ood) == L

    # ── Step 1: Load between-class ER_B from cache ───────────────────────────
    er_b = np.zeros(L)
    for l, layer_name in enumerate(layer_names):
        er_b_path = os.path.join(path, f'mm_pp_{layer_name}_er_b.npy')
        if not os.path.exists(er_b_path):
            raise FileNotFoundError(
                f'Between-class ER_B cache missing for layer {layer_name}. '
                f'Run evaluate_MM_plus_plus first to populate the cache.'
            )
        er_b[l] = float(np.load(er_b_path))

    print(f'[MM++Paper] Between-class ER_B: {er_b.round(2)}')

    # ── Steps 2-4: Select top-K layers by absolute ER_B ──────────────────────
    # Primary criterion: absolute ER_B (highest = most class-discriminative).
    # Tie-breaking: prefer deeper layers with a 0.2%-per-position depth bonus.
    # Rationale: the final normalization layer (e.g. ViT's LayerNorm after all
    # blocks) has nearly identical ER_B to the penultimate block (367.0 vs 367.4)
    # but dramatically better Mahalanobis performance because LayerNorm makes
    # the feature distribution more Gaussian — a structural property of any
    # norm layer that ER_B cannot capture.  The 0.2% bonus is small enough that
    # it never overrides genuine ER_B differences between non-adjacent layers
    # (e.g. the block_09→block_10 gap of 204 >> 0.2% × 367 ≈ 0.7 per step).
    #
    # Weights: w_l = ER_B_l / Σ_{j∈K} ER_B_j  (proportional to discriminability)
    K_eff      = min(K, L)
    depth_bonus = np.arange(L) * (er_b.max() * 0.002)   # 0.2% × max_ER_B per step
    sort_score  = er_b + depth_bonus
    rank_idx   = np.argsort(sort_score)[-K_eff:]         # indices of top-K
    sel_layers = rank_idx                                 # 0-indexed layer positions
    sel_erb    = er_b[sel_layers]                         # [K_eff]

    print(f'[MM++Paper] K={K_eff} selected layers (0-idx): {sel_layers.tolist()}')
    print(f'[MM++Paper] Selected names:  {[layer_names[i] for i in sel_layers]}')
    print(f'[MM++Paper] Selected ER_B:   {sel_erb.round(2)}')

    # ── Weights  w_l = ER_B_l / Σ_{j∈K} ER_B_j ─────────────────────────────
    weights = sel_erb / sel_erb.sum()

    print(f'[MM++Paper] Aggregation weights: {weights.round(4)}')

    # ── Steps 5-6: Score ONLY the K gated layers (Algorithm 2) ───────────────
    id_scores  = []
    ood_scores = []

    for rank, l in enumerate(sel_layers):
        layer_name   = layer_names[l]
        tag          = f'mm_pp_{layer_name}'
        mean_path    = os.path.join(path, f'{tag}_mean.npy')
        prec_path    = os.path.join(path, f'{tag}_prec.npy')
        id_score_path = os.path.join(path, f'{tag}_id_scores.npy')

        mean_l = np.load(mean_path)
        prec_l = np.load(prec_path)
        mean_t = torch.from_numpy(mean_l).cuda().double()
        prec_t = torch.from_numpy(prec_l).cuda().double()

        def _score(feats_np, mean_t=mean_t, prec_t=prec_t):
            feats_np = feats_np.astype(np.float64)
            norms = np.linalg.norm(feats_np, axis=-1, keepdims=True).clip(min=1e-10)
            feats_np = feats_np / norms
            out = []
            for f in torch.from_numpy(feats_np).cuda().double():
                diff = f - mean_t                        # [C, D]
                d    = ((diff @ prec_t) * diff).sum(-1)  # [C]
                out.append(-d.min().cpu().item())
            return np.array(out)

        # ID val scores (cached)
        if os.path.exists(id_score_path):
            s_val = np.load(id_score_path)
        else:
            print(f'[MM++Paper] Computing ID scores for layer {layer_name}...')
            s_val = _score(layer_feats_val[l])

        # OOD scores (not cached — per Algorithm 2, compute only for K layers)
        print(f'[MM++Paper] [{rank+1}/{K_eff}] Scoring OOD — layer {layer_name}...')
        s_ood = _score(layer_feats_ood[l])

        id_scores.append(s_val)
        ood_scores.append(s_ood)

    S_val = np.stack(id_scores)   # [K_eff, N_val]
    S_ood = np.stack(ood_scores)  # [K_eff, N_ood]

    if zscore:
        mean_l = S_val.mean(axis=1, keepdims=True)
        std_l  = S_val.std(axis=1, keepdims=True).clip(min=1e-8)
        S_val  = (S_val - mean_l) / std_l
        S_ood  = (S_ood - mean_l) / std_l

    score_id  = weights @ S_val  # [N_val]
    score_ood = weights @ S_ood  # [N_ood]

    return score_id, score_ood


def evaluate_MM_plus_plus_relative(train_inter_path, layer_feats_val, layer_feats_ood,
                                   train_labels, path, n_classes=1000,
                                   top_k=None, zscore=False):
    """
    Relative MM++ (RMM++): per-layer Relative Mahalanobis++ aggregated with ER_B weights.

    At each layer l, the score is:
        S_rel_l(x) = S_class_l(x) - S_global_l(x)

    where:
        S_class_l(x)  = -min_c (h̃_l - μ_{c,l})^T Λ_l (h̃_l - μ_{c,l})   [class-conditional]
        S_global_l(x) = -(h̃_l - μ_g_l)^T Λ_g_l (h̃_l - μ_g_l)            [global background]
        μ_g_l         = global mean of L2-normalized training features at layer l
        Λ_g_l         = Ledoit-Wolf precision of (train features - μ_g_l)

    The background subtraction removes the global geometry contribution that varies
    across layers, isolating the class-discriminative signal.  This is the multi-layer
    extension of Relative_Mahalanobis_norm (which consistently beats M++ at single layer).

    Aggregation uses the same Boltzmann-over-ER_B scheme as evaluate_MM_plus_plus.
    Optional: top_k layer selection and z-score normalization.
    """
    from sklearn.covariance import LedoitWolf
    import json

    names_path = os.path.join(train_inter_path, 'layer_names.json')
    with open(names_path) as f:
        layer_names = json.load(f)

    L = len(layer_names)
    assert len(layer_feats_val) == L and len(layer_feats_ood) == L

    layer_scores_val  = []
    layer_scores_ood  = []
    effective_ranks   = []
    effective_ranks_b = []

    for l, layer_name in enumerate(layer_names):
        tag        = f'mm_pp_{layer_name}'
        mean_path  = os.path.join(path, f'{tag}_mean.npy')
        prec_path  = os.path.join(path, f'{tag}_prec.npy')
        er_path    = os.path.join(path, f'{tag}_er.npy')
        er_b_path  = os.path.join(path, f'{tag}_er_b.npy')
        # global (background) statistics — new files for relative variant
        gmean_path = os.path.join(path, f'{tag}_gmean.npy')
        gprec_path = os.path.join(path, f'{tag}_gprec.npy')

        # ── Load or fit class-conditional statistics (reuse MM++ cache) ──────
        if os.path.exists(mean_path) and os.path.exists(prec_path) and os.path.exists(er_path):
            mean_l = np.load(mean_path)
            prec_l = np.load(prec_path)
            er_l   = float(np.load(er_path))
        else:
            raise FileNotFoundError(
                f'Class-conditional MM++ cache missing for layer {layer_name}. '
                f'Run evaluate_MM_plus_plus first to populate the cache.'
            )

        # ── Load or fit global (background) statistics ────────────────────────
        if os.path.exists(gmean_path) and os.path.exists(gprec_path):
            gmean_l = np.load(gmean_path)
            gprec_l = np.load(gprec_path)
        else:
            print(f'[RMM++] Layer {l+1}/{L} ({layer_name}): fitting global statistics...')
            parts = sorted(
                [fn for fn in os.listdir(train_inter_path)
                 if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
                key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
            )
            f_train = np.concatenate(
                [np.load(os.path.join(train_inter_path, p)) for p in parts], axis=0
            ).astype(np.float64)
            norms = np.linalg.norm(f_train, axis=-1, keepdims=True).clip(min=1e-10)
            f_train /= norms

            gmean_l = f_train.mean(axis=0)
            centered_g = f_train - gmean_l
            lw_g = LedoitWolf(assume_centered=True)
            lw_g.fit(centered_g)
            gprec_l = lw_g.precision_
            np.save(gmean_path, gmean_l)
            np.save(gprec_path, gprec_l)
            del f_train, centered_g

        # ── Load between-class ER (reuse MM++ cache) ─────────────────────────
        if os.path.exists(er_b_path):
            er_b_l = float(np.load(er_b_path))
        else:
            raise FileNotFoundError(
                f'ER_B cache missing for layer {layer_name}. '
                f'Run evaluate_MM_plus_plus first.'
            )

        effective_ranks.append(er_l)
        effective_ranks_b.append(er_b_l)

        # ── GPU tensors ───────────────────────────────────────────────────────
        mean_t  = torch.from_numpy(mean_l).cuda().double()
        prec_t  = torch.from_numpy(prec_l).cuda().double()
        gmean_t = torch.from_numpy(gmean_l).cuda().double()
        gprec_t = torch.from_numpy(gprec_l).cuda().double()

        def _rel_scores(feats_np):
            feats_np = feats_np.astype(np.float64)
            norms = np.linalg.norm(feats_np, axis=-1, keepdims=True).clip(min=1e-10)
            feats_np = feats_np / norms
            scores = []
            for f in torch.from_numpy(feats_np).cuda().double():
                # class-conditional score
                diff_c = f - mean_t                           # [C, D]
                s_cls  = -((diff_c @ prec_t) * diff_c).sum(-1).min().item()
                # global background score
                diff_g = f - gmean_t                          # [D]
                s_glb  = -(((diff_g @ gprec_t) * diff_g).sum()).item()
                scores.append(s_cls - s_glb)
            return np.array(scores)

        # ── ID val scores (cached per layer) ─────────────────────────────────
        id_score_path = os.path.join(path, f'{tag}_rel_id_scores.npy')
        if os.path.exists(id_score_path):
            s_val_l = np.load(id_score_path)
        else:
            print(f'[RMM++] Layer {l+1}/{L} ({layer_name}): computing ID relative scores...')
            s_val_l = _rel_scores(layer_feats_val[l])
            np.save(id_score_path, s_val_l)

        print(f'[RMM++] Layer {l+1}/{L} ({layer_name}): computing OOD relative scores...')
        s_ood_l = _rel_scores(layer_feats_ood[l])

        layer_scores_val.append(s_val_l)
        layer_scores_ood.append(s_ood_l)

    # ── Aggregation (same as MM++) ────────────────────────────────────────────
    er   = np.array(effective_ranks)
    er_b = np.array(effective_ranks_b)

    S_val = np.stack(layer_scores_val)   # [L, N_val]
    S_ood = np.stack(layer_scores_ood)   # [L, N_ood]

    if top_k is not None and top_k < len(er_b):
        keep  = np.argsort(er_b)[-top_k:]
        er_b  = er_b[keep]
        er    = er[keep]
        S_val = S_val[keep]
        S_ood = S_ood[keep]
        print(f'[RMM++] top_k={top_k}: keeping layers {keep.tolist()}')

    tau = float(np.max(np.log(er_b[1:] / er_b[:-1].clip(min=1e-8)))) if len(er_b) > 1 else 1.0
    log_w = +tau * er_b
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    print(f'[RMM++] Within-class ER:        {er.round(2)}')
    print(f'[RMM++] Between-class ER (ER_B):{er_b.round(2)}')
    print(f'[RMM++] Adaptive temperature:   τ = {tau:.4f}')
    print(f'[RMM++] Aggregation weights:    {w.round(4)}')

    if zscore:
        mean_l = S_val.mean(axis=1, keepdims=True)
        std_l  = S_val.std(axis=1, keepdims=True).clip(min=1e-8)
        S_val  = (S_val - mean_l) / std_l
        S_ood  = (S_ood - mean_l) / std_l

    score_id  = w @ S_val
    score_ood = w @ S_ood

    return score_id, score_ood


def evaluate_MM_plus_plus_xmaha_rank(train_inter_path, layer_feats_val, layer_feats_ood,
                                      train_labels, path, n_classes=1000, rank_by='erb'):
    """
    X-Maha style feature blending with rank-based fusion weights.

    Instead of raw ER_B values (which can produce extreme weight distributions),
    weights are proportional to the rank of each layer's criterion value:
        α_l = rank_l / Σ_{j=1}^{L} j  =  rank_l / (L*(L+1)/2)
    where rank_l ∈ {1, …, L} with L=highest criterion.

    Two ranking criteria are supported:
        rank_by='erb': rank by cached between-class ER_B (higher ER_B → higher rank)
        rank_by='var': rank by total variance Tr(f^T f / N) of raw (pre-L2-norm)
                       training features at each layer

    For homogeneous-width backbones (e.g. ViT, all layers same dim):
      1. Compute rank-based α_l for each layer.
      2. Blend: Φ(x) = Σ α_l · L2_norm(x^l)   (additive, same dim)
      3. Fit class means + Ledoit-Wolf precision on Φ(train).
      4. Score: S(x) = −min_c (Φ(x) − μ_c)ᵀ Σ⁻¹ (Φ(x) − μ_c)

    For heterogeneous-width backbones (e.g. ResNet, Swin, ConvNeXt):
      Additive fusion is not defined across different dims.  Instead, fall back to
      rank-weighted concatenation using the top-K layers by ER_B (default K=2):
      1. Select top-K layers by ER_B; compute rank weights α within that set.
      2. Scale each layer's L2-normalized features by α_l.
      3. Concatenate: Φ(x) = [α_1·h_1(x); α_2·h_2(x); …; α_K·h_K(x)]
      4. Fit class means + Ledoit-Wolf on Φ(train).
      5. Score: S(x) = −min_c (Φ(x) − μ_c)ᵀ Σ⁻¹ (Φ(x) − μ_c)

    Pre-requisite: evaluate_MM_plus_plus() must have been run first (ER_B + mean cached).

    Args:
        train_inter_path: directory of per-layer training feature shards
        layer_feats_val:  list of L arrays [N_val, D_l]
        layer_feats_ood:  list of L arrays [N_ood, D_l]
        train_labels:     [N_train] integer class labels
        path:             cache directory (same as used by evaluate_MM_plus_plus)
        n_classes:        number of ID classes
        rank_by:          'erb' or 'var' — criterion for ranking layers

    Returns:
        score_id:  [N_val]  higher = more ID-like
        score_ood: [N_ood]  higher = more ID-like
    """
    from sklearn.covariance import LedoitWolf
    import json

    assert rank_by in ('erb', 'var'), f"rank_by must be 'erb' or 'var', got '{rank_by}'"

    names_path = os.path.join(train_inter_path, 'layer_names.json')
    if not os.path.exists(names_path):
        raise FileNotFoundError(
            f'layer_names.json not found in {train_inter_path}. '
            f'Run extract_intermediate_features() first.'
        )
    with open(names_path) as f:
        layer_names = json.load(f)
    L = len(layer_names)

    # ── Pass 1: load criterion values per layer ────────────────────────────
    er_b_vals  = []
    mean_list  = []   # [L]  each entry: [C, D]
    for l, layer_name in enumerate(layer_names):
        tag       = f'mm_pp_{layer_name}'
        er_b_path = os.path.join(path, f'{tag}_er_b.npy')
        mean_path = os.path.join(path, f'{tag}_mean.npy')
        if not os.path.exists(er_b_path) or not os.path.exists(mean_path):
            raise FileNotFoundError(
                f'Cached stats for layer {layer_name} not found in {path}. '
                f'Run MM_plus_plus first to populate the cache.'
            )
        er_b_vals.append(float(np.load(er_b_path)))
        mean_list.append(np.load(mean_path).astype(np.float64))  # [C, D]

    er_b_full = np.array(er_b_vals)   # [L]
    dims      = np.array([mean_list[i].shape[1] for i in range(L)])
    is_homogeneous = len(np.unique(dims)) == 1

    # ── Helper: fast CPU Mahalanobis scoring ──────────────────────────────
    def _fast_maha(feats, means, prec):
        """feats:[N,D], means:[C,D], prec:[D,D] -> [N] scores"""
        Pmu    = means @ prec
        mu_Pmu = np.sum(means * Pmu, axis=1)
        xP     = feats @ prec
        xPx    = np.sum(feats * xP, axis=1)
        return -np.min(xPx[:, None] - 2 * (xP @ means.T) + mu_Pmu[None, :], axis=1)

    if is_homogeneous:
        # ── Homogeneous path: additive rank-weighted fusion ────────────────
        active_idx = np.arange(L)
        K = L

        if rank_by == 'erb':
            criterion = er_b_full
            print(f'[XMahaRank] Homogeneous dims ({dims[0]}). rank_by=erb, ER_B: {criterion.round(2)}')
        else:
            criterion = np.zeros(K)
            for a_pos, layer_name in enumerate(layer_names):
                parts = sorted(
                    [fn for fn in os.listdir(train_inter_path)
                     if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
                    key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
                )
                if not parts:
                    raise FileNotFoundError(f'No training features for {layer_name} in {train_inter_path}')
                f_l = np.concatenate(
                    [np.load(os.path.join(train_inter_path, p)) for p in parts], axis=0
                ).astype(np.float64)
                criterion[a_pos] = np.sum(f_l ** 2) / f_l.shape[0]
                del f_l
            print(f'[XMahaRank] Homogeneous dims ({dims[0]}). rank_by=var, Tr(F^TF)/N: {criterion.round(2)}')

        order = np.argsort(criterion)
        ranks = np.empty(K, dtype=float)
        ranks[order] = np.arange(1, K + 1)
        alpha = ranks / ranks.sum()

        print(f'[XMahaRank] Active layers:  {layer_names}')
        print(f'[XMahaRank] Ranks:          {ranks.astype(int).tolist()}')
        print(f'[XMahaRank] Fusion weights: {alpha.round(4)}')

        fused_tag       = f'mm_pp_xmaha_rank_{rank_by}'
        fused_mean_path = os.path.join(path, f'{fused_tag}_means.npy')
        fused_prec_path = os.path.join(path, f'{fused_tag}_prec.npy')
        id_score_path   = os.path.join(path, f'{fused_tag}_id_scores.npy')
        D = int(dims[0])

        if os.path.exists(fused_mean_path) and os.path.exists(fused_prec_path):
            print('[XMahaRank] Loading cached fused class means + precision...')
            fused_means = np.load(fused_mean_path)
            prec_fused  = np.load(fused_prec_path)
        else:
            N_train     = len(train_labels)
            fused_train = np.zeros((N_train, D), dtype=np.float64)
            fused_means = np.zeros((n_classes, D), dtype=np.float64)
            for a_pos, layer_name in enumerate(layer_names):
                print(f'[XMahaRank] Layer {a_pos+1}/{L} ({layer_name}): fusing train features...')
                parts = sorted(
                    [fn for fn in os.listdir(train_inter_path)
                     if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
                    key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
                )
                if not parts:
                    raise FileNotFoundError(f'No training features for {layer_name} in {train_inter_path}')
                f_l = np.concatenate(
                    [np.load(os.path.join(train_inter_path, p)) for p in parts], axis=0
                ).astype(np.float64)
                norms = np.linalg.norm(f_l, axis=-1, keepdims=True).clip(min=1e-10)
                f_l /= norms
                fused_train += alpha[a_pos] * f_l
                fused_means += alpha[a_pos] * mean_list[a_pos]
                del f_l

            print('[XMahaRank] Centering + fitting Ledoit-Wolf...')
            lw = LedoitWolf(assume_centered=True)
            lw.fit(fused_train - fused_means[train_labels])
            prec_fused = lw.precision_
            del fused_train
            np.save(fused_mean_path, fused_means)
            np.save(fused_prec_path, prec_fused)

        def _fuse_and_score(layer_feats_list):
            N     = layer_feats_list[0].shape[0]
            fused = np.zeros((N, D), dtype=np.float64)
            for a_pos in range(L):
                feats = layer_feats_list[a_pos].astype(np.float64)
                norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-10)
                fused += alpha[a_pos] * (feats / norms)
            return _fast_maha(fused, fused_means, prec_fused)

    else:
        # ── Heterogeneous path: rank-weighted concatenation ────────────────
        # Select top-2 layers by ER_B; rank weights within that set.
        het_top_k  = 2
        active_idx = np.sort(np.argsort(er_b_full)[-het_top_k:])
        K          = len(active_idx)
        criterion  = er_b_full[active_idx]

        order = np.argsort(criterion)
        ranks = np.empty(K, dtype=float)
        ranks[order] = np.arange(1, K + 1)
        alpha = ranks / ranks.sum()

        active_names = [layer_names[i] for i in active_idx]
        active_dims  = [int(dims[i]) for i in active_idx]
        D = int(sum(active_dims))

        print(f'[XMahaRank] Heterogeneous dims {dims.tolist()} — rank-weighted concat fallback.')
        print(f'[XMahaRank] Top-{het_top_k} layers by ER_B: {active_names}, ER_B: {criterion.round(2)}')
        print(f'[XMahaRank] Rank weights: {alpha.round(4)}, joint dim: {D}')

        fused_tag       = f'mm_pp_xmaha_rank_{rank_by}_cat{het_top_k}'
        fused_mean_path = os.path.join(path, f'{fused_tag}_means.npy')
        fused_prec_path = os.path.join(path, f'{fused_tag}_prec.npy')
        id_score_path   = os.path.join(path, f'{fused_tag}_id_scores.npy')

        if os.path.exists(fused_mean_path) and os.path.exists(fused_prec_path):
            print('[XMahaRank] Loading cached concat class means + precision...')
            fused_means = np.load(fused_mean_path)
            prec_fused  = np.load(fused_prec_path)
        else:
            N_train     = len(train_labels)
            cat_train   = np.zeros((N_train, D), dtype=np.float64)
            fused_means = np.zeros((n_classes, D), dtype=np.float64)
            col = 0
            for a_pos, l in enumerate(active_idx):
                layer_name = layer_names[l]
                d_l        = active_dims[a_pos]
                print(f'[XMahaRank] Loading train features for {layer_name} (dim={d_l})...')
                parts = sorted(
                    [fn for fn in os.listdir(train_inter_path)
                     if fn.startswith(f'layer_{layer_name}_features_') and fn.endswith('.npy')],
                    key=lambda x: int(x.split('_')[-1].replace('.npy', ''))
                )
                if not parts:
                    raise FileNotFoundError(f'No training features for {layer_name} in {train_inter_path}')
                f_l = np.concatenate(
                    [np.load(os.path.join(train_inter_path, p)) for p in parts], axis=0
                ).astype(np.float64)
                norms = np.linalg.norm(f_l, axis=-1, keepdims=True).clip(min=1e-10)
                f_l /= norms
                cat_train[:, col:col+d_l]   = alpha[a_pos] * f_l
                fused_means[:, col:col+d_l] = alpha[a_pos] * mean_list[l]
                col += d_l
                del f_l

            print('[XMahaRank] Centering + fitting Ledoit-Wolf on concatenated features...')
            lw = LedoitWolf(assume_centered=True)
            lw.fit(cat_train - fused_means[train_labels])
            prec_fused = lw.precision_
            del cat_train
            np.save(fused_mean_path, fused_means)
            np.save(fused_prec_path, prec_fused)

        def _fuse_and_score(layer_feats_list):
            N   = layer_feats_list[active_idx[0]].shape[0]
            cat = np.zeros((N, D), dtype=np.float64)
            col = 0
            for a_pos, l in enumerate(active_idx):
                d_l   = active_dims[a_pos]
                feats = layer_feats_list[l].astype(np.float64)
                norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-10)
                cat[:, col:col+d_l] = alpha[a_pos] * (feats / norms)
                col += d_l
            return _fast_maha(cat, fused_means, prec_fused)

    # ── ID val scores (cached) ─────────────────────────────────────────────
    if os.path.exists(id_score_path):
        score_id = np.load(id_score_path)
    else:
        print('[XMahaRank] Computing ID val scores...')
        score_id = _fuse_and_score(layer_feats_val)
        np.save(id_score_path, score_id)

    # ── OOD scores ─────────────────────────────────────────────────────────
    print('[XMahaRank] Computing OOD scores...')
    score_ood = _fuse_and_score(layer_feats_ood)

    return score_id, score_ood


def evaluate_MM_plus_plus_logrank(train_inter_path, layer_feats_val, layer_feats_ood,
                                   train_labels, path, n_classes=1000, K=2):
    """
    MM++ paper: Top-K Information Gating via discrete log-rank gap (Algorithm 1 & 2).

    Layer selection criterion (Eq. 9):
        Δ_l = ln(ER_{l-1}/D_{l-1}) - ln(ER_l/D_l)   for l ∈ {2..L}
    where ER_l is the within-class effective rank (Shannon entropy of the tied
    covariance eigenspectrum) and D_l is the feature dimension at layer l.

    A large Δ_l means the relative intrinsic dimensionality drops sharply at layer l
    — i.e., strong semantic compression / neural collapse.

    Aggregation weights (Eq. 10):
        w_l = Δ_l / Σ_{j∈K} Δ_j   for l ∈ K,   0 otherwise

    Final OOD score (Eq. 11):
        S_total(x) = Σ_{l∈K} w_l · S_l(x)

    Pre-requisite: evaluate_MM_plus_plus() must have been run first to cache
    per-layer ER (mm_pp_{name}_er.npy), means, precision, and ID scores.

    Args:
        train_inter_path: directory of per-layer training feature shards
        layer_feats_val:  list of L arrays [N_val, D_l]
        layer_feats_ood:  list of L arrays [N_ood, D_l]
        train_labels:     [N_train] integer class labels (not used — all stats cached)
        path:             cache directory used by evaluate_MM_plus_plus
        n_classes:        number of ID classes
        K:                number of layers to gate (paper uses K=2 for ViT, K=3 for CNN)

    Returns:
        score_id:  [N_val]  higher = more ID-like
        score_ood: [N_ood]  higher = more ID-like
    """
    import json

    names_path = os.path.join(train_inter_path, 'layer_names.json')
    if not os.path.exists(names_path):
        raise FileNotFoundError(f'layer_names.json not found in {train_inter_path}.')
    with open(names_path) as f:
        layer_names = json.load(f)
    L = len(layer_names)

    # ── Load cached within-class ER and dimension per layer ───────────────
    er_vals = []
    D_vals  = []
    for layer_name in layer_names:
        tag       = f'mm_pp_{layer_name}'
        er_path   = os.path.join(path, f'{tag}_er.npy')
        mean_path = os.path.join(path, f'{tag}_mean.npy')
        if not os.path.exists(er_path) or not os.path.exists(mean_path):
            raise FileNotFoundError(
                f'Cached stats for layer {layer_name} not found in {path}. '
                f'Run MM_plus_plus first.'
            )
        er_vals.append(float(np.load(er_path)))
        D_vals.append(int(np.load(mean_path).shape[1]))   # [C, D] → D

    er = np.array(er_vals, dtype=np.float64)   # [L]
    D  = np.array(D_vals,  dtype=np.float64)   # [L]

    # ── Relative rank and log-rank gaps (Eq. 9) ───────────────────────────
    rel_rank = er / D                              # [L]  ER_l / D_l
    log_rel  = np.log(rel_rank.clip(min=1e-10))   # [L]
    # delta[i] = Δ_{i+2}  (gap from layer i to layer i+1, 0-indexed)
    delta = log_rel[:-1] - log_rel[1:]             # [L-1], all should be >= 0 near collapse

    # ── Top-K selection ───────────────────────────────────────────────────
    K_eff = min(K, L - 1)
    top_pos = np.argsort(delta)[-K_eff:]          # positions in delta (0-indexed)
    top_idx = top_pos + 1                          # actual layer indices (layer producing Δ)

    sel_delta = delta[top_pos]
    denom     = sel_delta.sum()
    if denom <= 0:
        weights = np.ones(K_eff) / K_eff
    else:
        weights = sel_delta / denom

    print(f'[MM++LogRank] Relative rank (ER/D):  {rel_rank.round(6)}')
    print(f'[MM++LogRank] Log-rank gaps Δ:       {np.round(delta, 4)}')
    print(f'[MM++LogRank] Selected layer idx:    {sorted(top_idx.tolist())}')
    print(f'[MM++LogRank] Selected layers:       {[layer_names[i] for i in sorted(top_idx)]}')
    print(f'[MM++LogRank] Selected Δ values:     {sel_delta.round(4)}')
    print(f'[MM++LogRank] Aggregation weights:   {weights.round(4)}')

    # ── Load cached ID scores; compute OOD scores ─────────────────────────
    id_scores  = []
    ood_scores = []
    for pos, l in zip(top_pos, top_idx):
        layer_name    = layer_names[l]
        tag           = f'mm_pp_{layer_name}'
        id_score_path = os.path.join(path, f'{tag}_id_scores.npy')
        mean_path     = os.path.join(path, f'{tag}_mean.npy')
        prec_path     = os.path.join(path, f'{tag}_prec.npy')

        if not os.path.exists(id_score_path):
            raise FileNotFoundError(
                f'Cached ID scores for {layer_name} not found. Run MM_plus_plus first.'
            )
        id_scores.append(np.load(id_score_path))

        # OOD scores (computed online, not cached)
        mean_l = np.load(mean_path)
        prec_l = np.load(prec_path)
        mean_t = torch.from_numpy(mean_l).cuda().double()
        prec_t = torch.from_numpy(prec_l).cuda().double()

        feats = layer_feats_ood[l].astype(np.float64)
        norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-10)
        feats /= norms

        s_ood = []
        print(f'[MM++LogRank] Scoring OOD for layer {layer_name}...')
        for f in torch.from_numpy(feats).cuda().double():
            diff = f - mean_t
            d    = ((diff @ prec_t) * diff).sum(-1)
            s_ood.append(-d.min().cpu().item())
        ood_scores.append(np.array(s_ood))

    S_val = np.stack(id_scores)    # [K_eff, N_val]
    S_ood = np.stack(ood_scores)   # [K_eff, N_ood]

    score_id  = weights @ S_val
    score_ood = weights @ S_ood

    return score_id, score_ood
