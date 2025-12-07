import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple


import numpy as np

def generate_non_iid_measurements(n_data: int, n_subjects: int, rho: float, 
    feat_corr: float=0.3,
    subj_sigma: float=0.1, noise_sigma: float=1, gamma: float=0, 
    mu_cls_1: float=0, mu_cls_2: float=2, 
    unevenness: float=0.3, class_imbalance: float=0.5, overlap: float=0.5, 
    seed: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates a dataset where subjects may be missing from one of the classes (dropped clusters).
    
    Args:
        n_data (int): Total number of measurements (sum of both classes).
        n_subjects (int): Total number of unique subjects.
        rho (float): Correlation between subject effects in the two classes.
        feat_corr (float): Correlation between features.
        subj_sigma (float): Subject-level variance.
        noise_sigma (float): Noise variance.
        gamma (float): Bias parameter.
        mu_cls_1, mu_cls_2 (float/array): Class means.
        unevenness (float): Dirichlet alpha. Lower = more uneven sample counts per subject.
        class_imbalance (float): Fraction of total data belonging to Class 1 (0.0 to 1.0).
        overlap (float): Probability that a subject appears in BOTH classes. 
                         (1-overlap) is split between being unique to C1 or unique to C2.
        seed (int): Random seed.
    """

    rng = np.random.default_rng(seed)
    
    mu_c1 = np.array(mu_cls_1)
    mu_c2 = np.array(mu_cls_2)
    n_dim = mu_c1.shape[0]

    if mu_c1.shape != mu_c2.shape:
        raise ValueError("Means must have same shape")

    # --- Step 0: Precompute Feature Correlation Transform ---
    # Construct the correlation matrix for features (R_feat)
    # For 2D: [[1, r], [r, 1]]
    if n_dim > 1:
        # Create a matrix with 1s on diagonal and feat_corr elsewhere
        R_feat = np.full((n_dim, n_dim), feat_corr)
        np.fill_diagonal(R_feat, 1.0)
        
        # Compute Cholesky decomposition L such that L @ L.T = R_feat
        # We use this to transform independent normal variables into correlated ones
        try:
            L_feat = np.linalg.cholesky(R_feat)
        except np.linalg.LinAlgError:
            raise ValueError("feat_corr resulted in a non-positive-definite matrix. Try lowering the correlation.")
    else:
        L_feat = np.eye(1)

    # --- Step 1: Sample Sizes & Subject States ---
    n_c1 = int(n_data * class_imbalance)
    n_c2 = n_data - n_c1

    # Subject states: 0=Shared, 1=Exc C1, 2=Exc C2
    probs = [overlap, (1-overlap)/2, (1-overlap)/2]
    subj_states = rng.choice([0, 1, 2], size=n_subjects, p=probs)
    
    active_c1 = np.where((subj_states == 0) | (subj_states == 1))[0]
    active_c2 = np.where((subj_states == 0) | (subj_states == 2))[0]

    # Safety fallbacks
    if len(active_c1) == 0: active_c1 = np.array([0])
    if len(active_c2) == 0: active_c2 = np.array([0])

    # --- Step 2: Distribute Counts ---
    def get_counts(n_samples, active_indices):
        n_active = len(active_indices)
        if n_samples < n_active:
             counts = np.zeros(n_active, dtype=int)
             chosen = rng.choice(n_active, n_samples, replace=False)
             counts[chosen] = 1
             return counts
        
        base = np.ones(n_active, dtype=int)
        remaining = n_samples - n_active
        if remaining > 0:
            probs = rng.dirichlet(np.ones(n_active) * unevenness)
            extra = rng.multinomial(remaining, probs)
            counts = base + extra
        else:
            counts = base
        return counts

    counts_c1 = get_counts(n_c1, active_c1)
    counts_c2 = get_counts(n_c2, active_c2)
    
    valid_c1 = counts_c1 > 0
    valid_c2 = counts_c2 > 0

    ids_expanded_c1 = np.repeat(active_c1[valid_c1], counts_c1[valid_c1])
    ids_expanded_c2 = np.repeat(active_c2[valid_c2], counts_c2[valid_c2])

    # --- Step 3: Generate Correlated Subject Effects ---
    # 1. Generate effects with Class-to-Class correlation (rho), but independent features.
    #    Covariance for classes: [[1, rho], [rho, 1]]
    cov_class = np.array([[1, rho], [rho, 1]])
    
    # Shape: (n_subjects, 2, n_dim)
    # We generate standard normals first, with class correlation
    b_s_white = np.zeros((n_subjects, 2, n_dim))
    for d in range(n_dim):
        b_s_white[:, :, d] = rng.multivariate_normal([0, 0], cov_class, size=n_subjects)
    
    # 2. Apply Feature-to-Feature correlation.
    #    We multiply the last dimension (features) by L_feat.T
    #    b_s_white is (N, 2, D). L_feat.T is (D, D).
    #    Matrix multiplication acts on the last dimension: resulting in (N, 2, D).
    #    We also scale by subj_sigma.
    b_s = np.matmul(b_s_white, L_feat.T) * subj_sigma

    # --- Step 4: Bias Calculation (Class 2) ---
    w_s = counts_c2[valid_c2] / counts_c2[valid_c2].sum()
    w_tilde = w_s - np.sum(w_s**2)
    bias_values = gamma * w_tilde 
    
    bias_expanded = np.repeat(bias_values, counts_c2[valid_c2])[:, np.newaxis]

    # --- Step 5: Generate Data with Correlated Noise ---
    
    # Class 1
    # Generate white noise (N, D)
    eps_c1_white = rng.standard_normal((len(ids_expanded_c1), n_dim))
    # Apply feature correlation and scale
    eps_c1 = np.matmul(eps_c1_white, L_feat.T) * noise_sigma
    
    b_s1_expanded = b_s[ids_expanded_c1, 0, :] 
    data_c1 = mu_c1 + b_s1_expanded + eps_c1

    # Class 2
    eps_c2_white = rng.standard_normal((len(ids_expanded_c2), n_dim))
    eps_c2 = np.matmul(eps_c2_white, L_feat.T) * noise_sigma
    
    b_s2_expanded = b_s[ids_expanded_c2, 1, :]
    data_c2 = mu_c2 + b_s2_expanded + bias_expanded + eps_c2

    # --- Step 6: Merge ---
    data = np.concatenate([data_c1, data_c2], axis=0)
    labels = np.concatenate([np.zeros(len(data_c1)), np.ones(len(data_c2))], axis=0)
    subject_ids = np.concatenate([ids_expanded_c1, ids_expanded_c2], axis=0)

    perm = rng.permutation(len(data))
    
    return data[perm], labels[perm], subject_ids[perm]



def generate_paired_data(n_subjects: int, n_measurements: int, effect_size: float=0.5, dim: int=1, min_samples: int=2, d_temp=2, mean_range: tuple=None, base_cov: np.ndarray=None, seed: int=None):
    """
    Generates paired data for two groups with a block-structured covariance matrix.
    The function simulates data when a patient was measured before and after a treatment.
    
    Args:
        n_subjects (int): Number of subjects in the study.
        n_measurements (int): Total number of measurements across all subjects.
        effect_size (float): Effect size of the treatment.
        dim (int): Dimensionality of each measurement.
        min_samples (int): Minimum number of measurements per subject.
        d_temp (float): Parameter for the Dirichlet distribution.
        mean_range (tuple): Specifies the low and high range for sampling each subject's mean vector.
        base_cov (np.ndarray, optional): Base covariance matrix used to define the intra-subject correlations. Defaults to None.
        seed (int, optional): Seed for the random number generator. Defaults to None.
    Returns:
        paired_data (np.ndarray): Array of shape (N, 2, dim) containing all pairs of measurements.
        s_ids_paired (np.ndarray): Array of shape (N,) containing subject IDs corresponding to each pair of measurements
    """ 
    if seed is not None:
        np.random.seed(seed)

    if mean_range is None:
        mean_range = np.array((-0.5, 0))
    if base_cov is None:
        base_cov = np.ones((dim, dim))  # Example structured covariance
        
    assert base_cov.shape == (dim, dim), f"Base covariance matrix must be of shape (dim, dim), but is {base_cov.shape}"
    
    n_measurements_per_subj = generate_n_measurements(n_subjects, n_measurements, random=True, d_temp=d_temp, min_samples=min_samples)

    data, subject_ids, _ = generate_non_iid_measurements(n_measurements_per_subj, mean_range, base_cov)

    # Let us now generate paired data. This will mimic the case
    # when we measure effect of a treatment within the same subject.
    # In the case of stambo, this mimics the case of model comparison.
    # That is: we have two models on the SAME data. 
    # A good example here is the accuracy metric. It nicely decomposes into sample-level accuracy,
    # where 1 is correct prediction, and 0 is incorrect prediction, and we can consider an 
    # imporved model as "treatment. 

    paired_data = []
    s_ids_paired = []
    for s_id in np.unique(subject_ids):
        subj_data = data[np.where(subject_ids == s_id)]
        np.random.shuffle(subj_data)
        for i in range(0, len(subj_data) - 1, 2):
            paired_data.append((subj_data[i], subj_data[i+1] + effect_size))
            s_ids_paired.append(s_id)
            
    paired_data = np.array(paired_data)
    s_ids_paired = np.array(s_ids_paired)
    
    return paired_data, s_ids_paired


