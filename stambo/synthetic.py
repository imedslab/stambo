import numpy as np
import numpy.typing as npt

def generate_n_measurements(n_groups: int, n_measurements: int, random: bool=True, d_temp: float=3, min_samples: int=1, seed: int=None) -> npt.NDArray[np.int64]:
    """Generates the random number of measurements per group.
    
    Args:
        n_groups (int): Total number of groups / subjects.
        n_measurements (int): Total number of measurements.
        random (bool, optional): Whether to generate random number of measurements per group. Defaults to True.
        d_temp (float, optional): Parameter for the Dirichlet distribution. Defaults to 3.
        min_samples (int, optional): Minimum number of measurements per group. Defaults to 1.
        seed (int, optional): Seed for the random number generator. Defaults to None.
    Returns:
        np.ndarray: An array of shape (M,) containing the number of measurements generated per group.
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure that group / subject gets at least one measurement
    subject_measurements = np.ones(n_groups, dtype=int) * min_samples  # Start with min_samples measurements per group
    remaining_measurements = n_measurements - n_groups * min_samples  # Distribute the remaining measurements
    if random:
        probs = np.random.dirichlet([d_temp] * n_groups)
    else:
        probs = np.ones(n_groups) / n_groups
    subject_measurements += np.random.multinomial(remaining_measurements, probs)
    
    return subject_measurements

def generate_non_iid_measurements(measurements_per_group: npt.NDArray[np.int64], mean_range: tuple, base_cov: npt.NDArray[np.float64], seed: int=None):
    """
    Simulates a dataset from `measurements_per_group` subjects with a total of `n_measurements` measurements.
    The dataset follows a multivariate Gaussian distribution with a block-structured covariance matrix,
    ensuring intra-subject correlation but no inter-subject correlation. Each subject has a unique mean.
    
    Args:
        measurements_per_group (np.ndarray): Array of shape (M,) containing the number of measurements per subject.
        mean_range (tuple): Range (low, high) from which each subject's mean vector is sampled.
        base_cov (np.ndarray): Base covariance matrix defining intra-subject correlations.
        seed (int, optional): Seed for the random number generator. Defaults to None.
    Returns:
        np.ndarray: An array of shape (N, dim) containing all measurements.
        np.ndarray: An array of shape (N,) containing subject IDs corresponding to each measurement.
        np.ndarray: An array of shape (M, dim) containing the mean vector for each subject.
    """
    if seed is not None:
        np.random.seed(seed)

    dim = base_cov.shape[0]  # Dimensionality of the measurements
    
    data = []
    group_ids = []
    group_means = [np.random.uniform(mean_range[0], mean_range[1], size=dim) for _ in range(measurements_per_group.shape[0])]  # Unique mean per subject
    
    for subject_id, num_measurements in enumerate(measurements_per_group):
        # If only one subject, use a diagonal covariance matrix
        if measurements_per_group.shape[0] == 1:
            subject_cov = np.diag(np.diag(base_cov))  # Enforce diagonal covariance
        else:
            subject_cov = base_cov
        subject_data = np.random.multivariate_normal(group_means[subject_id], subject_cov, size=num_measurements)
        
        data.append(subject_data)
        group_ids.extend([subject_id] * num_measurements)
    
    data = np.vstack(data)
    group_ids = np.array(group_ids)
    
    return data, group_ids, group_means


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


