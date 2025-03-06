import numpy as np


def generate_n_measurements(n_subjects: np.ndarray, n_measurements: int, random: bool=True, d_temp: float=3) -> np.ndarray:
    """Generates the number of measurements per subject.
    """
    # Ensure that every subject gets at least one measurement
    subject_measurements = np.ones(n_subjects, dtype=int)  # Start with one measurement per subject
    remaining_measurements = n_measurements - n_subjects  # Distribute the remaining measurements
    if random:
        probs = np.random.dirichlet([d_temp] * n_subjects)
    else:
        probs = np.ones(n_subjects) / n_subjects
    subject_measurements += np.random.multinomial(remaining_measurements, probs)
    
    return subject_measurements

def generate_subject_measurements(subject_measurements: np.ndarray, mean_range: tuple, base_cov: np.ndarray):
    """
    Simulates a dataset from M subjects with a total of N measurements.
    The dataset follows a multivariate Gaussian distribution with a block-structured covariance matrix,
    ensuring intra-subject correlation but no inter-subject correlation. Each subject has a unique mean.
    
    Parameters:
        M (int): Number of subjects.
        N (int): Total number of measurements.
        mean_range (tuple): Range (low, high) from which each subject's mean vector is sampled.
        base_cov (np.ndarray): Base covariance matrix defining intra-subject correlations.
    
    Returns:
        data (np.ndarray): Array of shape (N, dim) containing all measurements.
        subject_ids (np.ndarray): Array of shape (N,) containing subject IDs corresponding to each measurement.
    """
    dim = base_cov.shape[0]  # Dimensionality of the measurements
    
    data = []
    subject_ids = []
    subject_means = []
    
    for subject_id, num_measurements in enumerate(subject_measurements):
        # If only one subject, use a diagonal covariance matrix
        if subject_measurements.shape[0] == 1:
            subject_cov = np.diag(np.diag(base_cov))  # Enforce diagonal covariance
        else:
            subject_cov = base_cov
        subject_mean = np.random.uniform(mean_range[0], mean_range[1], size=dim)  # Unique mean per subject
        subject_data = np.random.multivariate_normal(subject_mean, subject_cov, size=num_measurements)
        
        data.append(subject_data)
        subject_ids.extend([subject_id] * num_measurements)
        subject_means.append(subject_mean)
    
    data = np.vstack(data)
    subject_ids = np.array(subject_ids)
    
    return data, subject_ids, subject_means
