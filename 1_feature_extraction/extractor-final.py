import os
import glob
import logging
import time
import re
import argparse
import warnings # For suppressing ConvergenceWarning
from sklearn.exceptions import ConvergenceWarning # For NMF/KMeans warnings
import gc # For garbage collection

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from leaflet2 import AssignCurvedLeaflets # LiPyphilic
import healpy as hp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from collections import Counter
import multiprocessing # For parallelization
from scipy.optimize import linear_sum_assignment # Added for new method

# --- Global Configuration ---
# File names
TOPOLOGY_FILENAME = "bilayer.data"
TRAJECTORY_FILENAME = "concat-traj.lammpstrj"

# MDAnalysis Selections
SAT_SEL = "type 8" 
UNS_SEL = "type 1" 
CHOL_SEL = "type 9" 
HEAD_SEL = "type 1 8 9"

# LiPyphilic Parameters
LIPYPHILIC_LF_CUTOFF = 3.0
LIPYPHILIC_MIDPLANE_SEL = "type 9"
LIPYPHILIC_MIDPLANE_CUTOFF = 2.5

# HEALPix Parameters for Radial Distance Maps (NOT CHANGED)
NSIDE_OUTER_RADIAL = 16
NSIDE_INNER_RADIAL = 8
APPLY_ITERATIVE_MEDIAN_FILL_RADIAL = True
MAX_ITERATIONS_FILL_RADIAL = 10
APPLY_GAUSSIAN_SMOOTHING_RADIAL = True
GAUSSIAN_FWHM_DEG_RADIAL = 10.0

# NMFk Analysis Parameters
NSIDE_NMFK_PHASE_MAP_OUTER = 16
NSIDE_NMFK_PHASE_MAP_INNER = 8
N_FRAMES_FOR_NMFK_TRAINING = 101      
K_CLUSTERS_FINAL_KMEANS = 3        
WINDOW_SIZE_5_FRAME_NMFK = 5        # This will define the block size for tumbling windows
NMF_MAX_ITER_WINDOW_NMF = 150       

# NMFk K-selection parameters
NMFK_N_ENSEMBLE = 50                
NMFK_N_RESTARTS_PER_ENSEMBLE = 2   
NMFK_K_SEARCH_RANGE = range(2, 6)  
NMFK_NOISE_LEVEL_STD_FACTOR = 0.05 
NMF_MAX_ITER_K_SEARCH = 150         
CONTACT_CUTOFF_PHASE_ANGSTROM = 3.0 

NMF_MAX_ITER_FINAL_NMF = 600 # Kept from target, new logic uses NMF_MAX_ITER_WINDOW_NMF

LIPID_PAIRS_NMFK = [
    (UNS_SEL, SAT_SEL, "UNS", "SAT"),
    (SAT_SEL, CHOL_SEL, "SAT", "CHOL"),
    (CHOL_SEL, UNS_SEL, "CHOL", "UNS")
]

# --- Suppress Warnings ---
#warnings.filterwarnings("ignore", category=ConvergenceWarning)
#warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans") 
try:
    from healpy.utils import HealpyDeprecationWarning
    warnings.filterwarnings("ignore", category=HealpyDeprecationWarning)
except ImportError:
    warnings.filterwarnings("ignore", message=r'.*"verbose" was deprecated.*', category=FutureWarning, module=r'healpy.*')


# --- Logging Setup --- (unchanged)
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def setup_main_error_logger(log_file='main_error_log.txt', level=logging.ERROR):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(folder)s - %(message)s')
    logger = logging.getLogger('MainErrorLogger')
    if not logger.hasHandlers():
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(handler)
    return logger

main_error_logger = setup_main_error_logger()

# --- HEALPix Helper Functions (iterative_median_fill unchanged) ---
def iterative_median_fill(map_data, nside, max_iterations, logger_obj, enabled=True):
    if not enabled:
        return np.copy(map_data)
    filled_map = np.copy(map_data)
    for iteration in range(max_iterations):
        nan_indices = np.where(np.isnan(filled_map))[0]
        if len(nan_indices) == 0:
            logger_obj.debug(f"  Iterative fill: No NaNs. Iter {iteration}.")
            break
        n_filled_this_iteration = 0
        map_in_this_iteration = np.copy(filled_map)
        for pix_idx in nan_indices:
            neighbours_indices = hp.get_all_neighbours(nside, pix_idx)
            neighbours_indices = neighbours_indices[neighbours_indices != -1]
            if len(neighbours_indices) > 0:
                neighbour_values = map_in_this_iteration[neighbours_indices]
                valid_neighbour_values = neighbour_values[~np.isnan(neighbour_values)]
                if len(valid_neighbour_values) > 0:
                    median_val = np.median(valid_neighbour_values)
                    filled_map[pix_idx] = median_val
                    n_filled_this_iteration += 1
        logger_obj.debug(f"  Iterative fill: Iter {iteration+1}/{max_iterations}, filled {n_filled_this_iteration} NaNs.")
        if n_filled_this_iteration == 0:
            logger_obj.debug(f"  Iterative fill: No NaNs filled. Iter {iteration + 1}.")
            break
    else: 
        remaining_nans = np.sum(np.isnan(filled_map))
        if remaining_nans > 0: 
            logger_obj.warning(f"  Iterative fill: Max iter reached. {remaining_nans} NaNs may remain.")
    return filled_map

# --- NMFk Helper Functions ---
# find_optimal_K_nmfk_detailed (unchanged from previous response)
def find_optimal_K_nmfk_detailed(X_original, k_search_range, n_ensemble, n_restarts_per_nmf,
                                  noise_factor_std, nmf_max_iter_ksearch, logger_obj,
                                  random_seed_base=42, pair_name_diag="Diagnostic", leaflet_diag=""):
    logger_obj.debug(f"  Starting NMFk K-selection for {leaflet_diag} {pair_name_diag}, matrix {X_original.shape}...")
    avg_reconstruction_errors_per_K = []
    avg_silhouette_scores_W_cols_per_K = []
    X_original_non_negative = np.maximum(0, X_original)
    X_positive_elements = X_original_non_negative[X_original_non_negative > 1e-9]
    matrix_std_for_noise = np.std(X_positive_elements) if len(X_positive_elements) > 1 else 1.0
    if matrix_std_for_noise < 1e-9: matrix_std_for_noise = 1.0

    for K_test in k_search_range:
        current_K_reconstruction_errors_normalized_sq = []
        all_W_columns_for_this_K_test = []
        for i_ens in range(n_ensemble):
            noise = np.random.normal(0, noise_factor_std * matrix_std_for_noise, X_original_non_negative.shape)
            X_perturbed = np.maximum(0, X_original_non_negative + noise)
            best_W_this_ensemble_run = None
            min_re_sq_this_ensemble_run = np.inf
            for i_init in range(n_restarts_per_nmf):
                current_random_state = random_seed_base + K_test * 1000 + i_ens * 100 + i_init
                model = NMF(n_components=K_test, init='random', solver='mu',
                            beta_loss='frobenius', max_iter=nmf_max_iter_ksearch,
                            random_state=current_random_state, tol=1e-4)
                try:
                    W_run = model.fit_transform(X_perturbed); H_run = model.components_
                except Exception: continue
                if W_run.shape[1] != K_test: continue
                reconstruction_error_sq = np.linalg.norm(X_perturbed - W_run @ H_run, 'fro')**2
                if reconstruction_error_sq < min_re_sq_this_ensemble_run:
                    min_re_sq_this_ensemble_run = reconstruction_error_sq; best_W_this_ensemble_run = W_run
            if best_W_this_ensemble_run is not None:
                X_perturbed_norm_sq_fro = np.linalg.norm(X_perturbed, 'fro')**2
                if X_perturbed_norm_sq_fro < 1e-9: X_perturbed_norm_sq_fro = 1.0
                current_K_reconstruction_errors_normalized_sq.append(min_re_sq_this_ensemble_run / X_perturbed_norm_sq_fro)
                for k_col_idx in range(best_W_this_ensemble_run.shape[1]):
                    all_W_columns_for_this_K_test.append(best_W_this_ensemble_run[:, k_col_idx])
        if not all_W_columns_for_this_K_test or not current_K_reconstruction_errors_normalized_sq:
            avg_reconstruction_errors_per_K.append(1.0); avg_silhouette_scores_W_cols_per_K.append(-1.0)
            logger_obj.debug(f"    No valid NMF runs for {leaflet_diag} K={K_test} for {pair_name_diag}.")
            continue
        avg_reconstruction_errors_per_K.append(np.mean(current_K_reconstruction_errors_normalized_sq))
        if len(all_W_columns_for_this_K_test) >= K_test and K_test > 1:
            W_col_matrix = np.stack(all_W_columns_for_this_K_test, axis=0)
            W_col_matrix_normalized = normalize(W_col_matrix, norm='l2', axis=1)
            if W_col_matrix_normalized.shape[0] >= K_test and W_col_matrix_normalized.shape[0] >= 2 : 
                try:
                    kmeans_W_cols = KMeans(n_clusters=K_test, random_state=random_seed_base, n_init=10).fit(W_col_matrix_normalized)
                    if len(np.unique(kmeans_W_cols.labels_)) == K_test and len(np.unique(kmeans_W_cols.labels_)) > 1 :
                        silhouette_avg_W_cols = silhouette_score(W_col_matrix_normalized, kmeans_W_cols.labels_, metric='euclidean')
                        avg_silhouette_scores_W_cols_per_K.append(silhouette_avg_W_cols)
                    else: avg_silhouette_scores_W_cols_per_K.append(-1.0)
                except ValueError: avg_silhouette_scores_W_cols_per_K.append(-1.0)
            else: avg_silhouette_scores_W_cols_per_K.append(-1.0)
        elif K_test == 1: avg_silhouette_scores_W_cols_per_K.append(0.0) 
        else: avg_silhouette_scores_W_cols_per_K.append(-1.0)

    norm_recon_errors = np.array(avg_reconstruction_errors_per_K); sil_scores_W_cols = np.array(avg_silhouette_scores_W_cols_per_K)
    norm_recon_errors = np.nan_to_num(norm_recon_errors, nan=1.0); sil_scores_W_cols = np.nan_to_num(sil_scores_W_cols, nan=-1.0)
    combined_metric = sil_scores_W_cols - norm_recon_errors
    
    K_optimal = k_search_range[0] 
    if not np.all(np.isnan(combined_metric)) and len(combined_metric) > 0:
        valid_indices_metric = np.where((~np.isnan(combined_metric)) & (sil_scores_W_cols > -1.0))[0]
        if len(valid_indices_metric) > 0:
            optimal_K_idx_in_valid = np.nanargmax(combined_metric[valid_indices_metric])
            K_optimal = k_search_range[valid_indices_metric[optimal_K_idx_in_valid]]
        elif len(k_search_range) > 0 : 
            valid_recon_indices = np.where(~np.isnan(norm_recon_errors))[0]
            if len(valid_recon_indices) > 0:
                 K_optimal = k_search_range[valid_recon_indices[np.nanargmin(norm_recon_errors[valid_recon_indices])]]
    
    logger_obj.info(f"  NMFk K-selection for {leaflet_diag} {pair_name_diag} (K, Recon.Err, Sil(W), Combined):")
    for i, k_val in enumerate(k_search_range):
        combined_val_str = f"{combined_metric[i]:.3f}" if i < len(combined_metric) and not np.isnan(combined_metric[i]) else "N/A"
        logger_obj.info(f"    K={k_val}: {norm_recon_errors[i]:.3f}, {sil_scores_W_cols[i]:.3f}, {combined_val_str}")
    logger_obj.info(f"  Selected K_optimal = {K_optimal} for {leaflet_diag} {pair_name_diag}")
    return K_optimal

# get_optimal_K_trajectory (unchanged from previous response)
def get_optimal_K_trajectory(u, acl, ordered_resids_lipyphilic, leaflet_specs, lipid_pairs, global_config, logger_obj):
    N_TOTAL_FRAMES = len(u.trajectory)
    optimal_K_values_nmfk = {"OUTER": {}, "INNER": {}}
    n_frames_for_training = min(global_config['N_FRAMES_FOR_NMFK_TRAINING'], N_TOTAL_FRAMES)
    if n_frames_for_training == 0 and N_TOTAL_FRAMES > 0: n_frames_for_training = N_TOTAL_FRAMES

    last_frame_actual_idx = N_TOTAL_FRAMES - 1
    selected_frames_for_nmf_training = np.array([], dtype=int) 
    if N_TOTAL_FRAMES > 0:
        if N_TOTAL_FRAMES < n_frames_for_training or n_frames_for_training == 0:
            selected_frames_for_nmf_training = np.arange(N_TOTAL_FRAMES, dtype=int)
        elif n_frames_for_training == 1:
            selected_frames_for_nmf_training = np.array([last_frame_actual_idx], dtype=int)
        else:
            other_frames_count = n_frames_for_training - 1
            potential_other_frames = np.arange(N_TOTAL_FRAMES)
            if last_frame_actual_idx < len(potential_other_frames): 
                 potential_other_frames = np.delete(potential_other_frames, last_frame_actual_idx)
            
            sampled_other_indices = np.array([], dtype=int)
            if potential_other_frames.size > 0 :
                if len(potential_other_frames) < other_frames_count:
                    sampled_other_indices = np.random.choice(potential_other_frames, size=other_frames_count, replace=True)
                else:
                    sampled_other_indices = np.random.choice(potential_other_frames, size=other_frames_count, replace=False)
            
            selected_frames_for_nmf_training = np.sort(np.unique(np.concatenate((sampled_other_indices, [last_frame_actual_idx])))).astype(int)
    
    actual_n_frames_for_training = len(selected_frames_for_nmf_training)
    if actual_n_frames_for_training == 0:
        logger_obj.warning("NMFk training for optimal K skipped: No frames selected/available.")
        for leaflet_name, _ in leaflet_specs:
            for _, _, prim_name, sec_name in lipid_pairs:
                pair_name = f"{prim_name}_vs_{sec_name}"
                optimal_K_values_nmfk[leaflet_name][pair_name] = min(global_config['NMFK_K_SEARCH_RANGE']) if global_config['NMFK_K_SEARCH_RANGE'] else 2
        return optimal_K_values_nmfk
    logger_obj.info(f"NMFk training for optimal K using {actual_n_frames_for_training} frames: {selected_frames_for_nmf_training}")

    for leaflet_name, current_leaflet_id in leaflet_specs:
        logger_obj.info(f"NMFk Optimal K Training for {leaflet_name} leaflet:")
        for prim_sel_str, sec_sel_str, prim_name, sec_name in lipid_pairs:
            pair_name = f"{prim_name}_vs_{sec_name}"
            all_primary_lipid_resids_in_selected_frames = set()
            contacts_by_frame_then_lipid = {fid: {} for fid in selected_frames_for_nmf_training}
            for frame_idx_for_contact_matrix in selected_frames_for_nmf_training:
                u.trajectory[frame_idx_for_contact_matrix]
                la_frame = acl.leaflets[:, frame_idx_for_contact_matrix]
                leaflet_indices = np.where(la_frame == current_leaflet_id)[0]
                if not leaflet_indices.size: continue
                resids_curr_leaflet = set(ordered_resids_lipyphilic[leaflet_indices])
                prim_all_atoms = u.select_atoms(prim_sel_str); sec_all_atoms = u.select_atoms(sec_sel_str)
                prim_res_frame = [r for r in prim_all_atoms.residues if r.resid in resids_curr_leaflet]
                sec_res_frame  = [r for r in sec_all_atoms.residues  if r.resid in resids_curr_leaflet]
                if not prim_res_frame or not sec_res_frame: continue
                try:
                    prim_pos = np.array([res.atoms.center_of_mass() for res in prim_res_frame])
                    sec_pos  = np.array([res.atoms.center_of_mass() for res in sec_res_frame])
                except (ValueError, mda.exceptions.NoDataError): continue
                if prim_pos.ndim == 1: prim_pos = prim_pos.reshape(1,-1)
                if sec_pos.ndim == 1: sec_pos = sec_pos.reshape(1,-1)
                if prim_pos.shape[0] == 0 or sec_pos.shape[0] == 0: continue
                dist_matrix = distance_array(prim_pos, sec_pos, box=u.dimensions)
                contacts_per_prim = (dist_matrix < global_config['CONTACT_CUTOFF_PHASE_ANGSTROM']).sum(axis=1)
                for i_l, lipid_obj in enumerate(prim_res_frame):
                    all_primary_lipid_resids_in_selected_frames.add(lipid_obj.resid)
                    contacts_by_frame_then_lipid[frame_idx_for_contact_matrix][lipid_obj.resid] = contacts_per_prim[i_l]
            
            if not all_primary_lipid_resids_in_selected_frames:
                logger_obj.warning(f"    {pair_name} ({leaflet_name}): No primary lipids for NMFk optimal K. Skipping.")
                optimal_K_values_nmfk[leaflet_name][pair_name] = min(global_config['NMFK_K_SEARCH_RANGE']) if global_config['NMFK_K_SEARCH_RANGE'] else 2
                continue
            unique_lipids_list = sorted(list(all_primary_lipid_resids_in_selected_frames))
            lipid_to_idx_map = {r: i for i, r in enumerate(unique_lipids_list)}
            X_matrix_lipids = np.zeros((len(unique_lipids_list), actual_n_frames_for_training))
            for i_fcol, fid in enumerate(selected_frames_for_nmf_training):
                for l_resid, c_val in contacts_by_frame_then_lipid[fid].items():
                    if l_resid in lipid_to_idx_map: X_matrix_lipids[lipid_to_idx_map[l_resid], i_fcol] = c_val
            
            min_k_s = min(global_config['NMFK_K_SEARCH_RANGE']) if global_config['NMFK_K_SEARCH_RANGE'] else 2
            if X_matrix_lipids.shape[0] < max(2, min_k_s) or X_matrix_lipids.shape[1] < max(2, min_k_s) :
                logger_obj.warning(f"    {pair_name} ({leaflet_name}): Low data for NMFk optimal K ({X_matrix_lipids.shape}). Assigning default K.")
                optimal_K_values_nmfk[leaflet_name][pair_name] = min(global_config['NMFK_K_SEARCH_RANGE']) if global_config['NMFK_K_SEARCH_RANGE'] else 2
                continue

            K_optimal_L = find_optimal_K_nmfk_detailed(X_matrix_lipids, global_config['NMFK_K_SEARCH_RANGE'], global_config['NMFK_N_ENSEMBLE'], global_config['NMFK_N_RESTARTS_PER_ENSEMBLE'], global_config['NMFK_NOISE_LEVEL_STD_FACTOR'], global_config['NMF_MAX_ITER_K_SEARCH'], logger_obj, pair_name_diag=pair_name, leaflet_diag=leaflet_name)
            optimal_K_values_nmfk[leaflet_name][pair_name] = K_optimal_L
            
    logger_obj.info(f"Optimal K values (Trajectory-based for NMF components): {optimal_K_values_nmfk}")
    return optimal_K_values_nmfk

# Modified for tumbling/block window
def get_nmfk_labels_for_block(u, actual_block_frames, acl, ordered_resids_lipyphilic, current_leaflet_id,
                                     prim_sel_str, sec_sel_str, K_optimal_for_pair_nmf_components, global_config, logger_obj,
                                     prev_consistent_labels_for_pair=None):
    # N_TOTAL_FRAMES = len(u.trajectory) # Not strictly needed if actual_block_frames is well-defined
    if not actual_block_frames: 
        logger_obj.warning(f"    NMFk_Block: No frames in current block. Returning empty labels."); return {}
    
    # The 'central_ts_idx' concept is replaced by the block itself. Logging reflects the block.
    logger_obj.debug(f"    NMFk_Block: Processing block frames {actual_block_frames[0]}-{actual_block_frames[-1]} (K_NMF_components={K_optimal_for_pair_nmf_components})")
    
    all_primary_lipid_resids_in_block = set()
    contacts_by_frame_then_lipid_block = {fid: {} for fid in actual_block_frames}

    for frame_idx_in_block in actual_block_frames:
        u.trajectory[frame_idx_in_block] # Ensure universe is at the correct frame
        la_frame = acl.leaflets[:, frame_idx_in_block]
        leaflet_indices = np.where(la_frame == current_leaflet_id)[0]
        if not leaflet_indices.size: continue
        resids_curr_leaflet_in_block = set(ordered_resids_lipyphilic[leaflet_indices])
        
        prim_all_atoms = u.select_atoms(prim_sel_str); sec_all_atoms = u.select_atoms(sec_sel_str)
        prim_res_frame_block = [r for r in prim_all_atoms.residues if r.resid in resids_curr_leaflet_in_block]
        sec_res_frame_block = [r for r in sec_all_atoms.residues if r.resid in resids_curr_leaflet_in_block]
        
        if not prim_res_frame_block or not sec_res_frame_block: continue
        try:
            prim_pos = np.array([res.atoms.center_of_mass() for res in prim_res_frame_block])
            sec_pos = np.array([res.atoms.center_of_mass() for res in sec_res_frame_block])
        except (ValueError, mda.exceptions.NoDataError): continue
        
        if prim_pos.ndim == 1: prim_pos = prim_pos.reshape(1,-1)
        if sec_pos.ndim == 1: sec_pos = sec_pos.reshape(1,-1)
        if prim_pos.shape[0] == 0 or sec_pos.shape[0] == 0: continue
        
        dist_matrix = distance_array(prim_pos, sec_pos, box=u.dimensions)
        contacts_per_prim = (dist_matrix < global_config['CONTACT_CUTOFF_PHASE_ANGSTROM']).sum(axis=1)
        for i_l, lipid_obj in enumerate(prim_res_frame_block):
            all_primary_lipid_resids_in_block.add(lipid_obj.resid)
            contacts_by_frame_then_lipid_block[frame_idx_in_block][lipid_obj.resid] = contacts_per_prim[i_l]

    if not all_primary_lipid_resids_in_block: 
        logger_obj.debug(f"    NMFk_Block: No primary lipids found in block {actual_block_frames[0]}-{actual_block_frames[-1]}."); return {}
    
    unique_lipids_list_block = sorted(list(all_primary_lipid_resids_in_block))
    lipid_to_idx_map_block = {r: i for i, r in enumerate(unique_lipids_list_block)}
    # X_matrix_block: rows are lipids, columns are frames within the block
    X_matrix_block = np.zeros((len(unique_lipids_list_block), len(actual_block_frames)))
    for i_fcol, fid in enumerate(actual_block_frames):
        for l_resid, c_val in contacts_by_frame_then_lipid_block[fid].items():
            if l_resid in lipid_to_idx_map_block: X_matrix_block[lipid_to_idx_map_block[l_resid], i_fcol] = c_val
    
    num_kmeans_clusters_for_labels = global_config['K_CLUSTERS_FINAL_KMEANS']
    nmf_n_components = max(1, K_optimal_for_pair_nmf_components)
    
    per_lipid_labels_this_block_raw = {resid: 0 for resid in unique_lipids_list_block}
    
    if X_matrix_block.shape[0] < num_kmeans_clusters_for_labels or \
       X_matrix_block.shape[0] < nmf_n_components or \
       X_matrix_block.shape[1] == 0 :
        logger_obj.debug(f"    NMFk_Block: Low data for NMF/KMeans ({X_matrix_block.shape}, NMF_K={nmf_n_components}, KMeans_K={num_kmeans_clusters_for_labels}). Assigning default labels for block.")
        return per_lipid_labels_this_block_raw

    nmf_init_method = 'nndsvda'
    if nmf_n_components > min(X_matrix_block.shape[0], X_matrix_block.shape[1]):
        nmf_init_method = 'random'
        logger_obj.debug(f"    NMFk_Block: Switching NMF init to 'random' due to K={nmf_n_components} > min_dim={min(X_matrix_block.shape)}.")
    
    nmf_model_block = NMF(n_components=nmf_n_components, init=nmf_init_method, solver='mu', 
                           beta_loss='frobenius', max_iter=global_config['NMF_MAX_ITER_WINDOW_NMF'], 
                           random_state=42, tol=1e-4)
    try: 
        W_block = nmf_model_block.fit_transform(np.maximum(0, X_matrix_block))
    except ValueError as e: 
        logger_obj.warning(f"    NMFk_Block: NMF error with init='{nmf_init_method}' (K={nmf_n_components}, shape={X_matrix_block.shape}): {e}. Assigning default labels for block."); 
        return per_lipid_labels_this_block_raw

    if W_block.shape[0] < num_kmeans_clusters_for_labels:
        logger_obj.debug(f"    NMFk_Block: Not enough samples post-NMF ({W_block.shape[0]}) for KMeans (K={num_kmeans_clusters_for_labels}). Assigning default labels for block.")
        return per_lipid_labels_this_block_raw

    kmeans_block = KMeans(n_clusters=num_kmeans_clusters_for_labels, random_state=42, n_init=10)
    try:
        raw_kmeans_labels_for_lipids_array = kmeans_block.fit_predict(W_block)
    except ValueError as e:
        logger_obj.warning(f"    NMFk_Block: KMeans error (K={num_kmeans_clusters_for_labels}, W_shape={W_block.shape}): {e}. Assigning default labels for block.")
        return per_lipid_labels_this_block_raw

    raw_labels_dict_this_block = {
        unique_lipids_list_block[i]: raw_kmeans_labels_for_lipids_array[i] 
        for i in range(len(unique_lipids_list_block))
    }

    # Label consistency matching (Hungarian algorithm)
    if prev_consistent_labels_for_pair is None or not prev_consistent_labels_for_pair or num_kmeans_clusters_for_labels <=1:
        logger_obj.debug("    LabelMatching: No previous block labels, or K_KMeans=1. Using raw labels as consistent for this block.")
        consistent_labels_this_block = raw_labels_dict_this_block
    else:
        common_resids = set(raw_labels_dict_this_block.keys()) & set(prev_consistent_labels_for_pair.keys())
        if not common_resids:
            logger_obj.debug("    LabelMatching: No common resids with previous block. Using raw labels for current block.")
            consistent_labels_this_block = raw_labels_dict_this_block
        else:
            logger_obj.debug(f"    LabelMatching: Found {len(common_resids)} common resids for matching between blocks.")
            overlap_matrix = np.zeros((num_kmeans_clusters_for_labels, num_kmeans_clusters_for_labels), dtype=int)
            for resid in common_resids:
                prev_label = prev_consistent_labels_for_pair[resid]
                curr_raw_label = raw_labels_dict_this_block[resid]
                if 0 <= prev_label < num_kmeans_clusters_for_labels and 0 <= curr_raw_label < num_kmeans_clusters_for_labels:
                    overlap_matrix[prev_label, curr_raw_label] += 1
            
            row_ind, col_ind = linear_sum_assignment(-overlap_matrix) 
            label_mapping = {curr_raw_idx: prev_cons_idx for prev_cons_idx, curr_raw_idx in zip(row_ind, col_ind)}
            
            consistent_labels_this_block = {}
            for resid, raw_label in raw_labels_dict_this_block.items():
                consistent_labels_this_block[resid] = label_mapping.get(raw_label, raw_label) 

            logger_obj.debug(f"    LabelMatching: Overlap Matrix (Prev_Block_Cons x Curr_Block_Raw):\n{overlap_matrix}")
            logger_obj.debug(f"    LabelMatching: Assignment (Prev_Cons_idx <- Curr_Raw_idx): {list(zip(row_ind, col_ind))}")
            logger_obj.debug(f"    LabelMatching: Effective Permutation from Curr_Raw to New_Cons: {label_mapping}")
    return consistent_labels_this_block


# get_primary_lipid_coords_and_resids_in_leaflet_frame (unchanged from previous response)
def get_primary_lipid_coords_and_resids_in_leaflet_frame(u, ts_idx, acl, ordered_resids_lipyphilic, current_leaflet_id, prim_sel_str, head_sel_for_com, logger_obj):
    u.trajectory[ts_idx]
    la_frame = acl.leaflets[:, ts_idx]
    vesicle_com_frame = None; all_heads_ag_frame = u.select_atoms(head_sel_for_com)
    if all_heads_ag_frame.n_atoms > 0:
        try: vesicle_com_frame = all_heads_ag_frame.center_of_mass()
        except Exception as e_com: logger_obj.warning(f"Frame {ts_idx}: Error calculating vesicle COM: {e_com}. Using origin."); vesicle_com_frame = np.array([0.,0.,0.])
    else: logger_obj.warning(f"Frame {ts_idx}: No head atoms for vesicle COM (sel: '{head_sel_for_com}'). Using origin."); vesicle_com_frame = np.array([0.,0.,0.])
    
    leaflet_indices = np.where(la_frame == current_leaflet_id)[0]
    if not leaflet_indices.size: logger_obj.debug(f"CoordGetter: Leaflet {current_leaflet_id} empty in frame {ts_idx}."); return None, None, vesicle_com_frame
    resids_in_leaflet = set(ordered_resids_lipyphilic[leaflet_indices])
    if not resids_in_leaflet: logger_obj.debug(f"CoordGetter: No resids for leaflet {current_leaflet_id} in frame {ts_idx}."); return None, None, vesicle_com_frame
    
    primary_lipid_coords_list, primary_lipid_resids_list = [], []
    all_prim_type_atoms = u.select_atoms(prim_sel_str)
    if all_prim_type_atoms.n_atoms == 0: logger_obj.debug(f"CoordGetter: Sel '{prim_sel_str}' found no atoms in frame {ts_idx}."); return None, None, vesicle_com_frame
    
    prim_res_in_leaflet_candidates = [res for res in all_prim_type_atoms.residues if res.resid in resids_in_leaflet]

    if not prim_res_in_leaflet_candidates: logger_obj.debug(f"CoordGetter: No lipids of type '{prim_sel_str}' in leaflet {current_leaflet_id}."); return None, None, vesicle_com_frame
    
    for res in prim_res_in_leaflet_candidates:
        try:
            if res.atoms.n_atoms > 0: 
                coord = res.atoms.center_of_mass(); primary_lipid_coords_list.append(coord); primary_lipid_resids_list.append(res.resid)
            else: logger_obj.warning(f"CoordGetter: Residue {res.resid} (type {prim_sel_str}) has no atoms.")
        except Exception as e_com_res: logger_obj.warning(f"CoordGetter: Error getting COM for resid {res.resid} (type {prim_sel_str}): {e_com_res}")
    
    if not primary_lipid_coords_list: logger_obj.debug(f"CoordGetter: No valid coords for '{prim_sel_str}' in leaflet {current_leaflet_id}."); return None, None, vesicle_com_frame
    return np.array(primary_lipid_coords_list), np.array(primary_lipid_resids_list), vesicle_com_frame

# generate_map_mode_label (unchanged from previous response)
def generate_map_mode_label(primary_lipid_coords, primary_lipid_resids, per_lipid_label_dict, vesicle_com, nside, npix):
    hmap = np.full(npix, np.nan)
    if primary_lipid_coords is None or primary_lipid_coords.shape[0] == 0 or not per_lipid_label_dict or vesicle_com is None: return hmap
    coords_shifted = primary_lipid_coords - vesicle_com; dist_from_com = np.linalg.norm(coords_shifted, axis=1)
    valid_mask = dist_from_com > 1e-6; valid_coords = coords_shifted[valid_mask]; valid_resids = primary_lipid_resids[valid_mask]
    if valid_coords.shape[0] == 0: return hmap
    pix_indices = hp.vec2pix(nside, valid_coords[:,0], valid_coords[:,1], valid_coords[:,2])
    pixel_labels_for_mode = [[] for _ in range(npix)]
    for i, pix_idx in enumerate(pix_indices):
        label = per_lipid_label_dict.get(valid_resids[i]) 
        if label is not None and label != -1: 
            pixel_labels_for_mode[pix_idx].append(label)
    for i in range(npix):
        if pixel_labels_for_mode[i]: counts = Counter(pixel_labels_for_mode[i]); hmap[i] = counts.most_common(1)[0][0]
    return hmap

# apply_healpix_mode_filter (unchanged from previous response)
def apply_healpix_mode_filter(hmap_in, nside):
    if np.all(np.isnan(hmap_in)): return hmap_in 
    npix = hp.nside2npix(nside); hmap_out = np.copy(hmap_in)
    for pix_idx in range(npix):
        if np.isnan(hmap_in[pix_idx]): continue 
        values_for_mode = [hmap_in[pix_idx]] 
        neighbours_indices = hp.get_all_neighbours(nside, pix_idx) 
        valid_neighbor_values = [hmap_in[n_idx] for n_idx in neighbours_indices if n_idx != -1 and not np.isnan(hmap_in[n_idx])]
        values_for_mode.extend(valid_neighbor_values)
        if values_for_mode: 
            counts = Counter(values_for_mode); hmap_out[pix_idx] = counts.most_common(1)[0][0]
    return hmap_out

# --- Main Processing Function for a Single Config ---
def process_config_folder(config_folder_path, global_config):
    folder_name = os.path.basename(config_folder_path)
    output_dir = os.path.join(config_folder_path, "analysis_output_test_final")
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(folder_name, os.path.join(output_dir, f"{folder_name}_processing_log.txt"))
    logger.info(f"Processing: {folder_name}")
    total_time_start = time.time()

    u = None
    acl = None
    all_frames_radial_outer, all_frames_radial_inner = None, None
    all_frames_nmfk_phase_maps = None 

    try:
        # --- File Loading and Initial Setup (largely unchanged) ---
        topo_file = os.path.join(config_folder_path, global_config['TOPOLOGY_FILENAME'])
        traj_file = os.path.join(config_folder_path, global_config['TRAJECTORY_FILENAME'])

        if not os.path.exists(topo_file) or not os.path.exists(traj_file):
            logger.error(f"File missing. Skipping.")
            main_error_logger.error(f"File missing", extra={'folder': folder_name})
            return f"FAILED: File missing in {folder_name}"

        try:
            u = mda.Universe(topo_file, traj_file, format="LAMMPSDUMP")
        except Exception:
            u = mda.Universe(topo_file, traj_file, topology_format='DATA', format='LAMMPSDUMP')
            
        N_TOTAL_FRAMES = len(u.trajectory)
        logger.info(f"Frames: {N_TOTAL_FRAMES} (Atoms: {u.atoms.n_atoms})")
        if N_TOTAL_FRAMES == 0:
            logger.error("No frames in trajectory. Skipping.")
            main_error_logger.error(f"No frames in trajectory", extra={'folder': folder_name})
            return f"FAILED: No frames in {folder_name}"

        ordered_resids_lipyphilic = u.select_atoms(global_config['HEAD_SEL']).residues.resids
        if not ordered_resids_lipyphilic.size:
            logger.error("No lipids for LiPyphilic. Check HEAD_SEL.")
            main_error_logger.error(f"No lipids for LiPyphilic", extra={'folder': folder_name})
            return f"FAILED: No LiPyphilic lipids in {folder_name}"

        acl_time_start = time.time()
        acl = AssignCurvedLeaflets(universe=u, lipid_sel=global_config['HEAD_SEL'],
                                     lf_cutoff=global_config['LIPYPHILIC_LF_CUTOFF'],
                                     midplane_sel=global_config['LIPYPHILIC_MIDPLANE_SEL'],
                                     midplane_cutoff=global_config['LIPYPHILIC_MIDPLANE_CUTOFF'])
        acl.run(verbose=False)
        logger.info(f"Leaflet assignment time: {time.time() - acl_time_start:.2f}s")
        
        OUTER_LEAFLET_ID_FROM_LIPY = 1 
        INNER_LEAFLET_ID_FROM_LIPY = -1
        ACTUAL_OUTER_LEAFLET_ID = OUTER_LEAFLET_ID_FROM_LIPY
        ACTUAL_INNER_LEAFLET_ID = INNER_LEAFLET_ID_FROM_LIPY

        if N_TOTAL_FRAMES > 0 and len(ordered_resids_lipyphilic) > 0 :
            original_frame_idx = u.trajectory.frame 
            u.trajectory[0] 
            assignments_frame0 = acl.leaflets[:, 0]
            indices_leaflet1_frame0 = np.where(assignments_frame0 == OUTER_LEAFLET_ID_FROM_LIPY)[0]
            indices_leaflet_minus1_frame0 = np.where(assignments_frame0 == INNER_LEAFLET_ID_FROM_LIPY)[0]
            if len(indices_leaflet1_frame0) > 0 and len(indices_leaflet_minus1_frame0) > 0:
                resids_leaflet1 = ordered_resids_lipyphilic[indices_leaflet1_frame0]
                resids_leaflet_minus1 = ordered_resids_lipyphilic[indices_leaflet_minus1_frame0]
                sel_str_l1 = f"({global_config['HEAD_SEL']}) and resid {' '.join(map(str, resids_leaflet1))}"
                sel_str_lm1 = f"({global_config['HEAD_SEL']}) and resid {' '.join(map(str, resids_leaflet_minus1))}"
                lipids_in_leaflet1_heads = u.select_atoms(sel_str_l1) if resids_leaflet1.size > 0 else u.select_atoms("resname NONEXISTENT")
                lipids_in_leaflet_minus1_heads = u.select_atoms(sel_str_lm1) if resids_leaflet_minus1.size > 0 else u.select_atoms("resname NONEXISTENT")
                if lipids_in_leaflet1_heads.n_atoms > 0 and lipids_in_leaflet_minus1_heads.n_atoms > 0:
                    all_head_beads_ag = u.select_atoms(global_config['HEAD_SEL'])
                    if all_head_beads_ag.n_atoms > 0:
                        com_all_heads_system = all_head_beads_ag.center_of_mass()
                        r_leaflet1 = np.mean(np.linalg.norm(lipids_in_leaflet1_heads.positions - com_all_heads_system, axis=1))
                        r_leaflet_minus1 = np.mean(np.linalg.norm(lipids_in_leaflet_minus1_heads.positions - com_all_heads_system, axis=1))
                        logger.info(f"  Leaflet convention check (Frame 0): Avg radius LiPyphilic '1': {r_leaflet1:.2f} Å, LiPyphilic '-1': {r_leaflet_minus1:.2f} Å")
                        if r_leaflet1 < r_leaflet_minus1: 
                            ACTUAL_OUTER_LEAFLET_ID, ACTUAL_INNER_LEAFLET_ID = INNER_LEAFLET_ID_FROM_LIPY, OUTER_LEAFLET_ID_FROM_LIPY
                            logger.warning("  Convention flipped: LiPyphilic '1' is INNER, '-1' is OUTER. Using this determination.")
                        else:
                            logger.info("  Convention confirmed: LiPyphilic '1' is OUTER, '-1' is INNER.")
                    else: logger.warning("  COM check skipped: No head atoms found for overall COM.")
                else: logger.warning("  COM check skipped: Not enough lipids in both LiPyphilic leaflet types in frame 0.")
            else: logger.warning("  COM check skipped: LiPyphilic leaflets '1' or '-1' (or both) are empty in frame 0.")
            u.trajectory[original_frame_idx] 
        logger.info(f"Using ACTUAL_OUTER_LEAFLET_ID = {ACTUAL_OUTER_LEAFLET_ID}, ACTUAL_INNER_LEAFLET_ID = {ACTUAL_INNER_LEAFLET_ID}")
        
        # --- Radial Maps Section (Unchanged) ---
        radial_maps_time_start = time.time()
        npix_outer_radial = hp.nside2npix(global_config['NSIDE_OUTER_RADIAL'])
        npix_inner_radial = hp.nside2npix(global_config['NSIDE_INNER_RADIAL'])
        all_frames_radial_outer = np.full((N_TOTAL_FRAMES, npix_outer_radial), np.nan)
        all_frames_radial_inner = np.full((N_TOTAL_FRAMES, npix_inner_radial), np.nan)
        gaussian_fwhm_rad_radial = np.deg2rad(global_config['GAUSSIAN_FWHM_DEG_RADIAL'])
        for i_frame_radial in range(N_TOTAL_FRAMES): 
            ts_radial = u.trajectory[i_frame_radial] 
            logger.debug(f"  Processing radial maps frame {i_frame_radial}...")
            all_heads_ag_radial = u.select_atoms(global_config['HEAD_SEL']) 
            if all_heads_ag_radial.n_atoms == 0: continue
            vesicle_com_radial = all_heads_ag_radial.center_of_mass()
            la_this_frame_radial = acl.leaflets[:, i_frame_radial]
            for leaflet_type, leaflet_id, nside_radial, storage_array in [
                ("OUTER", ACTUAL_OUTER_LEAFLET_ID, global_config['NSIDE_OUTER_RADIAL'], all_frames_radial_outer),
                ("INNER", ACTUAL_INNER_LEAFLET_ID, global_config['NSIDE_INNER_RADIAL'], all_frames_radial_inner)
            ]:
                hmap_avg = np.full(hp.nside2npix(nside_radial), np.nan)
                indices = np.where(la_this_frame_radial == leaflet_id)[0]
                if indices.size > 0:
                    resids_leaflet = ordered_resids_lipyphilic[indices]
                    sel_all_heads = u.select_atoms(global_config['HEAD_SEL'])
                    sel_all_heads_resids = sel_all_heads.resids
                    sel_all_heads_pos = sel_all_heads.positions
                    mask_leaflet_heads = np.isin(sel_all_heads_resids, resids_leaflet)
                    beads_coords = sel_all_heads_pos[mask_leaflet_heads]
                    if beads_coords.shape[0] > 0:
                        coords_shifted = beads_coords - vesicle_com_radial
                        rad_dist = np.linalg.norm(coords_shifted, axis=1)
                        valid_mask = rad_dist > 1e-6
                        if np.any(valid_mask):
                            pix_idx = hp.vec2pix(nside_radial, coords_shifted[valid_mask,0], coords_shifted[valid_mask,1], coords_shifted[valid_mask,2])
                            rad_dist_valid = rad_dist[valid_mask]
                            unique_pix, pix_inverse, pix_counts = np.unique(pix_idx, return_inverse=True, return_counts=True)
                            avg_vals = np.bincount(pix_inverse, weights=rad_dist_valid) / pix_counts
                            hmap_avg[unique_pix] = avg_vals
                proc_map = np.copy(hmap_avg)
                if global_config['APPLY_ITERATIVE_MEDIAN_FILL_RADIAL'] and np.any(np.isnan(proc_map)):
                    proc_map = iterative_median_fill(proc_map, nside_radial, global_config['MAX_ITERATIONS_FILL_RADIAL'], logger, global_config['APPLY_ITERATIVE_MEDIAN_FILL_RADIAL'])
                if global_config['APPLY_GAUSSIAN_SMOOTHING_RADIAL'] and not np.all(np.isnan(proc_map)):
                    proc_map = hp.smoothing(proc_map, fwhm=gaussian_fwhm_rad_radial, verbose=False)
                storage_array[i_frame_radial, :] = proc_map
        np.save(os.path.join(output_dir, "area_outer_radial_dist.npy"), all_frames_radial_outer)
        np.save(os.path.join(output_dir, "area_inner_radial_dist.npy"), all_frames_radial_inner)
        logger.info(f"Radial maps time: {time.time() - radial_maps_time_start:.2f}s")

        # --- NMFk Analysis (Tumbling Window with Adjusted Last Block) ---
        nmfk_analysis_start_time = time.time()
        
        leaflet_specs_for_optimal_k = [
            ("OUTER", ACTUAL_OUTER_LEAFLET_ID), 
            ("INNER", ACTUAL_INNER_LEAFLET_ID)
        ]
        optimal_K_trajectory = get_optimal_K_trajectory(
            u, acl, ordered_resids_lipyphilic, 
            leaflet_specs_for_optimal_k, 
            global_config['LIPID_PAIRS_NMFK'], global_config, logger
        )
        logger.info(f"NMFk optimal K determination time: {time.time() - nmfk_analysis_start_time:.2f}s")
        logger.info(f"Optimal K values from trajectory: {optimal_K_trajectory}")

        nmfk_phase_map_gen_time_start = time.time()
        npix_nmfk_outer = hp.nside2npix(global_config['NSIDE_NMFK_PHASE_MAP_OUTER'])
        npix_nmfk_inner = hp.nside2npix(global_config['NSIDE_NMFK_PHASE_MAP_INNER'])

        all_frames_nmfk_phase_maps = {
            "OUTER": {pair_key[2]+"_vs_"+pair_key[3]: np.full((N_TOTAL_FRAMES, npix_nmfk_outer), np.nan)
                      for pair_key in global_config['LIPID_PAIRS_NMFK']},
            "INNER": {pair_key[2]+"_vs_"+pair_key[3]: np.full((N_TOTAL_FRAMES, npix_nmfk_inner), np.nan)
                      for pair_key in global_config['LIPID_PAIRS_NMFK']}
        }

        previous_window_consistent_labels_cache = {"OUTER": {}, "INNER": {}}
        last_processed_frame_for_cache = {"OUTER": {}, "INNER": {}} 
        for ln_init in ["OUTER", "INNER"]:
            for _, _, p_name_init, s_name_init in global_config['LIPID_PAIRS_NMFK']:
                pair_n_init = f"{p_name_init}_vs_{s_name_init}"
                previous_window_consistent_labels_cache[ln_init][pair_n_init] = None
                last_processed_frame_for_cache[ln_init][pair_n_init] = -1 
        
        time_gap_threshold = global_config['WINDOW_SIZE_5_FRAME_NMFK'] 

        leaflet_specs_for_map_gen = [
            ("OUTER", ACTUAL_OUTER_LEAFLET_ID, global_config['NSIDE_NMFK_PHASE_MAP_OUTER']),
            ("INNER", ACTUAL_INNER_LEAFLET_ID, global_config['NSIDE_NMFK_PHASE_MAP_INNER'])
        ]
        
        block_size = global_config['WINDOW_SIZE_5_FRAME_NMFK']
        
        # New loop for defining blocks to avoid degenerate last block
        processed_frames_count = 0
        block_iteration_idx = 0 # For logging distinct blocks if needed

        while processed_frames_count < N_TOTAL_FRAMES:
            block_start_frame = processed_frames_count
            block_iteration_idx +=1

            # Determine the end of the current block
            potential_block_end_if_normal = block_start_frame + block_size
            block_end_exclusive = potential_block_end_if_normal

            if potential_block_end_if_normal < N_TOTAL_FRAMES:
                # If this block is not the last possible segment of frames
                frames_remaining_after_this_normal_block = N_TOTAL_FRAMES - potential_block_end_if_normal
                # If the "stub" remaining after this normal block is smaller than a full block size,
                # then extend the current block to include that stub.
                if 0 < frames_remaining_after_this_normal_block < block_size:
                    block_end_exclusive = N_TOTAL_FRAMES 
            else:
                # This block_start_frame is the beginning of the last segment of frames
                block_end_exclusive = N_TOTAL_FRAMES

            current_block_actual_frames = list(range(block_start_frame, block_end_exclusive))

            if not current_block_actual_frames: # Should not happen with this logic
                logger.warning(f"Empty block generated starting at frame {block_start_frame}, skipping.")
                processed_frames_count = block_end_exclusive # Ensure loop advances
                continue
            
            logger.info(f"  Processing NMFk for block #{block_iteration_idx} (nominal start: {block_start_frame}), "
                        f"actual frames: {current_block_actual_frames[0]}-{current_block_actual_frames[-1]} "
                        f"(Length: {len(current_block_actual_frames)})")

            # --- The rest of the NMFk processing for this block ---
            for leaflet_name, current_leaflet_id, current_nside_nmfk in leaflet_specs_for_map_gen:
                current_npix_nmfk = hp.nside2npix(current_nside_nmfk)
                
                for prim_sel_str, sec_sel_str, prim_name, sec_name in global_config['LIPID_PAIRS_NMFK']:
                    pair_name = f"{prim_name}_vs_{sec_name}"
                    # logger.debug(f"   Processing {leaflet_name} - {pair_name} for block starting {block_start_frame}") # Use block_iteration_idx for clarity

                    K_opt_for_NMF_components = optimal_K_trajectory.get(leaflet_name, {}).get(pair_name, 
                        min(global_config['NMFK_K_SEARCH_RANGE']) if global_config['NMFK_K_SEARCH_RANGE'] else 2)
                    
                    if isinstance(K_opt_for_NMF_components, str) and "N/A" in K_opt_for_NMF_components:
                         logger.warning(f"    Skipping {pair_name} ({leaflet_name}) for block starting {block_start_frame} due to N/A optimal K: {K_opt_for_NMF_components}")
                         for i_frame_in_block_to_nan in current_block_actual_frames:
                             if i_frame_in_block_to_nan < N_TOTAL_FRAMES : 
                                all_frames_nmfk_phase_maps[leaflet_name][pair_name][i_frame_in_block_to_nan, :] = np.full(current_npix_nmfk, np.nan)
                         continue

                    prev_labels_for_this_pair = None
                    cached_block_start_idx = last_processed_frame_for_cache[leaflet_name].get(pair_name, -1)
                    
                    if cached_block_start_idx != -1 and (block_start_frame - cached_block_start_idx) <= time_gap_threshold:
                        prev_labels_for_this_pair = previous_window_consistent_labels_cache[leaflet_name].get(pair_name)
                        if prev_labels_for_this_pair is not None: 
                            logger.debug(f"    LabelMatching: Using cache from block starting {cached_block_start_idx} for current block starting {block_start_frame} (Pair: {pair_name})")
                    else: 
                        logger.info(f"    LabelMatching: Cache from block starting {cached_block_start_idx} for {pair_name} ({leaflet_name}) is too old or missing for current block {block_start_frame}. Resetting for this block.")
                    
                    labels_for_lipids_in_block = get_nmfk_labels_for_block(
                        u, current_block_actual_frames, acl, 
                        ordered_resids_lipyphilic, current_leaflet_id, prim_sel_str, sec_sel_str, 
                        K_opt_for_NMF_components, global_config, logger,
                        prev_consistent_labels_for_pair=prev_labels_for_this_pair
                    )
                    
                    previous_window_consistent_labels_cache[leaflet_name][pair_name] = labels_for_lipids_in_block
                    last_processed_frame_for_cache[leaflet_name][pair_name] = block_start_frame 

                    for i_frame_in_block in current_block_actual_frames:
                        if i_frame_in_block >= N_TOTAL_FRAMES: continue 

                        if not labels_for_lipids_in_block:
                            # logger.debug(f"    No NMFk labels generated for block {block_start_frame} ({pair_name}, {leaflet_name}). Frame {i_frame_in_block} map will be NaN.")
                            all_frames_nmfk_phase_maps[leaflet_name][pair_name][i_frame_in_block, :] = np.full(current_npix_nmfk, np.nan)
                            continue

                        u.trajectory[i_frame_in_block] 
                        prim_coords_frame, prim_resids_frame, vesicle_com_frame = \
                            get_primary_lipid_coords_and_resids_in_leaflet_frame(
                                u, i_frame_in_block, acl, ordered_resids_lipyphilic, 
                                current_leaflet_id, prim_sel_str, global_config['HEAD_SEL'], logger
                            )

                        if prim_coords_frame is None or prim_coords_frame.shape[0] == 0:
                            # logger.debug(f"    No primary lipids ({prim_name}) coords for NMFk map gen. Leaflet: {leaflet_name}, Frame: {i_frame_in_block}.")
                            all_frames_nmfk_phase_maps[leaflet_name][pair_name][i_frame_in_block, :] = np.full(current_npix_nmfk, np.nan)
                            continue
                        
                        map_unfiltered = generate_map_mode_label(
                            prim_coords_frame, prim_resids_frame, labels_for_lipids_in_block, 
                            vesicle_com_frame, current_nside_nmfk, current_npix_nmfk
                        )
                        
                        frame_map_filtered = apply_healpix_mode_filter(map_unfiltered, current_nside_nmfk)
                        all_frames_nmfk_phase_maps[leaflet_name][pair_name][i_frame_in_block, :] = frame_map_filtered
            
            processed_frames_count = block_end_exclusive # Advance to the end of the processed block

        for leaflet_name_save, leaflet_data in all_frames_nmfk_phase_maps.items():
            for pair_name_save, data_array_save in leaflet_data.items():
                safe_pair_name = pair_name_save.replace(" ", "_")
                np.save(os.path.join(output_dir, f"nmfk_phase_map_{leaflet_name_save}_{safe_pair_name}.npy"), data_array_save)
        logger.info(f"NMFk phase maps (tumbling window method) generation time: {time.time() - nmfk_phase_map_gen_time_start:.2f}s")

        logger.info(f"Completed {folder_name}. Total time: {time.time() - total_time_start:.2f}s")
        return f"SUCCESS: Completed {folder_name}"

    except Exception as e:
        logger.error(f"Unhandled exception in {folder_name}: {e}", exc_info=True)
        main_error_logger.error(f"Unhandled exception: {e}", extra={'folder': folder_name}, exc_info=True)
        return f"FAILED_EXCEPTION: {folder_name} - {e}"
    finally: # (unchanged)
        if u is not None: del u
        if acl is not None: del acl
        if all_frames_radial_outer is not None: del all_frames_radial_outer
        if all_frames_radial_inner is not None: del all_frames_radial_inner
        if all_frames_nmfk_phase_maps is not None: del all_frames_nmfk_phase_maps
        gc.collect()
        logger.debug("Memory cleanup attempted.")


# --- Main Script Execution --- (unchanged from previous response)
def main():
    script_start_time = time.time()
    print(f"Main script started at {time.asctime(time.localtime(script_start_time))}")

    parser = argparse.ArgumentParser(description="Process MD simulation data for leaflet analysis and NMFk.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing configf_x folders.")
    parser.add_argument("--folder_list", type=str, default=None, help="Optional file containing specific folders to process.")
    parser.add_argument("--local_workers", type=int, default=None, help="Number of parallel workers.")
    args = parser.parse_args()

    if args.folder_list:
        with open(args.folder_list, 'r') as f:
            valid_config_folders = [line.strip() for line in f if os.path.isfile(os.path.join(line.strip(), "concat-traj.lammpstrj"))]
    else:
        current_directory = args.base_dir
        config_folders = sorted(glob.glob(os.path.join(current_directory, "configf_*")))
        config_folder_pattern = re.compile(r"configf_?\d+$")
        valid_config_folders = []
        for folder in config_folders:
            if config_folder_pattern.match(os.path.basename(folder)):
                traj_path = os.path.join(folder, "concat-traj.lammpstrj")
                if os.path.isfile(traj_path):
                    valid_config_folders.append(folder)

    if not valid_config_folders:
        print(f"No valid 'configf_x' folders found. Exiting.")
        main_error_logger.error(f"No valid 'configf_x' folders found.", extra={'folder': 'N/A'})
        return

    print(f"Found {len(valid_config_folders)} config folders to process.")

    # Determine number of workers
    num_pool_workers = 1

    if args.local_workers is not None and args.local_workers > 0:
        num_pool_workers = args.local_workers
        print(f"Using {num_pool_workers} workers (from --local_workers).")
    else:
        cpus_per_task_str = os.environ.get('SLURM_CPUS_PER_TASK')
        if cpus_per_task_str:
            try:
                cpus_per_task = int(cpus_per_task_str)
                if cpus_per_task > 0:
                    num_pool_workers = cpus_per_task
                    print(f"Using {num_pool_workers} workers (from SLURM_CPUS_PER_TASK).")
            except ValueError:
                print(f"Warning: Could not parse SLURM_CPUS_PER_TASK ('{cpus_per_task_str}'). Falling back.")
        
        if num_pool_workers == 1:
            try:
                cpu_count_val = os.cpu_count()
                if cpu_count_val and cpu_count_val > 0:
                    num_pool_workers = cpu_count_val
                    print(f"Using {num_pool_workers} workers (from os.cpu_count()).")
            except NotImplementedError:
                print("os.cpu_count() not implemented. Using 1 worker.")

    num_pool_workers = max(1, num_pool_workers)

    global_parameters = { 
        "TOPOLOGY_FILENAME": TOPOLOGY_FILENAME, "TRAJECTORY_FILENAME": TRAJECTORY_FILENAME,
        "SAT_SEL": SAT_SEL, "UNS_SEL": UNS_SEL, "CHOL_SEL": CHOL_SEL, "HEAD_SEL": HEAD_SEL,
        "LIPYPHILIC_LF_CUTOFF": LIPYPHILIC_LF_CUTOFF, "LIPYPHILIC_MIDPLANE_SEL": LIPYPHILIC_MIDPLANE_SEL,
        "LIPYPHILIC_MIDPLANE_CUTOFF": LIPYPHILIC_MIDPLANE_CUTOFF,
        "NSIDE_OUTER_RADIAL": NSIDE_OUTER_RADIAL, "NSIDE_INNER_RADIAL": NSIDE_INNER_RADIAL,
        "APPLY_ITERATIVE_MEDIAN_FILL_RADIAL": APPLY_ITERATIVE_MEDIAN_FILL_RADIAL,
        "MAX_ITERATIONS_FILL_RADIAL": MAX_ITERATIONS_FILL_RADIAL,
        "APPLY_GAUSSIAN_SMOOTHING_RADIAL": APPLY_GAUSSIAN_SMOOTHING_RADIAL,
        "GAUSSIAN_FWHM_DEG_RADIAL": GAUSSIAN_FWHM_DEG_RADIAL,
        "NSIDE_NMFK_PHASE_MAP_OUTER": NSIDE_NMFK_PHASE_MAP_OUTER,
        "NSIDE_NMFK_PHASE_MAP_INNER": NSIDE_NMFK_PHASE_MAP_INNER,
        "N_FRAMES_FOR_NMFK_TRAINING": N_FRAMES_FOR_NMFK_TRAINING,
        "K_CLUSTERS_FINAL_KMEANS": K_CLUSTERS_FINAL_KMEANS,
        "WINDOW_SIZE_5_FRAME_NMFK": WINDOW_SIZE_5_FRAME_NMFK, 
        "NMF_MAX_ITER_WINDOW_NMF": NMF_MAX_ITER_WINDOW_NMF,   
        "NMFK_N_ENSEMBLE": NMFK_N_ENSEMBLE, 
        "NMFK_N_RESTARTS_PER_ENSEMBLE": NMFK_N_RESTARTS_PER_ENSEMBLE,
        "NMFK_K_SEARCH_RANGE": NMFK_K_SEARCH_RANGE, 
        "NMFK_NOISE_LEVEL_STD_FACTOR": NMFK_NOISE_LEVEL_STD_FACTOR,
        "NMF_MAX_ITER_K_SEARCH": NMF_MAX_ITER_K_SEARCH, 
        "NMF_MAX_ITER_FINAL_NMF": NMF_MAX_ITER_FINAL_NMF, 
        "CONTACT_CUTOFF_PHASE_ANGSTROM": CONTACT_CUTOFF_PHASE_ANGSTROM,
        "LIPID_PAIRS_NMFK": LIPID_PAIRS_NMFK
    }

    starmap_args = [(folder_path, global_parameters) for folder_path in valid_config_folders]

    if num_pool_workers > 1 and len(valid_config_folders) > 1:
        actual_num_workers = min(num_pool_workers, len(valid_config_folders))
        print(f"Starting parallel processing of {len(valid_config_folders)} folders with {actual_num_workers} workers...")
        with multiprocessing.Pool(processes=actual_num_workers) as pool:
            results = pool.starmap(process_config_folder, starmap_args)
        print("\nParallel processing finished. Status per folder:")
        for i, res in enumerate(results): 
            print(f"- {os.path.basename(valid_config_folders[i])}: {res}")
    else:
        print("Starting sequential processing of folders...")
        results = []
        for folder_path_arg, global_params_arg in starmap_args:
            print(f"\n>>> Processing folder (sequentially): {os.path.basename(folder_path_arg)}")
            status = process_config_folder(folder_path_arg, global_params_arg)
            results.append(status)
            print(f"- {os.path.basename(folder_path_arg)}: {status}")

    script_end_time = time.time()
    print(f"\nMain script finished at {time.asctime(time.localtime(script_end_time))}")
    print(f"Total script runtime: {script_end_time - script_start_time:.2f} seconds.")

    for handler in main_error_logger.handlers[:]:
        handler.flush()
        handler.close()

if __name__ == "__main__":
    main()
