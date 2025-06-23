import os
import argparse
import glob
import numpy as np
import pandas as pd
import healpy as hp
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from functools import partial
import matplotlib.pyplot as plt
from scipy.special import softmax
from tensorflow.keras.saving import register_keras_serializable

from deepsphere import HealpyGCNN, healpy_layers as hp_layer

# --- Script Hyperparameters ---
K_PARAM = 5
LATENT_F_PARAM = 32
N_HEADS_PARAM = 8
TRANS_LAYERS_PARAM = 2
N_NEIGHBORS_PARAM = 20
POOL_TYPE_PARAM = "AVG"
BOTTLENECK_NSIDE = 2
ALPHA_FOCAL = 0.25
GAMMA_FOCAL = 2.0
SMOOTH_DICE = 1e-5
LAMBDA_GRAD_RADIAL = 0.1
HUBER_DELTA_RADIAL = 1.0
DEFAULT_EPOCHS = 130
DEFAULT_BATCH_SIZE = 32
DEFAULT_INITIAL_LR = 5e-5
SUBSET_FRACTION_DEBUG = None

# --- GPU Configuration ---
def setup_gpu(use_gpu_flag):
    if use_gpu_flag:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus}")
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}, proceeding with CPU.")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            print("No GPU found, proceeding with CPU.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Using CPU as per configuration.")
    print(f"GPUs visible to TF: {tf.config.list_physical_devices('GPU')}")

# --- Data Loading and Preprocessing ---
# Reverted to match original script structure as per user feedback
def fill_and_shift(cluster_map):
    out = np.empty_like(cluster_map, dtype=np.float32)
    mask_nan = np.isnan(cluster_map)
    out[mask_nan] = 0.0
    out[~mask_nan] = cluster_map[~mask_nan] + 1.0
    return out

def load_data_from_folders(config_dirs, leaflet_type, initial_nside):
    all_blocks = []
    target_npix = hp.nside2npix(initial_nside)
    for config_dir in config_dirs:
        path = os.path.join(config_dir, 'analysis_output_test_final')
        radial_file, chol_uns_file, sat_chol_file, uns_sat_file = "", "", "", "" 
        if leaflet_type == 'outer':
            radial_file = "area_outer_radial_dist.npy"
            chol_uns_file = "nmfk_phase_map_OUTER_CHOL_vs_UNS.npy"
            sat_chol_file = "nmfk_phase_map_OUTER_SAT_vs_CHOL.npy"
            uns_sat_file = "nmfk_phase_map_OUTER_UNS_vs_SAT.npy"
        elif leaflet_type == 'inner':
            radial_file = "area_inner_radial_dist.npy"
            chol_uns_file = "nmfk_phase_map_INNER_CHOL_vs_UNS.npy"
            sat_chol_file = "nmfk_phase_map_INNER_SAT_vs_CHOL.npy"
            uns_sat_file = "nmfk_phase_map_INNER_UNS_vs_SAT.npy"
        try:
            radial = np.load(os.path.join(path, radial_file))
            chol_uns = fill_and_shift(np.load(os.path.join(path, chol_uns_file)))
            sat_chol = fill_and_shift(np.load(os.path.join(path, sat_chol_file)))
            uns_sat = fill_and_shift(np.load(os.path.join(path, uns_sat_file)))
            
            all_blocks.append([[radial, uns_sat, sat_chol, chol_uns]])
        except FileNotFoundError as e:
            print(f"Warning: File not found in {path} for {config_dir}: {e}. Skipping.")
        except Exception as e:
            print(f"Warning: Error processing {config_dir}: {e}. Skipping.")

    try:
        outer_stacked = []
        if leaflet_type == 'outer':
            for block in all_blocks:
                block = np.stack(block, axis=-1)
                block = np.swapaxes(block, 0, 3)
                block = np.squeeze(block)
                outer_stacked.append(block)
        else: # Inner leaflet processing from original script
            for block in all_blocks:
                block = np.stack(block, axis=-1)
                block = np.swapaxes(block, 0, 3)
                block = np.squeeze(block)
                # Original script had: print(block.shape)
                # And no swapaxes/squeeze for inner, this implies 'block' structure or expectation is different.
                # To ensure it becomes (num_frames_this_block, npix, nfeatures) or (npix, nfeatures)
                # if block becomes e.g. (1, 1, Npix, 4) from np.stack([[r,u,s,c]], axis=-1)
                # it would need squeezing. The original print statement was key.
                # Assuming the result of np.stack is directly what's needed or concat handles it.
                # For safety, if it has excessive dims of 1:
                #if block.ndim == 4 and block.shape[0] == 1 and block.shape[2] ==1: # (1,X,1,Y) -> (X,Y)
                #    block = np.squeeze(block, axis=(0,2))
                #elif block.ndim == 3 and block.shape[0] == 1: # (1,X,Y) -> (X,Y)
                #    block = np.squeeze(block, axis=0)
                # If original `np.stack` for inner already yields (Npix, 4) or similar, no squeeze needed.
                outer_stacked.append(block)
        
        stacked = np.concatenate(outer_stacked, axis=0)
        print("before r2n",stacked.shape)
        # Assuming stacked is (N_total_frames, Npix, Nfeatures=4)
        for i in range(stacked.shape[0]): # Iterate over frames
            for j in range(stacked.shape[2]): # Iterate over features
                current_map = stacked[i, :, j]
                current_npix = current_map.shape[0]
                # current_nside = hp.npix2nside(current_npix) # Redundant check if target_npix is used
                if current_npix != target_npix:
                    print(f"Warning: Mismatch in npix for frame {i}, feature {j}. Expected {target_npix}, got {current_npix}. Skipping reorder.")
                    continue
                stacked[i, :, j] = hp.reorder(current_map, r2n=True)
        print("after r2n:",stacked.shape)
    except Exception as e:
        print(f"Warning: Error processing stacked data in {config_dir if 'config_dir' in locals() else 'some_dir'}: {e}.")
        # Return empty array or raise if critical
        if not all_blocks: raise ValueError("No data loaded.")
        return np.array([])


    if not all_blocks:
        raise ValueError("No data loaded. Check paths, leaflet type, and nside of input files.")
    if outer_stacked and stacked.size == 0 : # If outer_stacked had data but concat resulted in empty
        raise ValueError("Data processing resulted in an empty stacked array prior to reordering.")
    return stacked


def normalize_radial_channel(X_train, X_test):
    radial_train_data = X_train[..., 0]
    radial_mean = np.mean(radial_train_data)
    radial_std = np.std(radial_train_data)
    print(f"Calculated Radial Mean (from training data): {radial_mean}")
    print(f"Calculated Radial Std Dev (from training data): {radial_std}")
    if radial_std > 1e-7:
        X_train[..., 0] = (radial_train_data - radial_mean) / radial_std
        X_test[..., 0] = (X_test[..., 0] - radial_mean) / radial_std
    else:
        print("Warning: Radial std is very small. Applying mean centering only.")
        X_train[..., 0] = radial_train_data - radial_mean
        X_test[..., 0] = X_test[..., 0] - radial_mean
    print("Normalization of radial channel complete.")
    return X_train, X_test, radial_mean, radial_std

import numpy as np

def normalize_radial_channel_inner(X_train, X_test):
    # Extract radial channel (assumed to be at index 0)
    radial_train = X_train[..., 0]
    radial_test = X_test[..., 0]

    print(f"NaNs in X_train: {np.isnan(radial_train).sum()}, NaNs in X_test: {np.isnan(radial_test).sum()}")

    # Compute mean and std from non-NaN training values only
    radial_mean = np.nanmean(radial_train)
    radial_std = np.nanstd(radial_train)

    print(f"Calculated Radial Mean (from training data): {radial_mean}")
    print(f"Calculated Radial Std Dev (from training data): {radial_std}")

    # Step 1: Normalize
    if radial_std > 1e-7:
        X_train[..., 0] = (radial_train - radial_mean) / radial_std
        X_test[..., 0] = (radial_test - radial_mean) / radial_std
    else:
        print("Warning: Radial std is very small. Applying mean centering only.")
        X_train[..., 0] = radial_train - radial_mean
        X_test[..., 0] = radial_test - radial_mean

    # Step 2: Fill NaNs in normalized data with the value that would unnormalize to 0
    if radial_std > 1e-7:
        fill_value = -radial_mean / radial_std
    else:
        fill_value = -radial_mean  # mean-centering case

    X_train[..., 0] = np.nan_to_num(X_train[..., 0], nan=fill_value)
    X_test[..., 0] = np.nan_to_num(X_test[..., 0], nan=fill_value)

    if np.isnan(radial_train).sum() + np.isnan(radial_test).sum() > 0:
        print(f"Filled NaNs with value: {fill_value}")
    else:
        print("No NaNs found — no filling applied.")
    print("Normalization of radial channel complete.")
    return X_train, X_test, radial_mean, radial_std


def calculate_sample_weights(y_true_phase_map, num_classes=4):
    y_flat = y_true_phase_map.flatten()
    class_labels = np.arange(num_classes)
    present_labels = np.unique(y_flat)
    if len(present_labels) == 0: return np.ones_like(y_true_phase_map, dtype=np.float32)
    computed_class_weights = [1.0] if len(present_labels) == 1 else class_weight.compute_class_weight(
        class_weight='balanced', classes=present_labels, y=y_flat)
    class_weights_map = {label: weight for label, weight in zip(present_labels, computed_class_weights)}
    full_class_weights_map = {cls: class_weights_map.get(cls, 1.0) for cls in class_labels}
    return np.vectorize(full_class_weights_map.get)(y_true_phase_map).astype(np.float32)


# --- Model Definition (copied 1-for-1 from the notebooks) ------------
def HealpyUnpool(p, name=None):
    if p < 1:
        raise ValueError("Upsampling factor p must be >= 1")
    return tf.keras.layers.UpSampling1D(size=int(4**p), name=name)


@register_keras_serializable(package="DeepSphere_Outer")
class DeepSphereAE_OUTER(tf.keras.Model):
    def __init__(self, K_val, LATENT_F_val, N_HEADS_val, TRANS_LAYERS_val,
                 initial_nside, num_initial_features, num_output_radial=1, num_output_pm=4,
                 n_neighbors=20, pool_type="AVG", **kwargs):
        super().__init__(**kwargs) # Removed name="deep_sphere_ae"
        self.K                = K_val
        self.LATENT_F         = LATENT_F_val
        self.N_HEADS          = N_HEADS_val
        self.TRANS_LAYERS     = TRANS_LAYERS_val
        self.initial_nside    = initial_nside
        self.num_initial_features = num_initial_features              # ← store
        self.n_neighbors      = n_neighbors
        self.pool_type        = pool_type


        # --- Encoder ---
        current_nside_enc = initial_nside
        current_indices_enc = np.arange(hp.nside2npix(current_nside_enc))
        self.enc_conv1_gcnn = HealpyGCNN(current_nside_enc, current_indices_enc,
                                         [hp_layer.HealpyChebyshev(K=self.K, Fout=16, activation="elu")],
                                         n_neighbors=self.n_neighbors)
        self.pool1 = hp_layer.HealpyPool(p=1, pool_type=pool_type)
        current_nside_enc //= 2

        current_indices_enc = np.arange(hp.nside2npix(current_nside_enc))
        self.enc_conv2_gcnn = HealpyGCNN(current_nside_enc, current_indices_enc,
                                         [hp_layer.HealpyChebyshev(K=self.K, Fout=32, activation="elu")],
                                         n_neighbors=self.n_neighbors)
        self.pool2 = hp_layer.HealpyPool(p=1, pool_type=pool_type)
        current_nside_enc //= 2

        current_indices_enc = np.arange(hp.nside2npix(current_nside_enc))
        self.enc_conv3_gcnn = HealpyGCNN(current_nside_enc, current_indices_enc,
                                         [hp_layer.HealpyChebyshev(K=self.K, Fout=64, activation="elu")],
                                         n_neighbors=self.n_neighbors)
        self.pool3 = hp_layer.HealpyPool(p=1, pool_type=pool_type)
        current_nside_enc //= 2
        
        self.bottleneck_nside = current_nside_enc
        bottleneck_indices = np.arange(hp.nside2npix(self.bottleneck_nside))
        
        encoder_bottleneck_layers_list = [
            *[hp_layer.Healpy_ResidualLayer(
                layer_type="CHEBY",
                layer_kwargs={"K": self.K, "Fout": 64, "activation": "elu"},
                activation="elu", use_bn=True, norm_type="layer_norm")
              for _ in range(2)],
            hp_layer.HealpyChebyshev(K=self.K, Fout=self.LATENT_F, activation="linear")
        ]
        self.enc_bottleneck_gcnn = HealpyGCNN(self.bottleneck_nside, bottleneck_indices,
                                              encoder_bottleneck_layers_list, n_neighbors=self.n_neighbors)

        # --- Decoder ---
        decoder_start_layers_list = [
            hp_layer.Healpy_Transformer(
                num_heads=N_HEADS_val,
                key_dim=self.LATENT_F // N_HEADS_val,
                n_layers=TRANS_LAYERS_val,
                activation="elu"),
            *[hp_layer.Healpy_ResidualLayer(
                layer_type="CHEBY",
                layer_kwargs={"K": self.K, "Fout": 32, "activation": "elu"},
                activation="elu", use_bn=True, norm_type="layer_norm")
              for _ in range(2)]
        ]
        self.dec_start_gcnn = HealpyGCNN(self.bottleneck_nside, bottleneck_indices,
                                         decoder_start_layers_list, n_neighbors=self.n_neighbors)
        
        current_nside_dec = self.bottleneck_nside

        self.unpool1 = HealpyUnpool(p=1)
        current_nside_dec *= 2
        current_indices_dec = np.arange(hp.nside2npix(current_nside_dec))
        dec_block1_layers_list = [
            hp_layer.HealpyChebyshev(K=self.K, Fout=32, activation="elu"),
            hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": self.K, "Fout": 32, "activation": "elu"}, activation="elu", use_bn=True, norm_type="layer_norm")
        ]
        self.dec_block1_gcnn = HealpyGCNN(current_nside_dec, current_indices_dec, dec_block1_layers_list, n_neighbors=self.n_neighbors)

        self.unpool2 = HealpyUnpool(p=1)
        current_nside_dec *= 2
        current_indices_dec = np.arange(hp.nside2npix(current_nside_dec))
        dec_block2_layers_list = [
            hp_layer.HealpyChebyshev(K=self.K, Fout=32, activation="elu"),
            hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": self.K, "Fout": 32, "activation": "elu"}, activation="elu", use_bn=True, norm_type="layer_norm")
        ]
        self.dec_block2_gcnn = HealpyGCNN(current_nside_dec, current_indices_dec, dec_block2_layers_list, n_neighbors=self.n_neighbors)

        self.unpool3 = HealpyUnpool(p=1)
        current_nside_dec *= 2
        current_indices_dec = np.arange(hp.nside2npix(current_nside_dec))
        self.dec_final_conv_gcnn = HealpyGCNN(current_nside_dec, current_indices_dec,
                                              [hp_layer.HealpyChebyshev(K=self.K, Fout=32, activation="elu")],
                                              n_neighbors=self.n_neighbors)
        
        self.final_nside_for_heads = current_nside_dec
        self.final_indices_for_heads = current_indices_dec

        head_layer_spec = lambda Fout_head: [hp_layer.HealpyChebyshev(K=self.K, Fout=Fout_head, activation="linear")]
        
        self.rad_head = HealpyGCNN(self.final_nside_for_heads, self.final_indices_for_heads, head_layer_spec(num_output_radial), n_neighbors=self.n_neighbors)
        self.pm1_head = HealpyGCNN(self.final_nside_for_heads, self.final_indices_for_heads, head_layer_spec(num_output_pm), n_neighbors=self.n_neighbors)
        self.pm2_head = HealpyGCNN(self.final_nside_for_heads, self.final_indices_for_heads, head_layer_spec(num_output_pm), n_neighbors=self.n_neighbors)
        self.pm3_head = HealpyGCNN(self.final_nside_for_heads, self.final_indices_for_heads, head_layer_spec(num_output_pm), n_neighbors=self.n_neighbors)

        #    encoder_model / decoder_model) ─────────────────────────
        self._enc_model = None
        self._dec_model = None
    
    def get_config(self):
        base = super().get_config()
        cfg  = {
            "K_val":                self.K,
            "LATENT_F_val":         self.LATENT_F,
            "N_HEADS_val":          self.N_HEADS,
            "TRANS_LAYERS_val":     self.TRANS_LAYERS,
            "initial_nside":        self.initial_nside,
            "num_initial_features": self.num_initial_features,
            "n_neighbors":          self.n_neighbors,
            "pool_type":            self.pool_type,
        }
        return {**base, **cfg}
    # ──────────────────────────────────────────────────────────────
    #  1) pure-python helpers you can call directly
    # ──────────────────────────────────────────────────────────────
    def encode(self, x, training=False):
        """Run ONLY the encoder part and return the bottleneck tensor"""
        x = self.enc_conv1_gcnn(x, training=training);  x = self.pool1(x)
        x = self.enc_conv2_gcnn(x, training=training);  x = self.pool2(x)
        x = self.enc_conv3_gcnn(x, training=training);  x = self.pool3(x)
        z = self.enc_bottleneck_gcnn(x, training=training)
        return z                                 # (batch, Npix_bottleneck, LATENT_F)

    def decode(self, z, training=False):
        """Given a bottleneck tensor `z`, run ONLY the decoder & heads"""
        x = self.dec_start_gcnn(z, training=training)
        x = self.unpool1(x); x = self.dec_block1_gcnn(x, training=training)
        x = self.unpool2(x); x = self.dec_block2_gcnn(x, training=training)
        x = self.unpool3(x); x = self.dec_final_conv_gcnn(x, training=training)
        return {
            "radial"   : self.rad_head(x, training=training),
            "pm1_probs": self.pm1_head(x, training=training),
            "pm2_probs": self.pm2_head(x, training=training),
            "pm3_probs": self.pm3_head(x, training=training),
        }

    # ──────────────────────────────────────────────────────────────
    #  2) functional wrappers (lazy-built once, then cached)
    # ──────────────────────────────────────────────────────────────
    @property
    def encoder_model(self):
        if self._enc_model is None:
            dummy_npix = hp.nside2npix(self.initial_nside)
            inp  = tf.keras.Input(shape=(dummy_npix, self.num_initial_features))
            out  = self.encode(inp, training=False)
            self._enc_model = tf.keras.Model(inp, out, name=f"{self.name}_enc")
        return self._enc_model

    @property
    def decoder_model(self):
        if self._dec_model is None:
            dummy_npix_bot = hp.nside2npix(self.bottleneck_nside)
            inp  = tf.keras.Input(shape=(dummy_npix_bot, self.LATENT_F))
            out  = self.decode(inp, training=False)
            self._dec_model = tf.keras.Model(inp, out, name=f"{self.name}_dec")
        return self._dec_model

    # ──────────────────────────────────────────────────────────────
    #  3) the main forward pass stays unchanged but calls helpers
    # ──────────────────────────────────────────────────────────────
    def call(self, x, training=False):
        z    = self.encode(x, training=training)
        outs = self.decode(z, training=training)
        return outs

    def summary(self, *args, **kwargs):
        print_fn = kwargs.get('print_fn', print)

        print_fn(f"\n{'='*80}")
        print_fn(f"Overall Model: {self.name} (DeepSphereAE)")
        print_fn(f"{'='*80}")

        # Helper to print GCNN summaries and associated pool/unpool layers
        def print_stage_summary(stage_name, gcnn_instance, associated_layer=None, layer_type="Pool"):
            print_fn(f"\n--- {stage_name} ---")
            nside_info = f" (Operating at nside={gcnn_instance.nside})" if hasattr(gcnn_instance, 'nside') else ""
            print_fn(f"GCNN Block: {gcnn_instance.name}{nside_info}")
            gcnn_instance.summary(*args, **kwargs)
            if associated_layer:
                print_fn(f"  -> {layer_type} Layer: {associated_layer.name} (Type: {associated_layer.__class__.__name__})")

        print_fn("\n-------------------------- Encoder Stages -------------------------------")
        print_stage_summary("Encoder Stage 1", self.enc_conv1_gcnn, self.pool1, layer_type="Pool")
        print_stage_summary("Encoder Stage 2", self.enc_conv2_gcnn, self.pool2, layer_type="Pool")
        print_stage_summary("Encoder Stage 3", self.enc_conv3_gcnn, self.pool3, layer_type="Pool")
        print_stage_summary("Encoder Bottleneck", self.enc_bottleneck_gcnn)
        
        print_fn("\n-------------------------- Decoder Stages -------------------------------")
        print_stage_summary("Decoder Start (Bottleneck)", self.dec_start_gcnn) # Includes Transformer
        print_stage_summary("Decoder Upsampling Block 1", self.dec_block1_gcnn, self.unpool1, layer_type="Unpool (before GCNN)")
        print_stage_summary("Decoder Upsampling Block 2", self.dec_block2_gcnn, self.unpool2, layer_type="Unpool (before GCNN)")
        print_stage_summary("Decoder Upsampling Block 3 / Final Conv", self.dec_final_conv_gcnn, self.unpool3, layer_type="Unpool (before GCNN)")

        print_fn("\n-------------------------- Output Heads ---------------------------------")
        nside_heads = self.rad_head.nside if hasattr(self.rad_head, 'nside') else 'N/A'
        print_fn(f"(All heads operate on nside={nside_heads} features from final decoder stage)")
        print_stage_summary("Radial Head", self.rad_head)
        print_stage_summary("PM1 Head", self.pm1_head)
        print_stage_summary("PM2 Head", self.pm2_head)
        print_stage_summary("PM3 Head", self.pm3_head)
        print_fn(f"{'='*80}")

    def build(self, input_shape):
        """
        Force-materialise all sub-layers **before** weight loading happens.
        Keras calls build() automatically during `load_model()`.
        """
        batch, npix, nfeat = input_shape
        # run a single forward pass through *both* halves
        z_dummy   = self.encode(tf.zeros((1, npix, nfeat)), training=False)
        _ = self.decode(z_dummy,                  training=False)
        super().build(input_shape)           # mark the model as built



@register_keras_serializable(package="DeepSphere_Inner")
class DeepSphereAE_INNER(tf.keras.Model):
    def __init__(self, K_val, LATENT_F_val, N_HEADS_val, TRANS_LAYERS_val,
                 initial_nside, num_initial_features,  # ← already present
                 num_output_radial=1, num_output_pm=4,
                 n_neighbors=20, pool_type="AVG", **kwargs):      
        super().__init__(**kwargs)                                     # ← pass **kwargs
        self.K                = K_val
        self.LATENT_F         = LATENT_F_val
        self.N_HEADS          = N_HEADS_val
        self.TRANS_LAYERS     = TRANS_LAYERS_val
        self.initial_nside    = initial_nside
        self.num_initial_features = num_initial_features              # ← store
        self.n_neighbors      = n_neighbors
        self.pool_type        = pool_type

        # --- Encoder ---
        current_nside_enc = initial_nside
        current_indices_enc = np.arange(hp.nside2npix(current_nside_enc))
        self.enc_conv1_gcnn = HealpyGCNN(current_nside_enc, current_indices_enc,
                                         [hp_layer.HealpyChebyshev(K=self.K, Fout=16, activation="elu")],
                                         n_neighbors=self.n_neighbors)
        self.pool1 = hp_layer.HealpyPool(p=1, pool_type=pool_type)
        current_nside_enc //= 2

        current_indices_enc = np.arange(hp.nside2npix(current_nside_enc))
        self.enc_conv2_gcnn = HealpyGCNN(current_nside_enc, current_indices_enc,
                                         [hp_layer.HealpyChebyshev(K=self.K, Fout=32, activation="elu")],
                                         n_neighbors=self.n_neighbors)
        self.pool2 = hp_layer.HealpyPool(p=1, pool_type=pool_type)
        current_nside_enc //= 2

        current_indices_enc = np.arange(hp.nside2npix(current_nside_enc))
        self.bottleneck_nside = current_nside_enc
        bottleneck_indices = np.arange(hp.nside2npix(self.bottleneck_nside))
        
        encoder_bottleneck_layers_list = [
            *[hp_layer.Healpy_ResidualLayer(
                layer_type="CHEBY",
                layer_kwargs={"K": self.K, "Fout": 32, "activation": "elu"},
                activation="elu", use_bn=True, norm_type="layer_norm")
              for _ in range(2)],
            hp_layer.HealpyChebyshev(K=self.K, Fout=self.LATENT_F, activation="linear")
        ]
        self.enc_bottleneck_gcnn = HealpyGCNN(self.bottleneck_nside, bottleneck_indices,
                                              encoder_bottleneck_layers_list, n_neighbors=self.n_neighbors)

        # --- Decoder ---
        decoder_start_layers_list = [
            hp_layer.Healpy_Transformer(
                num_heads=N_HEADS_val,
                key_dim=self.LATENT_F // N_HEADS_val,
                n_layers=TRANS_LAYERS_val,
                activation="elu"),
            *[hp_layer.Healpy_ResidualLayer(
                layer_type="CHEBY",
                layer_kwargs={"K": self.K, "Fout": 32, "activation": "elu"},
                activation="elu", use_bn=True, norm_type="layer_norm")
              for _ in range(2)]
        ]
        self.dec_start_gcnn = HealpyGCNN(self.bottleneck_nside, bottleneck_indices,
                                         decoder_start_layers_list, n_neighbors=self.n_neighbors)
        
        current_nside_dec = self.bottleneck_nside

        self.unpool1 = HealpyUnpool(p=1)
        current_nside_dec *= 2
        current_indices_dec = np.arange(hp.nside2npix(current_nside_dec))
        dec_block1_layers_list = [
            hp_layer.HealpyChebyshev(K=self.K, Fout=32, activation="elu"),
            hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": self.K, "Fout": 32, "activation": "elu"}, activation="elu", use_bn=True, norm_type="layer_norm")
        ]
        self.dec_block1_gcnn = HealpyGCNN(current_nside_dec, current_indices_dec, dec_block1_layers_list, n_neighbors=self.n_neighbors)

        self.unpool2 = HealpyUnpool(p=1)
        current_nside_dec *= 2
        current_indices_dec = np.arange(hp.nside2npix(current_nside_dec))
        dec_block2_layers_list = [
            hp_layer.HealpyChebyshev(K=self.K, Fout=32, activation="elu"),
            hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": self.K, "Fout": 32, "activation": "elu"}, activation="elu", use_bn=True, norm_type="layer_norm")
        ]
        self.dec_block2_gcnn = HealpyGCNN(current_nside_dec, current_indices_dec, dec_block2_layers_list, n_neighbors=self.n_neighbors)

        
        self.final_nside_for_heads = current_nside_dec
        self.final_indices_for_heads = current_indices_dec

        head_layer_spec = lambda Fout_head: [hp_layer.HealpyChebyshev(K=self.K, Fout=Fout_head, activation="linear")]
        
        self.rad_head = HealpyGCNN(self.final_nside_for_heads, self.final_indices_for_heads, head_layer_spec(num_output_radial), n_neighbors=self.n_neighbors)
        self.pm1_head = HealpyGCNN(self.final_nside_for_heads, self.final_indices_for_heads, head_layer_spec(num_output_pm), n_neighbors=self.n_neighbors)
        self.pm2_head = HealpyGCNN(self.final_nside_for_heads, self.final_indices_for_heads, head_layer_spec(num_output_pm), n_neighbors=self.n_neighbors)
        self.pm3_head = HealpyGCNN(self.final_nside_for_heads, self.final_indices_for_heads, head_layer_spec(num_output_pm), n_neighbors=self.n_neighbors)

        # caches for functional wrappers
        self._enc_model = None
        self._dec_model = None

    # ───────────────────────────── helpers ──────────────────────────────
    def encode(self, x, training=False):
        x = self.enc_conv1_gcnn(x, training=training); x = self.pool1(x)
        x = self.enc_conv2_gcnn(x, training=training); x = self.pool2(x)
        z = self.enc_bottleneck_gcnn(x, training=training)
        return z                                                     # (B, N_bot, LATENT_F)

    def decode(self, z, training=False):
        x = self.dec_start_gcnn(z, training=training)
        x = self.unpool1(x); x = self.dec_block1_gcnn(x, training=training)
        x = self.unpool2(x); x = self.dec_block2_gcnn(x, training=training)
        return {
            "radial"   : self.rad_head(x, training=training),
            "pm1_probs": self.pm1_head(x, training=training),
            "pm2_probs": self.pm2_head(x, training=training),
            "pm3_probs": self.pm3_head(x, training=training),
        }
    
    @property
    def encoder_model(self):
        if self._enc_model is None:
            npix = hp.nside2npix(self.initial_nside)
            inp  = tf.keras.Input(shape=(npix, self.num_initial_features))
            out  = self.encode(inp, training=False)
            self._enc_model = tf.keras.Model(inp, out, name=f"{self.name}_enc")
        return self._enc_model

    @property
    def decoder_model(self):
        if self._dec_model is None:
            npix_bot = hp.nside2npix(self.bottleneck_nside)
            inp  = tf.keras.Input(shape=(npix_bot, self.LATENT_F))
            out  = self.decode(inp, training=False)
            self._dec_model = tf.keras.Model(inp, out, name=f"{self.name}_dec")
        return self._dec_model
    
    # main forward pass (unchanged except calling helpers)
    def call(self, x, training=False):
        z    = self.encode(x, training=training)
        outs = self.decode(z, training=training)
        return outs

    def get_config(self):
        base = super().get_config()
        cfg  = {
            "K_val":                self.K,
            "LATENT_F_val":         self.LATENT_F,
            "N_HEADS_val":          self.N_HEADS,
            "TRANS_LAYERS_val":     self.TRANS_LAYERS,
            "initial_nside":        self.initial_nside,
            "num_initial_features": self.num_initial_features,
            "n_neighbors":          self.n_neighbors,
            "pool_type":            self.pool_type,
        }
        return {**base, **cfg}

    def summary(self, *args, **kwargs):
        print_fn = kwargs.get('print_fn', print)

        print_fn(f"\n{'='*80}")
        print_fn(f"Overall Model: {self.name} (DeepSphereAE)")
        print_fn(f"{'='*80}")

        # Helper to print GCNN summaries and associated pool/unpool layers
        def print_stage_summary(stage_name, gcnn_instance, associated_layer=None, layer_type="Pool"):
            print_fn(f"\n--- {stage_name} ---")
            nside_info = f" (Operating at nside={gcnn_instance.nside})" if hasattr(gcnn_instance, 'nside') else ""
            print_fn(f"GCNN Block: {gcnn_instance.name}{nside_info}")
            gcnn_instance.summary(*args, **kwargs)
            if associated_layer:
                print_fn(f"  -> {layer_type} Layer: {associated_layer.name} (Type: {associated_layer.__class__.__name__})")

        print_fn("\n-------------------------- Encoder Stages -------------------------------")
        print_stage_summary("Encoder Stage 1", self.enc_conv1_gcnn, self.pool1, layer_type="Pool")
        print_stage_summary("Encoder Stage 2", self.enc_conv2_gcnn, self.pool2, layer_type="Pool")
        #print_stage_summary("Encoder Stage 3", self.enc_conv3_gcnn, self.pool3, layer_type="Pool")
        print_stage_summary("Encoder Bottleneck", self.enc_bottleneck_gcnn)
        
        print_fn("\n-------------------------- Decoder Stages -------------------------------")
        print_stage_summary("Decoder Start (Bottleneck)", self.dec_start_gcnn) # Includes Transformer
        print_stage_summary("Decoder Upsampling Block 1", self.dec_block1_gcnn, self.unpool1, layer_type="Unpool (before GCNN)")
        print_stage_summary("Decoder Upsampling Block 2", self.dec_block2_gcnn, self.unpool2, layer_type="Unpool (before GCNN)")
        #print_stage_summary("Decoder Upsampling Block 3 / Final Conv", self.dec_final_conv_gcnn, self.unpool3, layer_type="Unpool (before GCNN)")

        print_fn("\n-------------------------- Output Heads ---------------------------------")
        nside_heads = self.rad_head.nside if hasattr(self.rad_head, 'nside') else 'N/A'
        print_fn(f"(All heads operate on nside={nside_heads} features from final decoder stage)")
        print_stage_summary("Radial Head", self.rad_head)
        print_stage_summary("PM1 Head", self.pm1_head)
        print_stage_summary("PM2 Head", self.pm2_head)
        print_stage_summary("PM3 Head", self.pm3_head)
        print_fn(f"{'='*80}")

    def build(self, input_shape):
        """
        Force-materialise all sub-layers **before** weight loading happens.
        Keras calls build() automatically during `load_model()`.
        """
        batch, npix, nfeat = input_shape
        # run a single forward pass through *both* halves
        z_dummy   = self.encode(tf.zeros((1, npix, nfeat)), training=False)
        _ = self.decode(z_dummy,                  training=False)
        super().build(input_shape)           # mark the model as built


# --- Loss Functions and Metrics ---
NB_CONST_DYNAMIC = None
def sparse_focal_loss(y_true, y_pred, g=GAMMA_FOCAL, a=ALPHA_FOCAL):
    y_t, d = tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1]
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_t, y_pred, from_logits=True)
    p_t = tf.reduce_sum(tf.one_hot(y_t,d)*tf.nn.softmax(y_pred,-1),-1)
    focal = a*tf.pow(1.-p_t,g)*ce
    return focal

def multiclass_dice_loss(y_true, y_pred, s=SMOOTH_DICE):
    y_t, d = tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1] # y_pred (batch, Npix, Nclass), y_true (batch, Npix)
    y_oh = tf.one_hot(y_t,d) # (batch, Npix, Nclass)
    p_prob = tf.nn.softmax(y_pred,-1) # (batch, Npix, Nclass)
    inter = tf.reduce_sum(y_oh*p_prob, axis=[1,2]) # Sum over Npix and Nclass -> (batch,)
    union = tf.reduce_sum(y_oh,axis=[1,2])+tf.reduce_sum(p_prob,axis=[1,2]) # Sum over Npix and Nclass -> (batch,)
    return 1. - tf.reduce_mean((2.*inter+s)/(union+s)) # Mean over batch

def combined_seg_loss(y_true,y_pred):
    y_true_sq = tf.squeeze(y_true, -1) if len(y_true.shape)==3 and y_true.shape[-1]==1 else y_true
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_sq,y_pred,from_logits=True) + \
           sparse_focal_loss(y_true_sq,y_pred) + multiclass_dice_loss(y_true_sq,y_pred)
    return loss

def healpy_grad_magnitude_tf(map_vals): 
    global NB_CONST_DYNAMIC
    if NB_CONST_DYNAMIC is None: raise ValueError("NB_CONST_DYNAMIC not initialized.")
    #map_vals_squeezed = tf.squeeze(map_vals, axis=-1) if len(map_vals.shape) == 3 and map_vals.shape[-1] == 1 else map_vals
    #gathered_neighbors = tf.gather(map_vals_squeezed, NB_CONST_DYNAMIC, axis=1) 
    #diffs = tf.abs(gathered_neighbors - map_vals_squeezed[:,:,tf.newaxis])
    #return tf.reduce_mean(diffs,axis=[1,2])
    nb_vals = tf.gather(map_vals, NB_CONST_DYNAMIC, axis=1)
    diffs   = tf.abs(nb_vals - map_vals[..., tf.newaxis, :])
    return tf.reduce_mean(diffs, axis=(2, 3))     # -> (batch, N_PIX)

def radial_loss(y_true,y_pred):
    base = tf.keras.losses.Huber(delta=HUBER_DELTA_RADIAL)(y_true,y_pred)
    grad_true = healpy_grad_magnitude_tf(y_true)
    grad_pred = healpy_grad_magnitude_tf(y_pred)
    grad_term = tf.reduce_mean(tf.abs(grad_true - grad_pred))
    return base + LAMBDA_GRAD_RADIAL * grad_term

class CustomMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='custom_mean_iou', dtype=None, **kwargs):
        super(CustomMeanIoU, self).__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        # Initialize a variable to store the total confusion matrix.
        self.total_cm = self.add_weight(
            name='total_confusion_matrix_var', # Slightly changed name just in case of obscure collisions
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.int64 
        )
        # For debugging, you can add this print statement:
        # print(f"Metric {self.name}: Initialized self.total_cm type: {type(self.total_cm)}")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # (Your existing update_state logic a_s_is...)
        # ...
        # predicted_labels = tf.argmax(y_pred, axis=-1)
        # y_true_flat = tf.reshape(tf.cast(y_true, tf.int64), [-1])
        # predicted_labels_flat = tf.reshape(tf.cast(predicted_labels, tf.int64), [-1])
        # current_weights = None
        # if sample_weight is not None:
        #     current_weights = tf.reshape(tf.cast(sample_weight, tf.float32), [-1])
        #
        # batch_cm = tf.math.confusion_matrix(
        #     labels=y_true_flat,
        #     predictions=predicted_labels_flat,
        #     num_classes=self.num_classes,
        #     weights=current_weights,
        #     dtype=tf.int64
        # )
        # self.total_cm.assign_add(batch_cm)
        # (End of existing update_state logic)
        
        # --- Make sure your update_state correctly calculates and adds to self.total_cm ---
        # The following is a simplified placeholder for the actual calculation from the previous answer
        predicted_labels = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, dtype=tf.int64) # Ensure y_true is int64 for confusion_matrix
        predicted_labels = tf.cast(predicted_labels, dtype=tf.int64)

        y_true_flat = tf.reshape(y_true, [-1])
        predicted_labels_flat = tf.reshape(predicted_labels, [-1])
        
        current_weights_flat = None
        if sample_weight is not None:
            # Assuming sample_weight has same rank as y_true and can be flattened
            current_weights_flat = tf.reshape(tf.cast(sample_weight, dtype=tf.float32), [-1])

        batch_cm = tf.math.confusion_matrix(
            labels=y_true_flat,
            predictions=predicted_labels_flat,
            num_classes=self.num_classes,
            weights=current_weights_flat,
            dtype=tf.int64
        )
        self.total_cm.assign_add(batch_cm)


    def result(self):
        # (Your existing result logic as is...)
        # ...
        # sum_over_row = tf.cast(tf.reduce_sum(self.total_cm, axis=0), dtype=tf.float32)
        # sum_over_col = tf.cast(tf.reduce_sum(self.total_cm, axis=1), dtype=tf.float32)
        # true_positives = tf.cast(tf.linalg.diag_part(self.total_cm), dtype=tf.float32)
        # denominator = sum_over_row + sum_over_col - true_positives
        # iou_per_class = tf.math.divide_no_nan(true_positives, denominator)
        # is_present = tf.cast(tf.math.not_equal(denominator, 0), dtype=tf.float32)
        # num_present_classes = tf.reduce_sum(is_present)
        # sum_iou_present_classes = tf.reduce_sum(iou_per_class * is_present)
        # mean_iou = tf.math.divide_no_nan(sum_iou_present_classes, num_present_classes)
        # return mean_iou
        # (End of existing result logic)

        # --- Simplified placeholder for result logic from previous answer ---
        sum_over_row = tf.cast(tf.reduce_sum(self.total_cm, axis=0), dtype=tf.float32)
        sum_over_col = tf.cast(tf.reduce_sum(self.total_cm, axis=1), dtype=tf.float32)
        true_positives = tf.cast(tf.linalg.diag_part(self.total_cm), dtype=tf.float32)

        denominator = sum_over_row + sum_over_col - true_positives
        iou_per_class = tf.math.divide_no_nan(true_positives, denominator)
        
        is_present = tf.cast(tf.math.not_equal(denominator, 0), dtype=tf.float32)
        num_present_classes = tf.reduce_sum(is_present)
        
        sum_iou_present_classes = tf.reduce_sum(iou_per_class * is_present)
        mean_iou = tf.math.divide_no_nan(sum_iou_present_classes, num_present_classes)
        return mean_iou


    def reset_state(self):
        # For debugging, you can add these print statements:
        # print(f"Metric {self.name}: In reset_state. Current self.total_cm type: {type(self.total_cm)}")
        # if isinstance(self.total_cm, str):
        # print(f"Metric {self.name}: self.total_cm is a string with value: '{self.total_cm}'")

        # Use the .assign() method of the variable
        self.total_cm.assign(tf.zeros(self.total_cm.shape, dtype=self.total_cm.dtype))

def patch_precision(y_t,y_p,patch_size=8): 
    y_t_sq=tf.squeeze(y_t,-1) if len(y_t.shape)==3 and y_t.shape[-1]==1 else y_t 
    corr=tf.cast(tf.equal(tf.cast(y_t_sq, tf.int64),tf.argmax(y_p,-1,output_type=tf.int64)),tf.float32) 
    kern=tf.ones((patch_size,1,1),tf.float32)/float(patch_size) 
    corr_p=tf.nn.conv1d(corr[...,tf.newaxis],kern,1,"SAME") 
    return tf.reduce_mean(tf.cast(corr_p > 0.5,tf.float32))

# --- Plotting and Evaluation ---
def plot_training_history(history_data, out_dir, leaflet_type): # Added leaflet_type for filenames
    heads=['radial','pm1_probs','pm2_probs','pm3_probs']; plt.figure(figsize=(12,10))
    for i,h in enumerate(heads):
        plt.subplot(2,2,i+1); 
        plt.plot(history_data.get(f'{h}_loss',[]),label='tr'); 
        plt.plot(history_data.get(f'val_{h}_loss',[]),label='val')
        plt.title(f'{h} loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        if h=='radial': plt.ylim(0,max(2, np.max(history_data.get(f'val_{h}_loss',[2])) if history_data.get(f'val_{h}_loss') else 2))
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"{leaflet_type}_losses.png")); plt.close()
    
    plt.figure(figsize=(12,10))
    for i,h in enumerate(heads):
        plt.subplot(2,2,i+1); 
        m_label_display = 'MAE' if h == 'radial' else 'Accuracy' # For title/ylabel
        # Keras history keys for standard metrics
        metric_key_main = f'{h}_mae' if h == 'radial' else f'{h}_accuracy' 
        metric_key_val = f'val_{h}_mae' if h == 'radial' else f'val_{h}_accuracy'
        
        # Check for 'acc' as an alternative for accuracy if 'accuracy' not found
        if h != 'radial' and metric_key_main not in history_data and f'{h}_acc' in history_data:
            metric_key_main = f'{h}_acc'
        if h != 'radial' and metric_key_val not in history_data and f'val_{h}_acc' in history_data:
            metric_key_val = f'val_{h}_acc'

        if history_data.get(metric_key_main) and history_data.get(metric_key_val):
            plt.plot(history_data[metric_key_main],label=f'tr_{m_label_display}'); 
            plt.plot(history_data[metric_key_val],label=f'val_{m_label_display}')
            plt.title(f'{h} {m_label_display}'); plt.xlabel('Epoch'); plt.ylabel(m_label_display); plt.legend(); plt.grid(True)
        elif history_data.get(f'{h}_loss'):
             plt.text(0.5, 0.5, f'Metrics for {h} not found\n(tried {m_label_display.lower()})', ha='center', va='center', transform=plt.gca().transAxes)
             print(f"Warning: Metric keys for {h} (e.g., {metric_key_main}, {metric_key_val}) not found for plotting.")
        else:
             plt.text(0.5, 0.5, f'No data for {h}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"{leaflet_type}_metrics.png")); plt.close()


def iou_from_cm(cm):tp=np.diag(cm);den=tp+cm.sum(0)-tp+cm.sum(1)-tp; return np.divide(tp,den,out=np.zeros_like(tp,float),where=den!=0)

def eval_and_save(model, X_eval, y_eval_dict, out_dir, r_mean, r_std, leaflet_type): # Added leaflet_type
    print("\n--- Evaluating model on test set ---")
    pred_logits = model.predict(X_eval, batch_size=DEFAULT_BATCH_SIZE); report = ["Metrics on Test Set\n"+"="*30+"\n"]
    r_pred_n, y_r_test_n = pred_logits["radial"], y_eval_dict["radial"]
    report.extend([f"Radial MAE (norm): {np.mean(np.abs(r_pred_n-y_r_test_n)):.4f}",
                   f"Radial MSE (norm): {np.mean(np.square(r_pred_n-y_r_test_n)):.4f}",
                   f"Radial MAE (denorm): {np.mean(np.abs((r_pred_n*r_std+r_mean)-(y_r_test_n*r_std+r_mean))):.4f}\n"])
    true_lbls={"p1":y_eval_dict["pm1_probs"].ravel(),"p2":y_eval_dict["pm2_probs"].ravel(),"p3":y_eval_dict["pm3_probs"].ravel()}
    pred_lbls={"p1":np.argmax(pred_logits["pm1_probs"],-1).ravel(),"p2":np.argmax(pred_logits["pm2_probs"],-1).ravel(),"p3":np.argmax(pred_logits["pm3_probs"],-1).ravel()}
    cl_range=np.arange(4)
    for name in ["p1","p2","p3"]:
        y_t,y_p=true_lbls[name],pred_lbls[name]; report.append(f"\n=== {name.upper()} RESULTS ===")
        cm=confusion_matrix(y_t,y_p,labels=cl_range); report.append("CM (GT/PRED):\n"+np.array2string(cm))
        try:
            cr_dict=classification_report(y_t,y_p,labels=cl_range,digits=3,output_dict=True,zero_division=0)
            report.append("\nPrec/Rec/F1:"); [report.append(f" C{idx}: P={cr_dict[str(idx)]['precision']:.3f} R={cr_dict[str(idx)]['recall']:.3f} F1={cr_dict[str(idx)]['f1-score']:.3f} (S:{cr_dict[str(idx)]['support']})") for idx in cl_range if str(idx) in cr_dict]
            report.extend([f"Macro F1: {cr_dict['macro avg']['f1-score']:.3f}", f"Weighted F1: {cr_dict['weighted avg']['f1-score']:.3f}"])
        except Exception as e: report.append(f"\nCannot gen class report for {name}: {e}")
        iou=iou_from_cm(cm); report.append("\nIoU/Class:"); [report.append(f" C{idx}: {val:.3f}") for idx,val in enumerate(iou)]
        report.append(f"Mean IoU: {iou.mean():.3f}\nOverall Acc: {accuracy_score(y_t,y_p):.3f}\n")
    with open(os.path.join(out_dir,f"{leaflet_type}_perf_metrics.txt"),"w") as f: f.write("\n".join(report)) # Prefixed
    print(f"Perf metrics saved to {os.path.join(out_dir,f'{leaflet_type}_perf_metrics.txt')}\n" + "\n".join(report[-15:]))

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSphere Autoencoder.")
    parser.add_argument("--leaflet", type=str, required=True, choices=['inner', 'outer'])
    parser.add_argument("--data_folders_file", type=str, required=True)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--output_dir", type=str, default="training_output_inner")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--initial_learning_rate", type=float, default=DEFAULT_INITIAL_LR)
    parser.add_argument("--refit_on_all_data", action='store_true', help="Refit final model on all data after initial train/test.")
    args = parser.parse_args()

    leaflet_type=args.leaflet

    os.makedirs(args.output_dir, exist_ok=True); setup_gpu(args.use_gpu)
    initial_nside = 16 if args.leaflet == 'outer' else 8
    num_features = 4
    print(f"Leaflet: {args.leaflet}, Initial NSIDE: {initial_nside}")

    with open(args.data_folders_file, 'r') as f: all_config_dirs = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    if not all_config_dirs: raise ValueError(f"No data folders in {args.data_folders_file}")
    print(f"Found {len(all_config_dirs)} config dirs.")

    train_dirs, test_dirs = train_test_split(all_config_dirs, test_size=0.15, random_state=42, shuffle=True)
    print(f"Split: {len(train_dirs)} train sequences, {len(test_dirs)} test sequences.")

    
    # save into leaflet‐specific lists
    if args.leaflet == "outer":
        outer_train_folders = train_dirs
        outer_test_folders  = test_dirs
    else:  # "inner"
        inner_train_folders = train_dirs
        inner_test_folders  = test_dirs

    # optionally write them out to disk
    train_list_path = os.path.join(args.output_dir, f"{args.leaflet}_train_folders.txt")
    test_list_path  = os.path.join(args.output_dir, f"{args.leaflet}_test_folders.txt")
    with open(train_list_path, "w") as f:
        f.write("\n".join(train_dirs))
    with open(test_list_path, "w") as f:
        f.write("\n".join(test_dirs))

    print(f"Saved train list to {train_list_path}")
    print(f"Saved test list  to {test_list_path}")

    X_train_raw = load_data_from_folders(train_dirs, args.leaflet, initial_nside)
    X_test_raw = load_data_from_folders(test_dirs, args.leaflet, initial_nside)
    if X_train_raw.size == 0 or X_test_raw.size == 0:
        raise ValueError("Data loading returned empty arrays. Check load_data_from_folders function and paths.")
    print(f"Raw X_train: {X_train_raw.shape}, X_test: {X_test_raw.shape}")

    X_train, X_test = (X_train_raw, X_test_raw) 
    if SUBSET_FRACTION_DEBUG: 
        X_train = X_train_raw[:max(1,int(len(X_train_raw)*SUBSET_FRACTION_DEBUG))]
        X_test = X_test_raw[:max(1,int(len(X_test_raw)*SUBSET_FRACTION_DEBUG))]
    print(f"Using X_train: {X_train.shape}, X_test: {X_test.shape}")

    if leaflet_type =='outer':
        X_train_n, X_test_n, r_mean, r_std = normalize_radial_channel(X_train.copy(), X_test.copy())
    elif leaflet_type == 'inner':
        X_train_n, X_test_n, r_mean, r_std = normalize_radial_channel_inner(X_train.copy(), X_test.copy())
    else:
        print("error, no radial normalization done due to no detected leaflet.")
    np.save(os.path.join(args.output_dir,f"{args.leaflet}_X_train_norm.npy"),X_train_n) # Prefixed
    np.save(os.path.join(args.output_dir,f"{args.leaflet}_X_test_norm.npy"),X_test_n) # Prefixed
    print(f"Saved normalized X_train/X_test to {args.output_dir}")

    norm_params_path = os.path.join(args.output_dir, f"{args.leaflet}_radial_normalization_params.npz") # Prefixed
    np.savez(norm_params_path, mean=r_mean, std=r_std)
    print(f"Saved radial channel normalization parameters (mean, std) to {norm_params_path}")
    print(X_train_n.shape)
    y_tr_dict = {'radial':X_train_n[...,0:1],'pm1_probs':X_train_n[...,1].astype(np.int32),'pm2_probs':X_train_n[...,2].astype(np.int32),'pm3_probs':X_train_n[...,3].astype(np.int32)}
    y_te_dict = {'radial':X_test_n[...,0:1],'pm1_probs':X_test_n[...,1].astype(np.int32),'pm2_probs':X_test_n[...,2].astype(np.int32),'pm3_probs':X_test_n[...,3].astype(np.int32)}
    sw_fit = {'radial':np.ones_like(y_tr_dict['radial'],float), 'pm1_probs':calculate_sample_weights(y_tr_dict['pm1_probs']),
              'pm2_probs':calculate_sample_weights(y_tr_dict['pm2_probs']), 'pm3_probs':calculate_sample_weights(y_tr_dict['pm3_probs'])}

    
    # ------------------------------------------------------------------
    # 1.  first instantiation  (pick the right model class)
    # ------------------------------------------------------------------
    if leaflet_type == 'outer':
        model_ae = DeepSphereAE_OUTER(
            K_val=K_PARAM,
            LATENT_F_val=LATENT_F_PARAM,
            N_HEADS_val=N_HEADS_PARAM,
            TRANS_LAYERS_val=TRANS_LAYERS_PARAM,
            initial_nside=initial_nside,
            num_initial_features=num_features,
            n_neighbors=N_NEIGHBORS_PARAM,
            pool_type=POOL_TYPE_PARAM)
    else:
        model_ae = DeepSphereAE_INNER(
            K_val=K_PARAM,
            LATENT_F_val=LATENT_F_PARAM,
            N_HEADS_val=N_HEADS_PARAM,
            TRANS_LAYERS_val=TRANS_LAYERS_PARAM,
            initial_nside=initial_nside,
            num_initial_features=num_features,
            n_neighbors=N_NEIGHBORS_PARAM,
            pool_type=POOL_TYPE_PARAM)


    n_pix_init = hp.nside2npix(initial_nside)
    model_ae(Input(shape=(n_pix_init,num_features))) 
    print("Initial model built.")

    f_nside_heads = model_ae.final_nside_for_heads; N_PIX_FH = hp.nside2npix(f_nside_heads)
    nb_r_dyn_list = []
    for ip in range(N_PIX_FH):
        theta, phi = hp.pix2ang(f_nside_heads, ip, nest=True)
        nb_r_dyn_list.append(hp.get_all_neighbours(f_nside_heads, theta, phi, nest=True))
    nb_d_np = np.array(nb_r_dyn_list,dtype=np.int32)
    for i in range(N_PIX_FH): nb_d_np[i, nb_d_np[i,:] == -1] = i 
    NB_CONST_DYNAMIC = tf.constant(nb_d_np)
    print(f"NB_CONST_DYNAMIC for radial loss (nside={f_nside_heads}) initialized with shape {NB_CONST_DYNAMIC.shape}.")

    steps_epoch = max(1,len(X_train_n)//args.batch_size)
    lr_sched = CosineDecayRestarts(args.initial_learning_rate,steps_epoch*10,t_mul=2.,m_mul=0.9,alpha=0.1)
    p8 = partial(patch_precision,patch_size=8); p8.__name__="patch_precision_8"
    
    metrics_cfg = {
        "radial":["mae","mse"],
        "pm1_probs":['accuracy', p8, CustomMeanIoU(4,name="p1iou")], # Use 'accuracy' for Keras default
        "pm2_probs":['accuracy', p8, CustomMeanIoU(4,name="p2iou")],
        "pm3_probs":['accuracy', p8, CustomMeanIoU(4,name="p3iou")]
    }
    
    model_ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_sched), 
                       loss={'radial':radial_loss,'pm1_probs':combined_seg_loss,'pm2_probs':combined_seg_loss,'pm3_probs':combined_seg_loss},
                       loss_weights={'radial':1.,'pm1_probs':2.5,'pm2_probs':2.5,'pm3_probs':2.5}, metrics=metrics_cfg)
    model_ae.summary(print_fn=print)
    
    tb_log_dir_initial = os.path.join(args.output_dir,f'logs_initial_{args.leaflet}') # Prefixed
        # -----------------------------------------------------------------------------
    # before you create the callback
    # -----------------------------------------------------------------------------
    model_ckpt_path_initial = os.path.join(
        args.output_dir,
        f"{args.leaflet}_best_initial_model.weights.h5")   # ← new suffix


    print(f"\n--- Starting Initial Training (on {len(X_train_n)} train, {len(X_test_n)} val samples) ---")
        # ------------------------------------------------------------------
    # 1.  create the two callbacks
    # ------------------------------------------------------------------
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        restore_best_weights=True)

    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=tb_log_dir_initial,
        update_freq=1)
    
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    model_ckpt_path_initial,
    save_best_only=True,
    save_weights_only=True,   # ← critical
    monitor='val_loss',
    mode='min',
    verbose=1)



    # ------------------------------------------------------------------
    # 2.  call model.fit with just those callbacks
    # ------------------------------------------------------------------
    history = model_ae.fit(
        X_train_n, y_tr_dict,
        sample_weight=sw_fit,
        validation_data=(X_test_n, y_te_dict),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[tb_cb, early_stop_cb, ckpt_cb])
    print("\n--- Initial Training Complete ---")
    
    if leaflet_type == 'outer':
        custom_objects_load = {cls.__name__: cls for cls in [DeepSphereAE_OUTER,HealpyGCNN,hp_layer.HealpyChebyshev,hp_layer.HealpyPool,
                                                       hp_layer.Healpy_ResidualLayer,hp_layer.Healpy_Transformer,CustomMeanIoU]}
    else:
        custom_objects_load = {cls.__name__: cls for cls in [DeepSphereAE_INNER,HealpyGCNN,hp_layer.HealpyChebyshev,hp_layer.HealpyPool,
                                                       hp_layer.Healpy_ResidualLayer,hp_layer.Healpy_Transformer,CustomMeanIoU]}
    custom_objects_load.update({'HealpyUnpool':HealpyUnpool, 'radial_loss':radial_loss, 'combined_seg_loss':combined_seg_loss,
                                'sparse_focal_loss':sparse_focal_loss, 'multiclass_dice_loss':multiclass_dice_loss, 
                                'patch_precision_8':p8})
    
    #model_for_eval_and_saving = model_ae 
    # Check if best model needs to be loaded (if EarlyStopping didn't restore or ModelCheckpoint logic is preferred)
    # EarlyStopping with restore_best_weights=True should make model_ae the best model.
    # use the instance we already have
    if early_stop_cb.stopped_epoch > 0:
        print(f"EarlyStopping triggered at epoch {early_stop_cb.stopped_epoch}; "
            "best weights already restored.")
        model_for_eval_and_saving = model_ae
    elif os.path.exists(model_ckpt_path_initial):
        print("EarlyStopping did not trigger. Loading best weights from disk.")
        model_for_eval_and_saving = model_ae           # same architecture
        model_for_eval_and_saving.load_weights(model_ckpt_path_initial)
    else:
        print("Using last-epoch weights (no EarlyStopping, no checkpoint).")
        model_for_eval_and_saving = model_ae


    model_for_eval_and_saving.save(os.path.join(args.output_dir,f"{args.leaflet}_final_evaluated_model.keras")) # Prefixed
    np.save(os.path.join(args.output_dir,f"{args.leaflet}_history_initial.npy"),history.history) # Prefixed
    plot_training_history(history.history, args.output_dir, args.leaflet) # Pass leaflet
    eval_and_save(model_for_eval_and_saving, X_test_n, y_te_dict, args.output_dir, r_mean, r_std, args.leaflet) # Pass leaflet

    if args.refit_on_all_data:
        print("\n--- Refitting model on all available data ---")
        if 'val_loss' in history.history and history.history['val_loss']:
            optimal_epochs = np.argmin(history.history['val_loss']) + 1
        else: 
            optimal_epochs = len(history.epoch) if hasattr(history, 'epoch') and history.epoch else args.epochs
        print(f"Refitting for {optimal_epochs} epochs (determined from initial validation).")

        X_all_data = np.concatenate((X_train_n, X_test_n), axis=0)
        y_all_rad = np.concatenate((y_tr_dict['radial'], y_te_dict['radial']), axis=0)
        y_all_pm1 = np.concatenate((y_tr_dict['pm1_probs'], y_te_dict['pm1_probs']), axis=0)
        y_all_pm2 = np.concatenate((y_tr_dict['pm2_probs'], y_te_dict['pm2_probs']), axis=0)
        y_all_pm3 = np.concatenate((y_tr_dict['pm3_probs'], y_te_dict['pm3_probs']), axis=0)
        y_all_dict = {'radial':y_all_rad, 'pm1_probs':y_all_pm1, 'pm2_probs':y_all_pm2, 'pm3_probs':y_all_pm3}
        
        sw_all_fit = {'radial':np.ones_like(y_all_rad,float), 'pm1_probs':calculate_sample_weights(y_all_pm1),
                      'pm2_probs':calculate_sample_weights(y_all_pm2), 'pm3_probs':calculate_sample_weights(y_all_pm3)}
        print(f"Combined dataset for refit: {X_all_data.shape[0]} samples.")

        # ------------------------------------------------------------------
        # 2.  refit-on-all-data instantiation (same change)
        # ------------------------------------------------------------------
        if leaflet_type == 'outer':
            production_model = DeepSphereAE_OUTER(
                K_val=K_PARAM,
                LATENT_F_val=LATENT_F_PARAM,
                N_HEADS_val=N_HEADS_PARAM,
                TRANS_LAYERS_val=TRANS_LAYERS_PARAM,
                initial_nside=initial_nside,
                num_initial_features=num_features,
                n_neighbors=N_NEIGHBORS_PARAM,
                pool_type=POOL_TYPE_PARAM)
        else:
            production_model = DeepSphereAE_INNER(
                K_val=K_PARAM,
                LATENT_F_val=LATENT_F_PARAM,
                N_HEADS_val=N_HEADS_PARAM,
                TRANS_LAYERS_val=TRANS_LAYERS_PARAM,
                initial_nside=initial_nside,
                num_initial_features=num_features,
                n_neighbors=N_NEIGHBORS_PARAM,
                pool_type=POOL_TYPE_PARAM)

        production_model(Input(shape=(n_pix_init,num_features))) 

        lr_sched_refit = CosineDecayRestarts(args.initial_learning_rate, max(1, len(X_all_data)//args.batch_size)*2, 
                                             t_mul=1.5, m_mul=0.95, alpha=0.05)
        
        tb_log_dir_refit = os.path.join(args.output_dir,f'logs_refit_{args.leaflet}') # Prefixed

        production_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_sched_refit), 
                                   loss={'radial':radial_loss,'pm1_probs':combined_seg_loss,'pm2_probs':combined_seg_loss,'pm3_probs':combined_seg_loss},
                                   loss_weights={'radial':1.,'pm1_probs':2.5,'pm2_probs':2.5,'pm3_probs':2.5}, metrics=metrics_cfg) 
        print("Production model compiled for refitting.")
        
        history_refit = production_model.fit(X_all_data, y_all_dict, sample_weight=sw_all_fit,
                                             batch_size=args.batch_size, epochs=optimal_epochs, verbose=1,
                                             callbacks=[tf.keras.callbacks.TensorBoard(tb_log_dir_refit,update_freq=1)])
        
        production_model.save(os.path.join(args.output_dir, f"{args.leaflet}_production_model_refitted.keras")) # Prefixed
        np.save(os.path.join(args.output_dir,f"{args.leaflet}_history_refit.npy"), history_refit.history) # Prefixed
        print(f"Production model refitted on all data and saved to {args.output_dir}")

    print(f"\n--- All outputs saved to {args.output_dir} ---")