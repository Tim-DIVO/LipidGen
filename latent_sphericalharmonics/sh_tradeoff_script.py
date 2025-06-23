import pandas as pd
import numpy as np
import healpy as hp
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from deepsphere import HealpyGCNN, healpy_layers as hp_layer
import matplotlib.pyplot as plt
import glob
import os
print("GPUs visible to TF:", tf.config.list_physical_devices("GPU"))

leaflet_type = 'outer'
params = np.load("training_output/outer_radial_normalization_params.npz")
radial_mean = params['mean']
radial_std = params['std']
all_folders = [
    d for d in os.listdir(".")
    if d.startswith("configf_") and os.path.isdir(d)
]

with open("training_output/outer_test_folders.txt", "r") as f:
    test_folders = [line.strip() for line in f if line.strip()]
with open("training_output/outer_train_folders.txt", "r") as f:
    train_folders = [line.strip() for line in f if line.strip()]
with open("dryrun.txt", "r") as f:
    dummy_folders = [line.strip() for line in f if line.strip()]

print('all folders', len(all_folders), 'train folders', len(train_folders), 'test folders', len(test_folders))
print(all_folders[0:5], train_folders[:5], test_folders[:5])

outer_blocks = []
inner_blocks = []

def fill_and_shift(cluster_map):
    # cluster_map: floats, NaN where no lipid, 0/1/2 for cluster
    # we want: 0 → no lipid, 1→cluster0, 2→cluster1, 3→cluster2
    out = np.empty_like(cluster_map, dtype=np.float32)
    # first mark “no lipid”
    mask_nan = np.isnan(cluster_map)
    out[mask_nan] = 0.0
    # then shift the real clusters up by 1
    valid = ~mask_nan
    out[valid] = cluster_map[valid] + 1.0
    return out

for config in test_folders:
    path = os.path.join(config, 'analysis_output')

    # --- OUTER leaflet block (shape (4, x, y)) ---
    outer_radial = np.load(os.path.join(path, "area_outer_radial_dist.npy"))
    #print("radial NaNs?", np.isnan(outer_radial).any())
    outer_chol_uns = fill_and_shift(np.load(os.path.join(path, "nmfk_phase_map_OUTER_CHOL_vs_UNS.npy")))
    outer_sat_chol = fill_and_shift(np.load(os.path.join(path, "nmfk_phase_map_OUTER_SAT_vs_CHOL.npy")))
    outer_uns_sat = fill_and_shift(np.load(os.path.join(path, "nmfk_phase_map_OUTER_UNS_vs_SAT.npy")))

    outer_blocks.append([[outer_radial, outer_uns_sat, outer_sat_chol, outer_chol_uns]])

    # --- OUTER leaflet block (shape (4, x, y)) ---
    inner_radial = np.load(os.path.join(path, "area_inner_radial_dist.npy"))
    #print("radial NaNs?", np.isnan(inner_radial).any())
    inner_chol_uns = fill_and_shift(np.load(os.path.join(path, "nmfk_phase_map_INNER_CHOL_vs_UNS.npy")))
    inner_sat_chol = fill_and_shift(np.load(os.path.join(path, "nmfk_phase_map_INNER_SAT_vs_CHOL.npy")))
    inner_uns_sat = fill_and_shift(np.load(os.path.join(path, "nmfk_phase_map_INNER_UNS_vs_SAT.npy")))

    inner_blocks.append([inner_radial, inner_chol_uns, inner_sat_chol, inner_uns_sat])

outer_stacked = []
for block in outer_blocks:
    block = np.stack(block, axis=-1)
    #print(block.shape)
    block = np.swapaxes(block, 0, 3)
    block = np.squeeze(block)
    print(block.shape)
    outer_stacked.append(block)

stacked = np.concat(outer_stacked, axis=0)
#for i, frame in enumerate(stacked):
    #for j in range(4):
        #print(frame[:,j].shape)
        #stacked[i,:,j] = hp.reorder(frame[:,j], r2n=True)


# ------------------- Re-import necessary libraries after kernel reset -------------------
import numpy as np
import healpy as hp
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ---------------- Metric Definitions ----------------
class CustomMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='custom_mean_iou', dtype=None, **kwargs):
        super(CustomMeanIoU, self).__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            name='total_confusion_matrix_var',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.int64
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        predicted_labels = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, dtype=tf.int64)
        predicted_labels = tf.cast(predicted_labels, dtype=tf.int64)

        y_true_flat = tf.reshape(y_true, [-1])
        predicted_labels_flat = tf.reshape(predicted_labels, [-1])

        current_weights_flat = None
        if sample_weight is not None:
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
        self.total_cm.assign(tf.zeros(self.total_cm.shape, dtype=self.total_cm.dtype))

# ---------------- Evaluation Loop ----------------
nside = 16
lmax_values = np.arange(1, 51)
n_clusters = 3
n_samples = stacked.shape[0]
acc_values = []
f1_values = []
iou_values = []
dims_per_lmax = []

for lmax in lmax_values:
    acc_total = 0.0
    f1_total = 0.0
    iou_metric = CustomMeanIoU(num_classes=4)

    for idx in range(n_samples):
        sample = stacked[idx]  # shape: (3072, 4)
        for ch in [1]:
            label_map = sample[:, ch]
            mask = label_map == 0.0
            onehot = np.stack([(label_map == c).astype(float) for c in [1, 2, 3]], axis=-1)
            recon = np.zeros_like(onehot)

            for i in range(n_clusters):
                alm = hp.map2alm(onehot[:, i], lmax=lmax)
                recon[:, i] = hp.alm2map(alm, nside=nside, lmax=lmax, verbose=False)

            recon_probs = recon / np.clip(np.sum(recon, axis=-1, keepdims=True), 1e-6, None)
            recon_logits = np.log(np.clip(recon_probs, 1e-6, 1.0))

            y_true = tf.convert_to_tensor(label_map[~mask][None, ...] - 1, dtype=tf.int32)
            y_pred = tf.convert_to_tensor(recon_logits[~mask][None, ...], dtype=tf.float32)

            # Convert to 1D numpy arrays for weighted F1 computation
            y_true_flat = y_true.numpy().flatten()
            y_pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32).numpy().flatten()

            # Accuracy
            acc_total += np.mean(y_pred_labels == y_true_flat)

            # Weighted F1
            f1 = f1_score(y_true_flat, y_pred_labels, average='weighted')
            f1_total += f1

            # Mean IoU
            iou_metric.update_state(y_true, y_pred)

    acc_values.append(acc_total / n_samples)
    f1_values.append(f1_total / n_samples)
    iou_values.append(iou_metric.result().numpy())
    dims_per_lmax.append(n_clusters * 3 * (lmax + 1) ** 2)

# Convert lists to numpy arrays if needed
lmax_array = np.array(lmax_values)
acc_array = np.array(acc_values)
f1_array = np.array(f1_values)
iou_array = np.array(iou_values)
dims_array = np.array(dims_per_lmax)

np.save('phase_lmax_array.npy',   lmax_array)
np.save('phase_accuracy.npy',     acc_array)
np.save('phase_weighted_f1.npy',  f1_array)
np.save('phase_mean_iou.npy',     iou_array)
np.save('phase_dims.npy',         dims_array)


# ---------------- Baselines from Autoencoder ----------------
ae_acc = 0.589
ae_f1  = 0.600
ae_iou = 0.363
ae_dim = 1152

# ---------------- Find intersection point next biggest ----------------
dims_array_np = np.array(dims_array)
intersection_idx = np.argmin(np.abs(dims_array_np - ae_dim))
intersection_lmax = lmax_array[intersection_idx]

# ---------------- Plotting ----------------
fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.set_xlabel('SH Degree $l$', fontsize=14)
ax1.set_ylabel('Accuracy / Weighted F1 / IoU', color='black', fontsize=14)
ax1.tick_params(axis='both', labelsize=12)

# Plot Accuracy, Weighted F1, and Mean IoU curves
line1, = ax1.plot(lmax_array, acc_array, label='Accuracy', color='tab:blue', linewidth=2)
line2, = ax1.plot(lmax_array, f1_array, label='Weighted F1', color='tab:orange', linewidth=2)
line3, = ax1.plot(lmax_array, iou_array, label='Mean IoU', color='tab:green', linewidth=2)

# AE baseline lines
line4 = ax1.axhline(ae_acc, color='tab:blue', linestyle=':', linewidth=2, label=f'AE Accuracy ({ae_acc})')
line5 = ax1.axhline(ae_f1,  color='tab:orange', linestyle=':', linewidth=2, label=f'AE Weighted F1 ({ae_f1})')
line6 = ax1.axhline(ae_iou, color='tab:green', linestyle=':', linewidth=2, label=f'AE Mean IoU ({ae_iou})')

# Shaded regions (only up to intersection index)
ax1.fill_between(
    lmax_array[:intersection_idx+1], acc_array[:intersection_idx+1], ae_acc,
    where=acc_array[:intersection_idx+1] > ae_acc,
    interpolate=True, color='tab:blue', alpha=0.3
)
ax1.fill_between(
    lmax_array[:intersection_idx+1], f1_array[:intersection_idx+1], ae_f1,
    where=f1_array[:intersection_idx+1] > ae_f1,
    interpolate=True, color='tab:orange', alpha=0.3
)
ax1.fill_between(
    lmax_array[:intersection_idx+1], iou_array[:intersection_idx+1], ae_iou,
    where=iou_array[:intersection_idx+1] > ae_iou,
    interpolate=True, color='tab:green', alpha=0.3
)

# Vertical intersection line
line9 = ax1.axvline(
    intersection_lmax,
    color='gray',
    linestyle='--',
    linewidth=1.5,
    label=f'SH = AE Dim at $l$={intersection_lmax}'
)

ax1.tick_params(axis='y', labelcolor='black', labelsize=12)

# SH dimensionality axis
ax2 = ax1.twinx()
line7, = ax2.plot(lmax_array, dims_array, color='tab:red', linestyle='--', linewidth=2.5, label='SH Dimensionality')
line8 = ax2.axhline(ae_dim, color='tab:red', linestyle=':', linewidth=2, label='AE Dimensionality')
ax2.set_ylabel('SH Dimensionality', color='tab:red', fontsize=14)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=12)

# Combine legends and place inside plot
lines = [line1, line2, line3, line4, line5, line6, line7, line8, line9]
labels = [ln.get_label() for ln in lines]
fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.07, 0.97), fontsize=10)

plt.grid(True)
plt.tight_layout()
plt.savefig("segmentation-sh-vs-ae_weightedF1-dryrun.png", dpi=300)
plt.show()
