import os, logging, healpy as hp, tensorflow as tf
from functools import partial

# ------------------------------------------------------------------
# 0) silence verbose INFO / WARNING spam ---------------------------
# ------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # hide TF C++ info
logging.getLogger("sbi_flows").setLevel("ERROR")

# ------------------------------------------------------------------
# 1) wrap the 8-pixel patch metric so Keras can deserialize it -----
# ------------------------------------------------------------------
from ae_training import patch_precision           # your original func
p8 = partial(patch_precision, patch_size=8)
p8.__name__ = "patch_precision_8"                 # crucial!

# ------------------------------------------------------------------
# 2) collect every custom symbol the archive needs ----------------
# ------------------------------------------------------------------
from ae_training import (
    DeepSphereAE_OUTER, DeepSphereAE_INNER,
    HealpyUnpool, CustomMeanIoU,
    radial_loss, combined_seg_loss
)

CUSTOM = {
    "DeepSphereAE_OUTER": DeepSphereAE_OUTER,
    "DeepSphereAE_INNER": DeepSphereAE_INNER,
    "HealpyUnpool":      HealpyUnpool,
    "CustomMeanIoU":     CustomMeanIoU,
    "radial_loss":       radial_loss,
    "combined_seg_loss": combined_seg_loss,
    "patch_precision_8": p8,
    "partial":           partial,                # ← still required
}

# ------------------------------------------------------------------
# 3) convenience loader + single-shot infer ------------------------
# ------------------------------------------------------------------
def load_dsphere(model_path: str, *, gpu=False):
    """Return a ready-to-use (and already 'built') DeepSphere model."""
    if not gpu:            # force CPU if you like
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model = tf.keras.models.load_model(
        model_path, custom_objects=CUSTOM, compile=False
    )

    # one dummy call so that encoder_model / decoder_model
    # are materialised and you get nice `.summary()` output
    npix   = hp.nside2npix(model.initial_nside)
    nd_feat = model.num_initial_features
    _ = model(tf.zeros((1, npix, nd_feat)))   # builds the graph

    return model


def run_full(model, x):
    """Full forward pass → dict with radial / pm1 / pm2 / pm3 logits."""
    return model(x, training=False)


def run_encoder(model, x):
    """Only encoder → bottleneck tensor"""
    return model.encoder_model(x, training=False)    # cached Functional


def run_decoder(model, z):
    """Only decoder → heads dict"""
    return model.decoder_model(z, training=False)
