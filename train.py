# import time
# import tensorflow as tf
# import os
# from generator import Generator
# from discriminator import Discriminator
# from losses import Losses
# from dataset import DatasetLoader


# class TrainPix2Pix:
#     def __init__(self, dataset_path, epochs=200):
#         self.dataset_path = dataset_path
#         self.epochs = epochs
#         self.dataset_loader = DatasetLoader(dataset_path)

#         # Load data and verify
#         try:
#             self.images, self.masks = self.dataset_loader.load_data()
#             print(f"Loaded {len(self.images)} input images and {len(self.masks)} target images.")
#             if len(self.images) == 0 or len(self.masks) == 0:
#                 raise ValueError("No images loaded. Check dataset path and loader.")
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             raise

#         # Create dataset
#         self.data = self.dataset_loader.get_dataset(self.images, self.masks)

#         # Verify dataset with a single sample
#         try:
#             sample_image, sample_mask = next(iter(self.data))
#             print(f"Sample batch — Image: {sample_image.shape}, Mask: {sample_mask.shape}")
#             batch_size = sample_image.shape[0] or 1
#             self.expected_batches = len(self.images) // batch_size
#             print(f"Expected batches per epoch: {self.expected_batches}")
#             self.data = self.data.shuffle(buffer_size=1000)
#         except Exception as e:
#             print(f"Error validating dataset: {e}")
#             raise

#         # Build models
#         self.generator     = Generator().build_generator()
#         self.discriminator = Discriminator().build_discriminator()
#         self.losses        = Losses()

#         # Optimizers — Adam with lr=2e-4, beta_1=0.5 per Pix2Pix paper
#         self.gen_optimizer  = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
#         self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

#         # Checkpoint manager — saves every 10 epochs so you can resume
#         checkpoint_dir = "model/checkpoints"
#         os.makedirs(checkpoint_dir, exist_ok=True)
#         self.checkpoint = tf.train.Checkpoint(
#             generator=self.generator,
#             discriminator=self.discriminator,
#             gen_optimizer=self.gen_optimizer,
#             disc_optimizer=self.disc_optimizer,
#         )
#         self.ckpt_manager = tf.train.CheckpointManager(
#             self.checkpoint, checkpoint_dir, max_to_keep=5
#         )

#         # Restore latest checkpoint if available (useful for resuming on Kaggle)
#         if self.ckpt_manager.latest_checkpoint:
#             self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
#             print(f"Restored checkpoint: {self.ckpt_manager.latest_checkpoint}")
#         else:
#             print("No checkpoint found — starting from scratch.")

#     # FIX: @tf.function compiles the step into a graph, giving a significant
#     # speed boost on Kaggle GPUs (was running in slow eager mode before).
#     @tf.function
#     def train_step(self, inputs, target):
#         with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#             # Generator forward pass (training=True enables dropout)
#             generated_output = self.generator(inputs, training=True)

#             # Discriminator on real and fake pairs
#             real_output = self.discriminator([inputs, target],           training=True)
#             fake_output = self.discriminator([inputs, generated_output], training=True)

#             # Losses
#             gen_loss, gen_gan_loss, gen_l1_loss = self.losses.generator_loss(
#                 fake_output, generated_output, target
#             )
#             disc_loss = self.losses.discriminator_loss(real_output, fake_output)

#         # Apply gradients
#         gen_gradients  = gen_tape.gradient(gen_loss,  self.generator.trainable_variables)
#         disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

#         self.gen_optimizer.apply_gradients(
#             zip(gen_gradients,  self.generator.trainable_variables)
#         )
#         self.disc_optimizer.apply_gradients(
#             zip(disc_gradients, self.discriminator.trainable_variables)
#         )

#         return gen_loss, gen_gan_loss, gen_l1_loss, disc_loss

#     def fit(self):
#         print(f"\nStarting training for {self.epochs} epochs...\n")

#         for epoch in range(self.epochs):
#             start = time.time()
#             print(f"── Epoch {epoch + 1}/{self.epochs} ──")

#             batch_count  = 0
#             total_gen_loss  = 0.0
#             total_disc_loss = 0.0

#             for image, mask in self.data.take(self.expected_batches):
#                 gen_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(image, mask)
#                 total_gen_loss  += gen_loss
#                 total_disc_loss += disc_loss
#                 batch_count += 1

#                 if batch_count % 10 == 0:
#                     print(
#                         f"  Batch {batch_count:4d} | "
#                         f"Gen Loss: {gen_loss:.4f} "
#                         f"(GAN: {gen_gan_loss:.4f}, L1: {gen_l1_loss:.4f}) | "
#                         f"Disc Loss: {disc_loss:.4f}"
#                     )

#             if batch_count == 0:
#                 print("  Warning: No batches processed. Check dataset configuration.")
#                 continue

#             avg_gen  = total_gen_loss  / batch_count
#             avg_disc = total_disc_loss / batch_count
#             elapsed  = time.time() - start
#             print(
#                 f"  Epoch {epoch + 1} done in {elapsed:.1f}s | "
#                 f"Avg Gen Loss: {avg_gen:.4f} | Avg Disc Loss: {avg_disc:.4f}\n"
#             )

#             # Save checkpoint every 10 epochs (allows resuming on Kaggle session restart)
#             if (epoch + 1) % 10 == 0:
#                 ckpt_path = self.ckpt_manager.save()
#                 print(f"  Checkpoint saved: {ckpt_path}")

#                 # Also save .keras weights every 10 epochs
#                 os.makedirs("model", exist_ok=True)
#                 self.generator.save("model/GAN_Generator.keras")
#                 self.discriminator.save("model/GAN_Discriminator.keras")
#                 print("  Model weights saved to model/\n")

#         # Final save
#         os.makedirs("model", exist_ok=True)
#         self.generator.save("model/GAN_Generator.keras")
#         self.discriminator.save("model/GAN_Discriminator.keras")
#         print("Training complete. Final models saved to model/")


# if __name__ == "__main__":
#     try:
#         EPOCHS       = 200   # Increase from 100 — minimum for good Pix2Pix results
#         DATASET_PATH = "night2day/train"   # Update this path for Day<->Night dataset
#         trainer = TrainPix2Pix(DATASET_PATH, epochs=EPOCHS)
#         trainer.fit()
#     except Exception as e:
#         print(f"Training failed: {e}")
#         raise

# ============================================================
#  Pix2Pix Night → Day  |  Kaggle Training Notebook
#  Run this as a single cell after uploading all .py files
#  as a Kaggle Dataset or via the file upload panel.
# ============================================================

# ------------------------------------------------------------
# 0. INSTALL DEPENDENCIES
# ------------------------------------------------------------
import subprocess
subprocess.run(["pip", "install", "datasets", "pillow", "--quiet"], check=True)

# ------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------
import os
import sys
import time
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import load_dataset
import matplotlib.pyplot as plt

print(f"TensorFlow  : {tf.__version__}")
print(f"GPU devices : {tf.config.list_physical_devices('GPU')}")

# ------------------------------------------------------------
# 2. ADD YOUR SOURCE FILES TO PATH
# ------------------------------------------------------------
# If you uploaded the .py files as a Kaggle Dataset, they will
# be at /kaggle/input/<dataset-name>/  — update SOURCE_DIR below.
# If you pasted them directly into the notebook, skip this block.

SOURCE_DIR = "/kaggle/input/night2day-pix2pix"   # ← update to your dataset name
if os.path.isdir(SOURCE_DIR):
    sys.path.insert(0, SOURCE_DIR)
    print(f"Source files found at: {SOURCE_DIR}")
    print("Files:", os.listdir(SOURCE_DIR))
else:
    # Files are in the working directory (pasted or uploaded via sidebar)
    sys.path.insert(0, "/kaggle/working")
    print("Using working directory for source files.")

from generator     import Generator
from discriminator import Discriminator
from losses        import Losses
from dataset       import DatasetLoader

# ------------------------------------------------------------
# 3. CONFIG  ← tweak these as needed
# ------------------------------------------------------------
EPOCHS          = 200     # 200 minimum for good results; use 100 for a quick smoke test
BATCH_SIZE      = 4       # Pix2Pix paper uses batch size 1
IMAGE_SIZE      = 128
SAVE_EVERY      = 10      # Save checkpoint every N epochs
LOG_EVERY       = 50      # Print batch loss every N batches
MODEL_DIR       = "/kaggle/working/model"
CHECKPOINT_DIR  = f"{MODEL_DIR}/checkpoints"
SAMPLE_DIR      = "/kaggle/working/samples"   # Where sample prediction grids are saved
BACKUP_DIR      = "/kaggle/working/backup"   # NEW

os.makedirs(MODEL_DIR,      exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR,     exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
# ------------------------------------------------------------
# 4. LOAD DATASET
# ------------------------------------------------------------
print("\n── Loading dataset ──")
loader = DatasetLoader(image_size=IMAGE_SIZE)
night_images, day_images = loader.load_data()

print(f"Night images : {night_images.shape}  dtype={night_images.dtype}")
print(f"Day images   : {day_images.shape}    dtype={day_images.dtype}")
print(f"Value range  : [{night_images.min():.2f}, {night_images.max():.2f}]  (expected [-1, 1])")

train_dataset = loader.get_dataset(night_images, day_images, batch_size=BATCH_SIZE)

# train_dataset = loader.get_dataset(batch_size=BATCH_SIZE)
expected_batches = len(night_images) // BATCH_SIZE
print(f"Batches per epoch: {expected_batches}")

# ------------------------------------------------------------
# 5. BUILD MODELS
# ------------------------------------------------------------
print("\n── Building models ──")
generator     = Generator().build_generator()
discriminator = Discriminator().build_discriminator()
losses        = Losses()

generator.summary(line_length=80)
discriminator.summary(line_length=80)

gen_optimizer  = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

# ------------------------------------------------------------
# 6. CHECKPOINT — resume if a previous session exists
# ------------------------------------------------------------
checkpoint = tf.train.Checkpoint(
    generator=generator,
    discriminator=discriminator,
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer,
)
ckpt_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    # Infer which epoch we're resuming from based on checkpoint filename
    try:
        start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1]) * SAVE_EVERY
    except Exception:
        start_epoch = 0
    print(f"Resumed from checkpoint: {ckpt_manager.latest_checkpoint}  (epoch ~{start_epoch})")
else:
    print("No checkpoint found — starting from scratch.")

# ------------------------------------------------------------
# 7. SAMPLE VISUALISATION HELPER
#    Saves a grid of [night | ground truth day | generated day]
#    to SAMPLE_DIR every SAVE_EVERY epochs so you can watch
#    the model improve without running predict.py separately.
# ------------------------------------------------------------
def save_sample_grid(epoch, generator, night_images, day_images, n=3):
    indices = np.random.choice(len(night_images), n, replace=False)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(f"Epoch {epoch}", fontsize=14)

    col_titles = ["Night Input", "Ground Truth Day", "Generated Day"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11)

    def to_uint8(arr):
        return np.clip(((arr + 1.0) * 127.5), 0, 255).astype(np.uint8)

    for row, idx in enumerate(indices):
        night_in   = night_images[idx]
        day_gt     = day_images[idx]
        prediction = generator.predict(
            np.expand_dims(night_in, 0), verbose=0
        )[0]

        axes[row][0].imshow(to_uint8(night_in));   axes[row][0].axis("off")
        axes[row][1].imshow(to_uint8(day_gt));     axes[row][1].axis("off")
        axes[row][2].imshow(to_uint8(prediction)); axes[row][2].axis("off")

    plt.tight_layout()
    path = f"{SAMPLE_DIR}/epoch_{epoch:04d}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  Sample grid saved → {path}")

# ------------------------------------------------------------
# 8. TRAIN STEP
# ------------------------------------------------------------
@tf.function
def train_step(inputs, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_output = generator(inputs, training=True)

        real_output = discriminator([inputs, target],           training=True)
        fake_output = discriminator([inputs, generated_output], training=True)

        gen_loss, gen_gan_loss, gen_l1_loss = losses.generator_loss(
            fake_output, generated_output, target
        )
        disc_loss = losses.discriminator_loss(real_output, fake_output)

    gen_gradients  = gen_tape.gradient(gen_loss,  generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients( zip(gen_gradients,  generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, gen_gan_loss, gen_l1_loss, disc_loss

# ------------------------------------------------------------
# 9. TRAINING LOOP
# ------------------------------------------------------------
print(f"\n── Training for {EPOCHS} epochs (resuming from epoch {start_epoch}) ──\n")

history = {"gen_loss": [], "disc_loss": []}
best_loss = float('inf')
for epoch in range(start_epoch, EPOCHS):
    epoch_start   = time.time()
    batch_count   = 0
    total_gen_loss  = 0.0
    total_disc_loss = 0.0

    print(f"── Epoch {epoch + 1}/{EPOCHS} ──")

    for night_batch, day_batch in train_dataset.take(expected_batches):
        gen_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(night_batch, day_batch)
        total_gen_loss  += gen_loss.numpy()
        total_disc_loss += disc_loss.numpy()
        batch_count     += 1

        if batch_count % LOG_EVERY == 0:
            print(
                f"  Batch {batch_count:5d}/{expected_batches} | "
                f"Gen: {gen_loss:.4f}  "
                f"(GAN: {gen_gan_loss:.4f}  L1: {gen_l1_loss:.4f}) | "
                f"Disc: {disc_loss:.4f}"
            )

    if batch_count == 0:
        print("  ⚠  No batches processed — check dataset config.")
        continue

    avg_gen  = total_gen_loss  / batch_count
    avg_disc = total_disc_loss / batch_count
    elapsed  = time.time() - epoch_start

    history["gen_loss"].append(avg_gen)
    history["disc_loss"].append(avg_disc)

    print(
        f"  ✓ Epoch {epoch + 1} | "
        f"{elapsed:.1f}s | "
        f"Avg Gen: {avg_gen:.4f} | "
        f"Avg Disc: {avg_disc:.4f}\n"
    )

    # Save checkpoint + sample grid every SAVE_EVERY epochs
    if (epoch + 1) % SAVE_EVERY == 0:

        # 1. Save checkpoint (for full training resume)
        ckpt_path = ckpt_manager.save()
        print(f"Checkpoint saved → {ckpt_path}")
    
        # 2. Save models (backup safety)
        generator.save(f"{MODEL_DIR}/generator_latest.keras")
        discriminator.save(f"{MODEL_DIR}/discriminator_latest.keras")
    
        # 3. ALSO save versioned models (CRUCIAL)
        generator.save(f"{BACKUP_DIR}/generator_epoch_{epoch+1}.keras")
        discriminator.save(f"{BACKUP_DIR}/discriminator_epoch_{epoch+1}.keras")

        if avg_gen < best_loss:
            generator.save(f"{MODEL_DIR}/best_generator.keras")
    
        print(f"Models backed up at epoch {epoch+1}")

        save_sample_grid(epoch + 1, generator, night_images, day_images, n=3)

        import shutil
        
        shutil.make_archive(
            "/kaggle/working/checkpoints_backup",
            'zip',
            "/kaggle/working/model/checkpoints"
        )

# ------------------------------------------------------------
# 10. FINAL SAVE + LOSS PLOT
# ------------------------------------------------------------
generator.save(f"{MODEL_DIR}/GAN_Generator.keras")
discriminator.save(f"{MODEL_DIR}/GAN_Discriminator.keras")
print(f"\nTraining complete. Final models saved to {MODEL_DIR}/")

# Loss curve
plt.figure(figsize=(10, 4))
plt.plot(history["gen_loss"],  label="Generator Loss")
plt.plot(history["disc_loss"], label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.tight_layout()
loss_plot_path = f"{SAMPLE_DIR}/loss_curve.png"
plt.savefig(loss_plot_path)
plt.show()
print(f"Loss curve saved → {loss_plot_path}")