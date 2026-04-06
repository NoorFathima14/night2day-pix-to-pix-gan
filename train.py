# import subprocess
# subprocess.run(["pip", "install", "datasets", "pillow", "--quiet"], check=True)

import os
import sys
import time
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import load_dataset
import matplotlib.pyplot as plt

from configs import SOURCE_DIR, EPOCHS, BATCH_SIZE, IMAGE_SIZE, SAVE_EVERY, LOG_EVERY, MODEL_DIR, CHECKPOINT_DIR, SAMPLE_DIR, BACKUP_DIR

print(f"TensorFlow  : {tf.__version__}")
print(f"GPU devices : {tf.config.list_physical_devices('GPU')}")

if os.path.isdir(SOURCE_DIR):
    sys.path.insert(0, SOURCE_DIR)
    print(f"Source files found at: {SOURCE_DIR}")
    print("Files:", os.listdir(SOURCE_DIR))
else:
    sys.path.insert(0, "/kaggle/working")
    print("Using working directory for source files.")

from generator     import Generator
from discriminator import Discriminator
from losses        import Losses
from dataset       import DatasetLoader

os.makedirs(MODEL_DIR,      exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR,     exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)


print("\n── Loading dataset ──")
loader = DatasetLoader(image_size=IMAGE_SIZE)
night_images, day_images = loader.load_data()

print(f"Night images : {night_images.shape}  dtype={night_images.dtype}")
print(f"Day images   : {day_images.shape}    dtype={day_images.dtype}")
print(f"Value range  : [{night_images.min():.2f}, {night_images.max():.2f}]  (expected [-1, 1])")

train_dataset = loader.get_dataset(night_images, day_images, batch_size=BATCH_SIZE)

expected_batches = len(night_images) // BATCH_SIZE
print(f"Batches per epoch: {expected_batches}")

print("\n── Building models ──")
generator     = Generator().build_generator()
discriminator = Discriminator().build_discriminator()
losses        = Losses()

generator.summary(line_length=80)
discriminator.summary(line_length=80)

gen_optimizer  = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

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
    try:
        start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1]) * SAVE_EVERY
    except Exception:
        start_epoch = 0
    print(f"Resumed from checkpoint: {ckpt_manager.latest_checkpoint}  (epoch ~{start_epoch})")
else:
    print("No checkpoint found — starting from scratch.")

# SAMPLE VISUALISATION HELPER
def save_sample_grid(epoch, generator, night_images, day_images, n=3):
    """
    Saves a grid of [night | ground truth day | generated day]
    to SAMPLE_DIR every SAVE_EVERY epochs so you can watch
    the model improve without running predict.py separately.
    """
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

# TRAIN STEP
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

# TRAINING LOOP
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


# 10. FINAL SAVE 
generator.save(f"{MODEL_DIR}/GAN_Generator.keras")
discriminator.save(f"{MODEL_DIR}/GAN_Discriminator.keras")
print(f"\nTraining complete. Final models saved to {MODEL_DIR}/")


# plt.figure(figsize=(10, 4))
# plt.plot(history["gen_loss"],  label="Generator Loss")
# plt.plot(history["disc_loss"], label="Discriminator Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss Curve")
# plt.legend()
# plt.tight_layout()
# loss_plot_path = f"{SAMPLE_DIR}/loss_curve.png"
# plt.savefig(loss_plot_path)
# plt.show()
# print(f"Loss curve saved → {loss_plot_path}")