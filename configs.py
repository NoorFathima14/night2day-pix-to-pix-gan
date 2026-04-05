
EPOCHS          = 200     # 200 minimum for good results; use 100 for a quick smoke test
BATCH_SIZE      = 4       # Pix2Pix paper uses batch size 1
IMAGE_SIZE      = 128
SAVE_EVERY      = 10      # Save checkpoint every N epochs
LOG_EVERY       = 50      # Print batch loss every N batches
MODEL_DIR       = "/kaggle/working/model"
CHECKPOINT_DIR  = f"{MODEL_DIR}/checkpoints"
SAMPLE_DIR      = "/kaggle/working/samples"   # Where sample prediction grids are saved
BACKUP_DIR      = "/kaggle/working/backup"   # NEW
SOURCE_DIR = "/kaggle/input/night2day-pix2pix"  
