NT = 2
NR = 2
NP_DEFAULT = 8
ND_PER_BLOCK = 256

SNR_DB_LIST = (0, 4, 8, 12, 16, 20)
PILOT_SWEEP = (2, 4, 8, 16, 32)

TRAIN_SAMPLES = 200_000
VAL_SAMPLES = 40_000
TEST_SAMPLES = 40_000

DATASET_SEEDS = {"train": 1001, "val": 2001, "test": 3001}
MODEL_SEEDS = (4101, 4102, 4103)

FEATURE_DIM = 12
N_CLASSES = 4

FP32_ARCHITECTURES = (
    (), (16,), (16, 16), (32,), (64,), (32, 32), (128,),
    (64, 32), (64, 64), (128, 64), (128, 64, 32), (128, 128),
)

QAT_ARCHITECTURES = ((32,), (64,), (128,), (64, 64))
QAT_PRECISIONS = ((8, 8), (6, 6), (4, 4))

TRAINING = {"lr": 1e-3, "batch_size": 1024, "epochs": 30}
