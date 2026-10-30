"""`python -m piper.train fit` with lighter checkpointing: keep last.ckpt + the single best-sounding
(highest val_mos) checkpoint instead of 10 x 845 MB files. All CLI arguments pass through."""
from lightning.pytorch.callbacks import ModelCheckpoint
import piper.train.__main__ as piper_main

piper_main._DEFAULT_CALLBACKS[:] = [
    ModelCheckpoint(monitor="val_mos", mode="max", save_top_k=1, save_last=True,
                    filename="best-mos", auto_insert_metric_name=False),
]

if __name__ == "__main__":
    piper_main.main()
