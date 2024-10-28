
from easyeditor import EditTrainer, MENDTrainingHparams, ZsreDataset


training_hparams = MENDTrainingHparams.from_hparams('hparams/TRAINING/MEND/llama-7b.yaml')
train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()

## Loading config from hparams/MEMIT/gpt2-xl.yaml
# hparams = MENDHyperParams.from_hparams('./hparams/MEND/gpt2-xl')


