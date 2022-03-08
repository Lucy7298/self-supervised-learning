from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl 

class CustomCheckpoint(ModelCheckpoint): 
    """
    checkpoint_freq: list of (min_epoch, max_epoch, frequency of checkpoint 
    when min_epoch <= epoch <= max_epoch)
    """
    def __init__(self, checkpoint_freq, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.checkpoint_freq = checkpoint_freq

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the training epoch."""
        # as we advance one step at end of training, we use `global_step - 1` to avoid saving duplicates
        trainer.fit_loop.global_step -= 1
        # ignore self._every_n_epochs 
        # instead, go by checkpoint_freq_dict 
        curr_freq = None
        for (min_epoch, max_epoch, freq) in self.checkpoint_freq: 
            if min_epoch <= (trainer.current_epoch + 1) <= max_epoch: 
                curr_freq = freq 
                break 
        print(curr_freq)
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and self._save_on_train_epoch_end
            and (trainer.current_epoch + 1) % curr_freq == 0
        ):
            self.save_checkpoint(trainer)
        trainer.fit_loop.global_step += 1