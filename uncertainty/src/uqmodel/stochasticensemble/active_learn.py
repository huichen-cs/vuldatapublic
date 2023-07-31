import logging
import os
import pickle # nosec
import torch
from typing import Tuple, List
from uqmodel.stochasticensemble.experiment_config import ExperimentConfig
from uqmodel.stochasticensemble.train_utils import (
    build_datasets
)
from uqmodel.stochasticensemble.ps_data import (
    FeatureDataSet
)

logger = logging.getLogger('active_learn')

class PsFeatureExperimentDatasets(object):
    def __init__(self, config:ExperimentConfig, tag:str, seed:int=1432):
        self.ckpt = EnsembleCheckpoint(
            config.trainer.checkpoint.dir_path, 
            warmup_epochs=config.trainer.checkpoint.warmup_epochs,
            tag=tag)

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.ps_columns
        ) = (build_datasets(config, self.ckpt))
        
        init_train_size = len(self.train_dataset)//2
        pool_data_size = len(self.train_dataset) - init_train_size
        generator = torch.Generator().manual_seed(seed)
        self.run_dataset, self.pool_dataset = torch.utils.data.random_split(
            self.train_dataset,
            [init_train_size, pool_data_size],
            generator=generator
        )

    def update(self, size, entropy_epistermic, entropy_aleatoric, method, tag):
        self.ckpt.ckpt_tag, self.ckpt.min_total_loss = tag, None
        uq_list = list(zip(range(len(self.pool_dataset)), entropy_epistermic,
                           entropy_aleatoric))
        if method == 'ehal': # case 1
            sorted_uq_list = sorted(uq_list, key=lambda u: (-u[1], u[2]))
        elif method == 'elah': # case 2
            sorted_uq_list = sorted(uq_list, key=lambda u: (u[1], -u[2]))
        elif method == 'ehah': # case 3
            sorted_uq_list = sorted(uq_list, key=lambda u: (-u[1], -u[2]))
        elif method == 'elal': # case 4
            sorted_uq_list = sorted(uq_list, key=lambda u: (u[1], u[2]))
        #
        elif method == 'aleh': # case 5
            sorted_uq_list = sorted(uq_list, key=lambda u: (u[2], -u[1]))
        elif method == 'ahel': # case 6
            sorted_uq_list = sorted(uq_list, key=lambda u: (-u[2], u[1]))
        elif method == 'aheh': # case 7
            sorted_uq_list = sorted(uq_list, key=lambda u: (-u[2], -u[1]))
        elif method == 'alel': # case 8
            sorted_uq_list = sorted(uq_list, key=lambda u: (u[2], u[1]))
        else:
            raise ValueError('unimplemented method {}'.format(method))

        indices = [uq[0] for uq in sorted_uq_list[0:size]]
        selected = torch.utils.data.Subset(self.pool_dataset, indices)
        self.run_dataset = torch.utils.data.ConcatDataset([self.run_dataset, selected])

        indices = [uq[0] for uq in sorted_uq_list[size:]]
        self.pool_dataset = torch.utils.data.Subset(self.pool_dataset, indices)
        logger.info('method: {}, len(run_dataset): {}, len(pool_dataset): {}'.format(method, len(self.run_dataset), len(self.pool_dataset)))


class PsFeatureExperimentDataLoaders(object):
    def __init__(self, config, datasets, train=True):
        self.config = config
        self.datasets = datasets

        self.train_dataloader = None # prevent logic error
        
        self.run_dataloader = torch.utils.data.DataLoader(
            datasets.run_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.trainer.pin_memory
        )
        self.pool_dataloader = torch.utils.data.DataLoader(
            datasets.pool_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.trainer.pin_memory
        )
        self.val_dataloader =  torch.utils.data.DataLoader(
            datasets.val_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.trainer.pin_memory
        )
        if train: # prevent logic error
            self.test_dataloader = None
        else:
            self.test_dataloader =  torch.utils.data.DataLoader(
                datasets.test_dataset,
                batch_size=config.trainer.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.trainer.pin_memory
            )
        logger.info(f'len(run_dataset) of len(train_datasetf): {len(datasets.run_dataset)} of {len(datasets.train_dataset)}')


class EnsembleCheckpoint(object):
    """
        Save checkpoint.
    
        Reference:
            https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    """
    def __init__(self, checkpoint_dir_path, warmup_epochs=5, min_loss=None, tag=None, train=True):
        self._ckpt_path = checkpoint_dir_path
        if not os.path.exists(self._ckpt_path):
            os.makedirs(self._ckpt_path)
        self._min_total_loss = min_loss
        self._warmup_epochs = warmup_epochs
        self._checkpointed = False
        self._tag = tag
        self._train = train

    @property
    def warmup_epochs(self):
        return self._warmup_epochs

    @warmup_epochs.setter
    def warmup_epochs(self, epochs):
        self._warmup_epochs = epochs

    @property
    def min_total_loss(self):
        return self._min_total_loss

    @min_total_loss.setter
    def min_total_loss(self, loss):
        self._min_total_loss = loss

    @property
    def ckpt_path(self):
        return self._ckpt_path

    @property
    def ckpt_tag(self):
        return self._tag

    @ckpt_tag.setter
    def ckpt_tag(self, _tag):
        self._tag = _tag

    def ckpt_dataset_path(self, ds_type):
        if not self._train:
            file_path = os.path.join(self._ckpt_path, f'dataset_{ds_type}.pickle')
            return file_path

        if self._tag:
            file_path = os.path.join(self._ckpt_path, f'dataset_{ds_type}_{self._tag}.pickle')
        else:
            file_path = os.path.join(self._ckpt_path, f'dataset_{ds_type}.pickle')
        return file_path

    def ckpt_model_path(self, model_index):
        if self._tag:
            file_path = os.path.join(self._ckpt_path, f'model_{model_index}_{self._tag}')
        else:
            file_path = os.path.join(self._ckpt_path, f'model_{model_index}')
        return file_path

    def chkt_model_meta_path(self, use_default=False):
        if self._tag and not use_default:
            en_filepath = os.path.join(self._ckpt_path, 'model_en_{}'.format(self._tag))
        else:
            en_filepath = os.path.join(self._ckpt_path, 'model_en')
        return en_filepath

    def __call__(self, epoch, model_ensemble, optimizer_ensemble, scheduler_ensemble, criteria_ensemble, loss_ensemble, total_loss):
        logger.debug(f'Checkpoint: checking for min loss: {self._min_total_loss} loss: {total_loss}')
        if epoch < self._warmup_epochs:
            if  self._min_total_loss and total_loss < self._min_total_loss:
                self._min_total_loss = total_loss
            elif not self._min_total_loss:
                self._min_total_loss = total_loss
            return
        if (self._min_total_loss and total_loss < self._min_total_loss) or (not self._checkpointed):
            self.save_checkpoint(epoch, model_ensemble, optimizer_ensemble, scheduler_ensemble, criteria_ensemble, loss_ensemble, total_loss)
            self._min_total_loss = total_loss
            self._checkpointed = True
        if not self._min_total_loss:
            self._min_total_loss = total_loss

    def save_checkpoint(self, epoch, model_ensemble, optimizer_ensemble, scheduler_ensemble, criteria_ensemble, loss_ensemble, total_loss):
        en_filepath = self.chkt_model_meta_path()
        en_dict = {
                'epoch': epoch,
                'ensemble_size': len(model_ensemble),
                'total_loss': total_loss,
                'min_total_loss': self._min_total_loss
            }
        torch.save(en_dict, en_filepath)
        logger.info('saved checkpoint meta at {}'.format(en_filepath))
        for index,(model,optimizer,scheduler,criteria,loss) in enumerate(
                zip(model_ensemble,
                    optimizer_ensemble,
                    scheduler_ensemble,
                    criteria_ensemble,
                    loss_ensemble)):
            self.save_member_checkpoint(index, model, optimizer, scheduler, criteria, loss)
            logger.info(f'saved checkpoint for model {index} to {self._ckpt_path}, loss {self._min_total_loss} -> {total_loss}')

    def save_member_checkpoint(self, index, model, optimizer, scheduler, criteria, loss):
        file_path = self.ckpt_model_path(index)
        torch.save(
            {
                'index': index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'criteria_state_dict': criteria.state_dict(),
                'loss': loss
            }, file_path)
        logger.info('saved checkpoint ensemble member {} at {}'.format(index, file_path))

    def load_model_meta(self):
        try:
            en_filepath = self.chkt_model_meta_path(use_default=False)
            en_dict = torch.load(en_filepath)
        except FileNotFoundError:
            try:
                en_filepath = self.chkt_model_meta_path(use_default=True)
                en_dict = torch.load(en_filepath)
            except FileNotFoundError as err:
                raise err
        logger.info('load model meta for ensemble at {}'.format(en_filepath))
        return en_dict

    def load_checkpoint(self, model_ensemble, optimizer_ensemble, scheduler_ensemble, criteria_ensemble, loss_ensemble):
        en_dict = self.load_model_meta()
        epoch = en_dict['epoch']
        total_loss = en_dict['total_loss']
        ensemble_size = en_dict['ensemble_size']
        min_total_loss = en_dict['min_total_loss']
        for i in range(ensemble_size):
            try:
                file_path = self.ckpt_model_path(i)
                member_checkpoint = torch.load(file_path)
            except FileNotFoundError as err:
                raise err
            model_ensemble[i].load_state_dict(member_checkpoint['model_state_dict'])
            optimizer_ensemble[i].load_state_dict(member_checkpoint['optimizer_state_dict'])
            scheduler_ensemble[i].load_state_dict(member_checkpoint['scheduler_state_dict'])
            criteria_ensemble[i].load_state_dict(member_checkpoint['criteria_state_dict'])
            loss_ensemble[i] = member_checkpoint['loss']
            logger.info('load model {} for ensemble at {}'.format(i, file_path))
        return epoch, model_ensemble, optimizer_ensemble, scheduler_ensemble, criteria_ensemble, loss_ensemble, min_total_loss, total_loss

    def save_datasets(self, train_dataset:FeatureDataSet, val_dataset:FeatureDataSet, test_dataset:FeatureDataSet, ps_columns:List) -> None:
        for ds_type,ds in zip(['train','val', 'test', 'ps_columns'],
                              [train_dataset, val_dataset, test_dataset, ps_columns]):
            file_path = self.ckpt_dataset_path(ds_type)
            with open(file_path, 'wb') as f:
                pickle.dump(ds, f)


    def load_datasets(self) -> Tuple[FeatureDataSet, FeatureDataSet, FeatureDataSet, List]:
        ds_list = []
        for ds_type in ['train', 'val', 'test', 'ps_columns']:
            file_path = self.ckpt_dataset_path(ds_type)
            with open(file_path, 'rb') as f:
                ds_list.append(pickle.load(f)) # nosec
        return ds_list
