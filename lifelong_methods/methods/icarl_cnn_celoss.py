import torch.nn as nn
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple

from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import NO_LABEL_PLACEHOLDER
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.methods.base_method import BaseMethod
from lifelong_methods.utils import SubsetSampler, copy_freeze


class Model(BaseMethod):
    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, config)

        self.old_net = copy_freeze(self.net)
        
        self.dist_loss = nn.BCELoss()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.criterion=nn.CrossEntropyLoss()

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        pass

    def _prepare_model_for_new_task(self, **kwargs) -> None:
        self.old_net = copy_freeze(self.net)
        self.old_net.eval()

    def _preprocess_target(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        offset1, offset2 = self._compute_offsets(self.cur_task_id)
        y = y.clone()
        if self.cur_task_id > 0:
            distill_model_output, _ = self.old_net(x)
            distill_model_output = distill_model_output.detach()
            distill_model_output = torch.sigmoid(distill_model_output / self.temperature)
            y[:, :offset1] = distill_model_output[:, :offset1]
        return y

    def observe(self, x: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
                train: bool = True, super_class_index=None, sub_class_index=None) \
                            -> Tuple[torch.Tensor, float]:

        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)
        target = y

        if offset_1 > 0:
            distill_model_output, _ = self.old_net(x)
            distill_model_output = distill_model_output.detach()
            distill_model_output = torch.sigmoid(distill_model_output / self.temperature)
            # distill_model_output[:, :offset_1]

        assert target.shape[1] == offset_2
        output, _ = self.forward_net(x)
        output = output[:, :offset_2]

        # output_1=output[:, super_class_index]
        # output_2=output[:, sub_class_index]

        # target_1 = target[:, super_class_index]
        # target_2 = target[:, sub_class_index]
        # np.delete(logic_add, super_class_index)
        # target.type(torch.int64)

        lbl_target=torch.where(target==1)[1]
        loss = self.criterion(output / self.temperature, lbl_target)
        
        ####################33
        if offset_1 > 0:
            g = F.sigmoid(output)
            dist_loss = sum(self.dist_loss(g[:,y], distill_model_output[:,y])\
                    for y in range(offset_1))
            loss += dist_loss
        # if offset_1 > 0:
        #     # The original code does not match with the paper equation, maybe sigmoid could be removed from g
        #     g = torch.sigmoid(torch.cat(output[:offset_1], dim=1))
        #     q_i = torch.sigmoid(torch.cat(outputs_old[:offset_1], dim=1))
        #     loss += self.lamb * sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in
        #                             range(sum(self.model.task_cls[:t])))

        # if offset_1>0:
        #     dist_loss=0

        # loss_1 = self.bce(output_1 / self.temperature, target_1)
        # loss_2 = self.bce(output_2 / self.temperature, target_2)
        # loss=loss_1+loss_2

        if train:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        # output.max(dim=1)[0]
        ce_output=output-output.max(dim=1)[0].reshape(-1,1).repeat(1,offset_2)
        predictions = ce_output.ge(0.0)
        # predictions = output.ge(0.0)
        return predictions, loss.item()

    def forward(self, x: torch.Tensor, return_output=False) -> torch.Tensor:

        num_seen_classes = len(self.seen_classes)
        output, _ = self.forward_net(x)
        output = output[:, :num_seen_classes]
        
        ce_output=output-output.max(dim=1)[0].reshape(-1,1).repeat(1,num_seen_classes)
        
        predictions = ce_output.ge(0.0)

        # predictions = output.ge(0.0)
        if return_output:
            return predictions, output
        else:
            return predictions

    def _consolidate_epoch_knowledge(self, **kwargs) -> None:

        pass

    def consolidate_task_knowledge(self, **kwargs) -> None:
        
        pass


class Buffer(BufferBase):
    def __init__(self,
                 config: Dict,
                 buffer_dir: Optional[str] = None,
                 map_size: int = 1e9,
                 essential_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 augmentation_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        super(Buffer, self).__init__(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    def _reduce_exemplar_set(self, **kwargs) -> None:
        for label in self.seen_classes:
            if len(self.mem_class_x[label]) > self.n_mems_per_cla:
                n = len(self.mem_class_x[label]) - self.n_mems_per_cla
                self.remove_samples(label, n)

    def _construct_exemplar_set(self, task_data: Dataset, dist_args: Optional[dict] = None,
                                model: torch.nn.Module = None, batch_size=1, **kwargs):
        
        distributed = dist_args is not None
        if distributed:
            device = torch.device(f"cuda:{dist_args['gpu']}")
            rank = dist_args['rank']
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rank = 0
        new_class_labels = task_data.cur_task
        model.eval()

        with task_data.disable_augmentations(): # disable augmentations then enable them (if they were already enabled)
            with torch.no_grad():
                for class_label in new_class_labels:
                    class_data_indices = task_data.get_image_indices_by_cla(class_label, self.max_mems_pool_size)
                    if distributed:
                        device = torch.device(f"cuda:{dist_args['gpu']}")
                        class_data_indices_to_broadcast = torch.from_numpy(class_data_indices).to(device)
                        dist.broadcast(class_data_indices_to_broadcast, 0)
                        class_data_indices = class_data_indices_to_broadcast.cpu().numpy()
                    sampler = SubsetSampler(class_data_indices)
                    class_loader = DataLoader(task_data, batch_size=batch_size, sampler=sampler)
                    latent_vectors = []
                    for minibatch in class_loader:
                        images = minibatch[0].to(device)
                        output, out_latent = model.forward_net(images)
                        out_latent = out_latent.detach()
                        out_latent = F.normalize(out_latent, p=2, dim=-1)
                        latent_vectors.append(out_latent)
                    latent_vectors = torch.cat(latent_vectors, dim=0)
                    class_mean = torch.mean(latent_vectors, dim=0)

                    chosen_exemplars_ind = []
                    exemplars_mean = torch.zeros_like(class_mean)
                    while len(chosen_exemplars_ind) < min(self.n_mems_per_cla, len(class_data_indices)):
                        potential_exemplars_mean = (exemplars_mean.unsqueeze(0) * len(chosen_exemplars_ind) + latent_vectors) \
                                                   / (len(chosen_exemplars_ind) + 1)
                        distance = (class_mean.unsqueeze(0) - potential_exemplars_mean).norm(dim=-1)
                        shuffled_index = torch.argmin(distance).item()
                        exemplars_mean = potential_exemplars_mean[shuffled_index, :].clone()
                        exemplar_index = class_data_indices[shuffled_index]
                        chosen_exemplars_ind.append(exemplar_index)
                        latent_vectors[shuffled_index, :] = float("inf")

                    for image_index in chosen_exemplars_ind:
                        image, label1, label2 = task_data.get_item(image_index)
                        if label2 != NO_LABEL_PLACEHOLDER:
                            warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
                        self.add_sample(class_label, image, (label1, label2), rank=rank)
