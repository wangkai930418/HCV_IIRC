import argparse
import copy
import importlib
import logging
import os
import sys
import random
from math import ceil
import numpy as np
import torch
import torchvision
from ml_logger import logbook as ml_logbook
import time
import torch.multiprocessing as mp
import torch.distributed as dist
import pickle as pkl

from sklearn.metrics import multilabel_confusion_matrix
import sklearn.metrics as skm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from iirc.datasets_loader import get_lifelong_datasets
from iirc.utils.utils import print_msg
from iirc.definitions import CIL_SETUP, IIRC_SETUP
import lifelong_methods.utils
import lifelong_methods
import experiments.utils as utils
from experiments.prepare_config import prepare_config
from experiments.train import task_train, tasks_eval, task_eval_superclass, modified_tasks_eval
# from utils import modify_dataset


def get_transforms(dataset_name):
    essential_transforms_fn = None
    augmentation_transforms_fn = None
    if "cifar100" in dataset_name:
        essential_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
    elif "imagenet" in dataset_name:
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        essential_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    return essential_transforms_fn, augmentation_transforms_fn


def print_task_info(lifelong_dataset):
    class_names_samples = {class_: 0 for class_ in lifelong_dataset.cur_task}
    for idx in range(len(lifelong_dataset)):
        labels = lifelong_dataset.get_labels(idx)
        for label in labels:
            if label in class_names_samples.keys():
                class_names_samples[label] += 1
    print_msg(f"Task {lifelong_dataset.cur_task_id} number of samples: {len(lifelong_dataset)}")
    for class_name, num_samples in class_names_samples.items():
        print_msg(f"{class_name} is present in {num_samples} samples")


def main_worker_1(gpu, config: dict, dist_args: dict = None, test_task_id=0, Threshold=0.58, prior=False):
    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
        rank = 0
        print_msg(f"Using GPU: {gpu}")
    else:
        device = config["device"]
        rank = 0
        print_msg(f"using {config['device']}\n")

    checkpoint = None
    non_loadable_attributes = ["logging_path", "dataset_path", "batch_size"]
    temp = {key: val for key, val in config.items() if key in non_loadable_attributes}
    ##############################################################
    #########################################
    ######################################## loading from task id. and  --run_id 42177 --reduce_lr_on_plateau

    # test_task_id = 1
    checkpoint_path = os.path.join(config['logging_path'], f"task_{test_task_id-1}_model")

    # checkpoint_path = os.path.join(config['logging_path'], 'latest_model')
    ################################################
    #########################################
    ######################################## loading from task id

    json_logs_file_name = 'jsonlogs.jsonl'
    if os.path.isfile(checkpoint_path):
        logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                            filemode='a+',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
        
        print_msg(f"\n\nLoading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        ############## load checkpoint configurations
        config = checkpoint['config']
        for key in non_loadable_attributes:
            config[key] = temp[key]
        
        print_msg(f"Loaded the checkpoint successfully")

        if rank == 0:
            print_msg(f"Resuming from task {config['cur_task_id']} epoch {config['task_epoch']}")
            # Remove logs related to traing after the checkpoint was saved
            utils.remove_extra_logs(config['cur_task_id'], config['task_epoch'],
                                    os.path.join(config['logging_path'], json_logs_file_name))
            
        else:
            dist.barrier()
    else:
        if rank == 0:
            os.makedirs(config['logging_path'], exist_ok=True)
            if os.path.isfile(os.path.join(config['logging_path'], json_logs_file_name)):
                os.remove(os.path.join(config['logging_path'], json_logs_file_name))
            logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                                filemode='w',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)
            
        else:
            dist.barrier()
            logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)

    torch.random.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config["seed"])

    wandb_config = None

    logbook_config = ml_logbook.make_config(
        logger_dir=config['logging_path'],
        filename=json_logs_file_name,
        create_multiple_log_files=False,
        wandb_config=wandb_config,
    )
    logbook = ml_logbook.LogBook(config=logbook_config)

    essential_transforms_fn, augmentation_transforms_fn = get_transforms(config['dataset'])
    lifelong_datasets, tasks, class_names_to_idx = \
        get_lifelong_datasets(config['dataset'], dataset_root=config['dataset_path'],
                              tasks_configuration_id=config["tasks_configuration_id"],
                              essential_transforms_fn=essential_transforms_fn,
                              augmentation_transforms_fn=augmentation_transforms_fn, cache_images=False,
                              joint=config["joint"])

    metadata = checkpoint['metadata']
    config['patience']=10
    # Assert that methods files lie in the folder "methods"
    method = importlib.import_module('lifelong_methods.methods.' + config["method"])
    model = method.Model(metadata["n_cla_per_tsk"], metadata["class_names_to_idx"], config)

    buffer_dir = None
    map_size = None
    buffer = method.Buffer(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model.to(device)
        model.net = torch.nn.parallel.DistributedDataParallel(model.net, device_ids=[gpu])
    else:
        model.to(config["device"])

    # If loading a checkpoint, load the corresponding state_dicts
    if checkpoint is not None:
        lifelong_methods.utils.load_model(checkpoint, model, buffer, lifelong_datasets)
        print_msg(f"Loaded the state dicts successfully")
        starting_task = config["cur_task_id"]
    else:
        starting_task = 0

    #######################################################################
    #############################################
    #############################################
    for lifelong_dataset in lifelong_datasets.values():
            lifelong_dataset.enable_complete_information_mode()

    # prior=prior
    if prior:
        super_class_index=[0,1,2,3,4,5,6,7,8,9]
    else:
        super_class_index=None

    model.net.eval()

    verified_super_key,all_keys,all_probs,all_pos_super_key = task_eval_superclass(
        model, lifelong_datasets["train"], test_task_id, config, metadata, logbook=logbook,
        dataset_type="valid", dist_args=dist_args, Threshold=Threshold
    )
    return verified_super_key,all_keys,all_probs,all_pos_super_key, metadata["class_names_to_idx"]


def main_worker_2(gpu, config: dict, dist_args: dict = None, verified_super_key=None, test_task_id=0, prior=False):

    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
        rank = 0
        print_msg(f"Using GPU: {gpu}")
    else:
        device = config["device"]
        rank = 0
        print_msg(f"using {config['device']}\n")

    checkpoint = None
    non_loadable_attributes = ["logging_path", "dataset_path", "batch_size"]
    temp = {key: val for key, val in config.items() if key in non_loadable_attributes}
    ##############################################################
    #########################################
    ######################################## loading from task id. and  --run_id 42177 --reduce_lr_on_plateau

    # test_task_id = 1
    checkpoint_path = os.path.join(config['logging_path'], f"task_{test_task_id}_model")

    # checkpoint_path = os.path.join(config['logging_path'], 'latest_model')
    ################################################
    #########################################
    ######################################## loading from task id

    json_logs_file_name = 'jsonlogs.jsonl'
    if os.path.isfile(checkpoint_path):
        logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                            filemode='a+',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
        
        print_msg(f"\n\nLoading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        ############## load checkpoint configurations
        config = checkpoint['config']
        for key in non_loadable_attributes:
            config[key] = temp[key]
        
        print_msg(f"Loaded the checkpoint successfully")

        if rank == 0:
            print_msg(f"Resuming from task {config['cur_task_id']} epoch {config['task_epoch']}")
            # Remove logs related to traing after the checkpoint was saved
            utils.remove_extra_logs(config['cur_task_id'], config['task_epoch'],
                                    os.path.join(config['logging_path'], json_logs_file_name))
            
        else:
            dist.barrier()
    else:
        if rank == 0:
            os.makedirs(config['logging_path'], exist_ok=True)
            if os.path.isfile(os.path.join(config['logging_path'], json_logs_file_name)):
                os.remove(os.path.join(config['logging_path'], json_logs_file_name))
            logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                                filemode='w',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)
            
        else:
            dist.barrier()
            logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)
    config['patience']=10
    torch.random.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config["seed"])

    wandb_config = None

    logbook_config = ml_logbook.make_config(
        logger_dir=config['logging_path'],
        filename=json_logs_file_name,
        create_multiple_log_files=False,
        wandb_config=wandb_config,
    )
    logbook = ml_logbook.LogBook(config=logbook_config)

    essential_transforms_fn, augmentation_transforms_fn = get_transforms(config['dataset'])
    lifelong_datasets, tasks, class_names_to_idx = \
        get_lifelong_datasets(config['dataset'], dataset_root=config['dataset_path'],
                              tasks_configuration_id=config["tasks_configuration_id"],
                              essential_transforms_fn=essential_transforms_fn,
                              augmentation_transforms_fn=augmentation_transforms_fn, cache_images=False,
                              joint=config["joint"])

    metadata = checkpoint['metadata']

    # Assert that methods files lie in the folder "methods"
    method = importlib.import_module('lifelong_methods.methods.' + config["method"])
    model = method.Model(metadata["n_cla_per_tsk"], metadata["class_names_to_idx"], config)

    buffer_dir = None
    map_size = None
    buffer = method.Buffer(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model.to(device)
        model.net = torch.nn.parallel.DistributedDataParallel(model.net, device_ids=[gpu])
    else:
        model.to(config["device"])

    # If loading a checkpoint, load the corresponding state_dicts
    if checkpoint is not None:
        lifelong_methods.utils.load_model(checkpoint, model, buffer, lifelong_datasets)
        print_msg(f"Loaded the state dicts successfully")
        starting_task = config["cur_task_id"]
    else:
        starting_task = 0

    #######################################################################
    #############################################
    #############################################
    for lifelong_dataset in lifelong_datasets.values():
            lifelong_dataset.enable_complete_information_mode()

    # prior=False
    if prior:
        super_class_index=[0,1,2,3,4,5,6,7,8,9]
    else:
        super_class_index=[]

    prev_verified_super_key=[]

    if test_task_id > 1:
        save_super_class = os.path.join(config['logging_path'], 'super_' + str(test_task_id-1) + ".pkl")
        with open(save_super_class, 'rb') as f:
            super_class_index = pkl.load(f)

        take_verified_super_key = os.path.join(config['logging_path'], str(test_task_id-1) + ".pkl")
        with open(take_verified_super_key, 'rb') as f:
            prev_verified_super_key = pkl.load(f)

        for ind in range(len(prev_verified_super_key)):
            prev_verified_super_key[ind]=np.append(prev_verified_super_key[ind],[0,0,0,0,0])

    model.net.eval()

    for ind in range(len(verified_super_key)):
        find_where = np.where(verified_super_key[ind][:-5] == 1)[0]
        if len(find_where) > 0:
            if find_where[0] not in super_class_index:
                super_class_index.append(np.where(verified_super_key[ind][:-5] == 1)[0][0])

    if len(prev_verified_super_key)>0:
        verified_super_key = prev_verified_super_key + verified_super_key

    save_verified_super_key = os.path.join(config['logging_path'], str(test_task_id) + ".pkl")
    with open(save_verified_super_key, 'wb') as f:
        pkl.dump(verified_super_key, f)

    save_super_class = os.path.join(config['logging_path'], 'super_' + str(test_task_id) + ".pkl")
    with open(save_super_class, 'wb') as f:
        pkl.dump(super_class_index, f)

    verified_super_key = torch.from_numpy(np.vstack(verified_super_key))
    look_table=torch.eye(verified_super_key.shape[1],dtype=torch.int32)

    # for i in range(0,verified_super_key.shape[1]):
    #     for j in range(len(verified_super_key)):
    #         if (look_table[i] & verified_super_key[j]).sum() == 1:
    #             #### loop over top tiers
    #             look_table[i]=verified_super_key[j]
    #             break

    # for i in range(0,verified_super_key.shape[1]):
    for j in range(len(verified_super_key)):
        correspond_super_label=verified_super_key[j].nonzero()[0]
        ind=verified_super_key[j].nonzero()[-1]
        look_table[ind] = look_table[correspond_super_label] | verified_super_key[j]

        # if (look_table[i] & verified_super_key[j]).sum() == 1:
        #     #### loop over top tiers
        #     correspond_super_label=verified_super_key[j].nonzero()[0]
        #     look_table[correspond_super_label]
        #     look_table[i] = look_table[correspond_super_label] | verified_super_key[j]
        #     break

    look_table=look_table.bool().cuda()
    metrics_dict = modified_tasks_eval(model, lifelong_datasets["test"],
               test_task_id, config, metadata,
               logbook=logbook, dataset_type="test",
                        look_table=look_table)
    # modified_tasks_eval(model, lifelong_datasets["test"],
    #            test_task_id, config, metadata,
    #            logbook=logbook, dataset_type="test",
    #                     look_table=None)
    return metrics_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="iirc_cifar100",
                        choices=["incremental_cifar100", "iirc_cifar100", "incremental_imagenet_full",
                                 "incremental_imagenet_lite", "iirc_imagenet_full", "iirc_imagenet_lite"])

    parser.add_argument('--epochs_per_task', type=int, default=140,
                        help="The number of epochs per task. This number is multiplied by 2 for the first task.")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--dataset_path', type=str, default="./cifar100")
    parser.add_argument('--logging_path_root', type=str, default="results",
                        help="The directory where the logs and results will be saved")
    parser.add_argument('--wandb_project', type=str, default=None)
    #########################################################
    parser.add_argument('--run_id', type=int, default=None)
    # parser.add_argument('--run_id', type=int, default=None)
    #########################################################
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--n_layers', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=6,
                        help="Number of workers used to fetch the data for the dataloader")
    parser.add_argument('--group', type=str, default="final_cifar_experiments",
                        help="The parent folder of the experiment results, so as to group related experiments easily")
    # Parameters for creating the tasks
    parser.add_argument('--tasks_configuration_id', type=int, default=0, choices=range(0, 10),
                        help="The task configuration id. Ignore for joint training")
    # The training method  ###############################################
    parser.add_argument('--method', type=str, default="icarl_cnn",
                        choices=["finetune", "mask_seen_classes", "lucir", "agem", "icarl_cnn",
                                 "icarl_norm","icarl_cnn_celoss"])
    ################################################################################
    parser.add_argument('--complete_info', action='store_true',
                        help='use the complete information during training (a multi-label setting)')
    parser.add_argument('--incremental_joint', action='store_true',
                        help="keep all data from previous tasks, while updating their labels as per the observed "
                             "classes (use only with complete_info and without buffer)")
    parser.add_argument('--joint', action='store_true',
                        help="load all classes during the first task. This option ignores the tasks_configuration_id "
                             "(use only with complete_info and without buffer)")
    # The optimizer parameters
    parser.add_argument('--optimizer', type=str, default="momentum", choices=["adam", "momentum"])
    parser.add_argument('--lr', type=float, default=1.0, help="The initial learning rate for each task")
    parser.add_argument('--lr_gamma', type=float, default=.1,
                        help="The multiplicative factor for learning rate decay at the epochs specified")
    parser.add_argument('--lr_schedule', nargs='+', type=int, default=[80,110],
                        help="the epochs per task at which to multiply the current learning rate by lr_gamma "
                             "(resets after each task). This setting is ignored if reduce_lr_on_plateau is specified")
    #####################################################################
    parser.add_argument('--reduce_lr_on_plateau', action='store_true',
                        help='reduce the lr on plateau based on the validation performance metric')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # Model selection and validation set
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help="The number of epochs within each task after which the checkpoint is updated. When a task "
                             "is finished, the checkpoint is updated anyways, so set to 0 for checkpointing only after "
                             "each task")
    #########################################################################
    parser.add_argument('--use_best_model', action='store_true',
                        help='use the best model after training each task based on the best task validation accuracy')
    #########################################################################
    parser.add_argument('--save_each_task_model', action='store_true',
                        help='save the model after each task')
    # The buffer parameters
    parser.add_argument('--total_n_memories', type=int, default=-1,
                        help="The total replay buffer size, which is divided by the observed number of classes to get "
                             "the number of memories kept per task, note that the number of memories per task here is "
                             "not fixed but rather decreases as the number of tasks observed increases (with a minimum "
                             "of 1). If n_memories_per_class is set to a value greater than -1, the "
                             "n_memories_per_class is used instead.")
    parser.add_argument('--n_memories_per_class', type=int, default=20,
                        help="The number of samples to keep from each class, if set to -1, the total_n_memories "
                             "argument is used instead")
    parser.add_argument('--buffer_sampling_multiplier', type=float, default=1.0,
                        help="A multiplier for sampling from the buffer more/less times than the size of the buffer "
                             "(for example a multiplier of 2 samples from the buffer (with replacement) twice its size "
                             "per epoch)")
    parser.add_argument('--memory_strength', type=float, default=1.0,
                        help="a weight to be multiplied by the loss from the buffer")
    parser.add_argument('--max_mems_pool_per_class', type=int, default=1e5,
                        help="Maximum size of the samples pool per class from which the buffer chooses the exemplars, "
                             "use -1 for choosing from the whole class samples.")

    # LUCIR Hyperparameters
    parser.add_argument('--lucir_lambda', type=float, default=5.0,
                        help="a weight to be multiplied by the distillation loss (only for the LUCIR method)")
    parser.add_argument('--lucir_margin_1', type=float, default=0.5,
                        help="The 1st margin used with the margin ranking loss for in the LUCIR method")
    parser.add_argument('--lucir_margin_2', type=float, default=0.5,
                        help="The 2nd margin used with the margin ranking loss for in the LUCIR method")

    # Distributed arguments
    parser.add_argument('--num_nodes', type=int, default=1,
                        help="num of nodes to use")
    parser.add_argument('--node_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url', default="env://", type=str,
                        help='node rank for distributed training')

    args = parser.parse_args()
    print(args)
    config = prepare_config(args)
    if "iirc" in config["dataset"]:
        config["setup"] = IIRC_SETUP
    else:
        config["setup"] = CIL_SETUP

    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config["ngpus_per_node"] = torch.cuda.device_count()
    print_msg(f"number of gpus per node: {config['ngpus_per_node']}")

    #### Task 1: predict superclasses
    ####
    verified_super_key,all_keys,all_probs,all_pos_super_key, class_names_to_idx=\
                            main_worker_1(None, config, None,1)
    idx_to_class_names = dict((y,x) for x,y in class_names_to_idx.items())

    ##########################  compute and print accuracy curves #####################################

    prior = False
    f = open('./outputs/threshold_'+str(prior)+'.txt', 'w')
    
    for Threshold in [0.6]:

        acc_list=[Threshold,]
    
        for test_task_id in range(1,22):
            verified_super_key,all_keys,all_probs, _,class_names_to_idx=\
                main_worker_1(None, config, None,test_task_id,Threshold=Threshold, prior=prior)
            metrics_dict, original_pred_list, modified_pred_list, GT = \
                main_worker_2(None, config, None, verified_super_key,test_task_id,prior=prior)
            acc_list.append(metrics_dict['average_test_modified_jaccard'])
    
        print(acc_list, file=f)
