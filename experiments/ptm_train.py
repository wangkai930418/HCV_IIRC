import os
import time
import torch
import torch.utils.data as data
import torch.distributed as dist
from copy import deepcopy
import numpy as np


from iirc.utils.utils import print_msg
import lifelong_methods.utils
from lifelong_methods.buffer.buffer import TaskDataMergedWithBuffer
from lifelong_methods.utils import transform_labels_names_to_vector
from experiments import utils, metrics


def epoch_train(model, dataloader, config, metadata, gpu=None, rank=0,verified_super_key_dict=None,super_class_index=None,sub_class_index=None):
    train_loss = 0
    train_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0.}
    data_len = 0
    ##################
    class_names_to_idx = metadata["class_names_to_idx"]
    num_seen_classes = len(model.seen_classes)
    model.net.train()

    minibatch_i = 0
    for minibatch in dataloader:
        labels_names = list(zip(minibatch[1], minibatch[2]))
        labels = transform_labels_names_to_vector(
            labels_names, num_seen_classes, class_names_to_idx
        )

        if gpu is None:
            images = minibatch[0].to(config["device"], non_blocking=True)
            labels = labels.to(config["device"], non_blocking=True)
        else:
            images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

        if len(minibatch) > 3:
            if gpu is None:
                in_buffer = minibatch[3].to(config["device"], non_blocking=True)
            else:
                in_buffer = minibatch[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
        else:
            in_buffer = None

        ###########################################################
        # labels=labels.bool()
        if verified_super_key_dict is not None:
            for item_id in range(len(labels)):
                for key in verified_super_key_dict.keys():
                    if labels[item_id][key] ==1:
                        labels[item_id][verified_super_key_dict[key]]=1
                        
                # pred_dup = labels[item_id].view(1, -1).repeat(len(verified_super_key_dict), 1)
                # xor_matrix = pred_dup ^ verified_super_key_dict

                # if (xor_matrix.sum(1) == 0).sum() == 0:  # no match
                #     # most_match_table_item=xor_matrix.sum(1).argmin()
                #     # #############################################
                #     min_value = xor_matrix.sum(1).min()
                #     min_indexes = torch.where(xor_matrix.sum(1) == min_value)[0]
                #     ##########################
                #     most_match_table_item = min_indexes[torch.randperm(len(min_indexes))[0]]
                #     ###############################################
                #     # most_match_table_item=min_indexes[0]
                #     ################################################
                #
                #     labels[item_id] = verified_super_key_dict[most_match_table_item]

        predictions, loss = model.observe(images, labels, in_buffer, train=True,
                                          super_class_index=super_class_index,
                                          sub_class_index=sub_class_index)

        labels = labels.bool()
        train_loss += loss * images.shape[0]
        train_metrics['jaccard_sim'] += metrics.jaccard_sim(predictions, labels) * images.shape[0]
        train_metrics['modified_jaccard'] += metrics.modified_jaccard_sim(predictions, labels) * images.shape[0]
        train_metrics['strict_acc'] += metrics.strict_accuracy(predictions, labels) * images.shape[0]
        train_metrics['recall'] += metrics.recall(predictions, labels) * images.shape[0]
        data_len += images.shape[0]

        if minibatch_i == 0:
            print_msg(
                f"rank {rank}, max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB, "
                f"current memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB\n")
        minibatch_i += 1

    train_loss /= data_len
    train_metrics['jaccard_sim'] /= data_len
    train_metrics['modified_jaccard'] /= data_len
    train_metrics['strict_acc'] /= data_len
    train_metrics['recall'] /= data_len
    return train_loss, train_metrics


def evaluate(model, dataloader, config, metadata, test_mode=False, gpu=None,verified_super_key_dict=None,super_class_index=None,sub_class_index=None):
    valid_loss = 0
    valid_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0.}
    data_len = 0
    class_names_to_idx = metadata["class_names_to_idx"]
    num_seen_classes = len(model.seen_classes)
    model.net.eval()
    #########################################
    # valid_metrics['super_modified_jaccard']=0
 #########################################
    with torch.no_grad():
        for minibatch in dataloader:
            labels_names = list(zip(minibatch[1], minibatch[2]))
            labels = transform_labels_names_to_vector(
                labels_names, num_seen_classes, class_names_to_idx
            )

            if gpu is None:
                images = minibatch[0].to(config["device"], non_blocking=True)
                labels = labels.to(config["device"], non_blocking=True)
            else:
                images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
                labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

            if len(minibatch) > 3:
                if gpu is None:
                    in_buffer = minibatch[3].to(config["device"], non_blocking=True)
                else:
                    in_buffer = minibatch[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            else:
                in_buffer = None

            if verified_super_key_dict is not None:
                for item_id in range(len(labels)):
                    for key in verified_super_key_dict.keys():
                        if labels[item_id][key] == 1:
                            labels[item_id][verified_super_key_dict[key]] = 1

            if not test_mode:
                predictions, loss = model.observe(images, labels, in_buffer, train=False)
                valid_loss += loss * images.shape[0]
            else:
                predictions = model(images)

            labels = labels.bool()
            ###################
            # mask=(labels[:,:10].sum(dim=1)==1)
            # if torch.sum(mask)>0:
            #     valid_metrics['super_modified_jaccard']+= \
            #         metrics.super_modified_jaccard_sim\
            #             (predictions[mask,:10], labels[mask,:10]) \
            #             * images.shape[0]
            ##################
            valid_metrics['jaccard_sim'] += metrics.jaccard_sim(predictions, labels) * images.shape[0]
            valid_metrics['modified_jaccard'] += metrics.modified_jaccard_sim(predictions, labels) * images.shape[0]
            valid_metrics['strict_acc'] += metrics.strict_accuracy(predictions, labels) * images.shape[0]
            valid_metrics['recall'] += metrics.recall(predictions, labels) * images.shape[0]
            data_len += images.shape[0]
        valid_loss /= data_len
        valid_metrics['jaccard_sim'] /= data_len
        valid_metrics['modified_jaccard'] /= data_len
        valid_metrics['strict_acc'] /= data_len
        valid_metrics['recall'] /= data_len
        # valid_metrics['super_modified_jaccard'] /=data_len
    return valid_loss, valid_metrics


def task_train(model, buffer, lifelong_datasets, config, metadata, logbook, dist_args=None,
               verified_super_key_dict=None,super_class_index=None,sub_class_index=None):
    distributed = dist_args is not None
    if distributed:
        gpu = dist_args["gpu"]
        rank = dist_args["rank"]
    else:
        gpu = None
        rank = 0

    best_checkpoint = {
        "model_state_dict": deepcopy(model.method_state_dict()),
        "best_modified_jaccard": 0
    }
    best_checkpoint_file = os.path.join(config['logging_path'], "best_checkpoint")
    if config['use_best_model']:
        if config['task_epoch'] > 0 and os.path.exists(best_checkpoint_file):
            if distributed:
                best_checkpoint = torch.load(best_checkpoint_file, map_location=f"cuda:{dist_args['gpu']}")
            else:
                best_checkpoint = torch.load(best_checkpoint_file)

    task_train_data = lifelong_datasets['train']
    if config["method"] == "agem":
        bsm = 0.0
        task_train_data_with_buffer = TaskDataMergedWithBuffer(buffer, task_train_data, buffer_sampling_multiplier=bsm)
    else:
        bsm = config["buffer_sampling_multiplier"]
        task_train_data_with_buffer = TaskDataMergedWithBuffer(buffer, task_train_data, buffer_sampling_multiplier=bsm)
    task_valid_data = lifelong_datasets['intask_valid']
    cur_task_id = task_train_data.cur_task_id

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            task_train_data_with_buffer, num_replicas=dist_args["world_size"], rank=rank)
    else:
        train_sampler = None

    train_loader = data.DataLoader(
        task_train_data_with_buffer, batch_size=config["batch_size"], shuffle=(train_sampler is None),
        num_workers=config["num_workers"], pin_memory=True, sampler=train_sampler
    )
    valid_loader = data.DataLoader(
        task_valid_data, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
        pin_memory=True
    )

    if cur_task_id == 0:
        num_epochs = config['epochs_per_task'] * 2
        print_msg(f"Training for {num_epochs} epochs for the first task (double that the other tasks)")
    else:
        num_epochs = config['epochs_per_task']
    print_msg(f"Starting training of task {cur_task_id} epoch {config['task_epoch']} till epoch {num_epochs}")
    for epoch in range(config['task_epoch'], num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        start_time = time.time()
        log_dict = {}
        train_loss, train_metrics = epoch_train(model, train_loader, config, metadata, gpu, rank,
                                                verified_super_key_dict=verified_super_key_dict,
                                                super_class_index=super_class_index,
                                                sub_class_index=sub_class_index)

        log_dict[f"train_loss_{cur_task_id}"] = train_loss
        for metric in train_metrics.keys():
            log_dict[f"train_{metric}_{cur_task_id}"] = train_metrics[metric]

        valid_loss, valid_metrics = evaluate(model, valid_loader, config, metadata, test_mode=False, gpu=gpu,
                                             verified_super_key_dict=verified_super_key_dict,
                                             super_class_index=super_class_index,
                                             sub_class_index=sub_class_index)

        log_dict[f"valid_loss_{cur_task_id}"] = valid_loss
        for metric in valid_metrics.keys():
            log_dict[f"valid_{metric}_{cur_task_id}"] = valid_metrics[metric]

        model.net.eval()
        model.consolidate_epoch_knowledge(log_dict[f"valid_modified_jaccard_{cur_task_id}"],
                                          task_data=task_train_data,
                                          device=config["device"],
                                          batch_size=config["batch_size"])
        # If using the lmdb database, close it and open a new environment to kill active readers
        buffer.reset_lmdb_database()

        if config['use_best_model']:
            if log_dict[f"valid_modified_jaccard_{cur_task_id}"] >= best_checkpoint["best_modified_jaccard"]:
                best_checkpoint["best_modified_jaccard"] = log_dict[f"valid_modified_jaccard_{cur_task_id}"]
                best_checkpoint["model_state_dict"] = deepcopy(model.method_state_dict())
            log_dict[f"best_valid_modified_jaccard_{cur_task_id}"] = best_checkpoint["best_modified_jaccard"]

        if distributed:
            dist.barrier() #to calculate the time based on the slowest gpu
        end_time = time.time()
        log_dict[f"elapsed_time"] =  round(end_time - start_time, 2)

        if rank == 0:
            utils.log(epoch, cur_task_id, log_dict, logbook)
            if distributed:
                dist.barrier()
                log_dict["rank"] = rank
                print_msg(log_dict)
        else:
            dist.barrier()
            log_dict["rank"] = rank
            print_msg(log_dict)

        # Checkpointing
        config["task_epoch"] = epoch + 1
        if (config["task_epoch"] % config['checkpoint_interval']) == 0 and rank == 0:
            print_msg("Saving latest checkpoint")
            save_file = os.path.join(config['logging_path'], "latest_model")
            lifelong_methods.utils.save_model(save_file, config, metadata, model, buffer, lifelong_datasets)
            if config['use_best_model']:
                print_msg("Saving best checkpoint")
                torch.save(best_checkpoint, best_checkpoint_file)

    # reset the model parameters to the best performing model
    if config['use_best_model']:
        model.load_method_state_dict(best_checkpoint["model_state_dict"])


def tasks_eval(model, dataset, cur_task_id, config, metadata, logbook, dataset_type="valid", dist_args=None, verified_super_key_dict=None):
    """log the accuracies of the new model on all observed tasks
    :param metadata:
    """
    assert dataset.complete_information_mode is True

    distributed = dist_args is not None
    if distributed:
        gpu = dist_args["gpu"]
        rank = dist_args["rank"]
    else:
        gpu = None
        rank = 0

    metrics_dict = {}
    for task_id in range(cur_task_id + 1):
    # for task_id in range(1,cur_task_id + 1):
        dataset.choose_task(task_id)
        dataloader = data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
        )
        _, metrics = evaluate(model, dataloader, config, metadata, test_mode=True, gpu=gpu)
        for metric in metrics.keys():
            metrics_dict[f"task_{task_id}_{dataset_type}_{metric}"] = metrics[metric]

    dataset.load_tasks_up_to(cur_task_id)
    dataloader = data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )
    _, metrics = evaluate(model, dataloader, config, metadata,
                          test_mode=True, gpu=gpu,
                          verified_super_key_dict=verified_super_key_dict)

    for metric in metrics.keys():
        metrics_dict[f"average_{dataset_type}_{metric}"] = metrics[metric]

    if rank == 0:
        utils.log_task(cur_task_id, metrics_dict, logbook)
        if distributed:
            dist.barrier()
            metrics_dict["rank"] = rank
            print_msg(metrics_dict)
    else:
        dist.barrier()
        metrics_dict["rank"] = rank
        print_msg(metrics_dict)


def task_eval_superclass(model, dataset, cur_task_id, config, metadata, logbook,
                         dataset_type="valid", dist_args=None,Threshold=0.58, num_class_per_task=5):
    """log the accuracies of the new model on all observed tasks
    :param metadata:
    """
    assert dataset.complete_information_mode is True

    distributed = dist_args is not None
    if distributed:
        gpu = dist_args["gpu"]
        rank = dist_args["rank"]
    else:
        gpu = None
        rank = 0

    metrics_dict = {}
    # for task_id in range(cur_task_id + 1):
    for task_id in range(cur_task_id,cur_task_id + 1):
        dataset.choose_task(task_id)
        dataloader = data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
        )
        verified_super_key= eval_superclass(model, dataloader, config, metadata,
                                            test_mode=True, gpu=gpu, Threshold=Threshold, num_class_per_task=num_class_per_task)

    return verified_super_key


def task_eval_superclass_in_training(model, dataset, cur_task_id, config, metadata, logbook,
                         dataset_type="valid", dist_args=None,Threshold=0.6, num_class_per_task=5):
    """log the accuracies of the new model on all observed tasks
    :param metadata:
    """
    # assert dataset.complete_information_mode is True

    distributed = dist_args is not None
    if distributed:
        gpu = dist_args["gpu"]
        rank = dist_args["rank"]
    else:
        gpu = None
        rank = 0

    metrics_dict = {}
    # for task_id in range(cur_task_id + 1):
    for task_id in range(cur_task_id,cur_task_id + 1):
        dataset.choose_task(task_id)
        dataloader = data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
        )
        verified_super_key= eval_superclass_in_training(model, dataloader, config, metadata,
                                            test_mode=True, gpu=gpu, Threshold=Threshold, num_class_per_task=num_class_per_task)

    return verified_super_key


def eval_superclass(model, dataloader, config, metadata, test_mode=False, gpu=None, Threshold=0.58, num_class_per_task=5):
    valid_loss = 0
    valid_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0.}
    data_len = 0
    class_names_to_idx = metadata["class_names_to_idx"]
    num_seen_classes = len(model.seen_classes) + num_class_per_task
    model.net.eval()

    with torch.no_grad():
        super_pred_dict={}

        for minibatch in dataloader:
            labels_names = list(zip(minibatch[1], minibatch[2]))
            labels = transform_labels_names_to_vector(
                labels_names, num_seen_classes, class_names_to_idx
            )

            if gpu is None:
                images = minibatch[0].to(config["device"], non_blocking=True)
                labels = labels.to(config["device"], non_blocking=True)
            else:
                images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
                labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

            if len(minibatch) > 3:
                if gpu is None:
                    in_buffer = minibatch[3].to(config["device"], non_blocking=True)
                else:
                    in_buffer = minibatch[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            else:
                in_buffer = None

            if not test_mode:
                predictions, loss = model.observe(images, labels, in_buffer, train=False)
                valid_loss += loss * images.shape[0]
            else:
                predictions, output = model(images, return_output=True)

            outputs = output[:, :(num_seen_classes-num_class_per_task)]
            super_predictions = outputs > 0.0
            # super_labels=labels[:, :10].bool()

            for index in range(len(labels)):
                complete_label=labels[index]
                complete_label=complete_label.data.cpu().numpy()
                # value=bool2int(complete_label.astype(np.int32))
                value=np.array2string(complete_label.astype(np.int32))
                # value=hot_to_value(complete_label)
                if not value in super_pred_dict.keys():
                    super_pred_dict[value]=[]

                super_pred_dict[value].append(super_predictions[index].data.cpu().detach().numpy())

            # np.vstack(super_pred_dict[16386]).astype(np.int64).sum(axis=0)  16386, 4098, 1024, 8192, 2048

        verified_super_key=[]
        all_pos_super_key=[]
        all_keys=[]
        all_probs=[]
        for key in super_pred_dict.keys():
            probs=np.vstack(super_pred_dict[key]).astype(np.int64).sum(axis=0)
            prob= float(probs.max()) / float(len(super_pred_dict[key]))

            max_label=np.zeros(len(probs),dtype=np.int64)
            max_label[probs.argmax()]=1
            max_string=np.fromstring(key[1:-1], dtype=np.int, sep=' ')
            max_string=np.hstack((max_label,max_string[-num_class_per_task:]))
            all_pos_super_key.append(max_string)

            if prob > Threshold:
                preset_label=np.zeros(len(probs),dtype=np.int64)
                preset_label[probs.argmax()]=1
                ########### ::-1 reverse
                label_string=np.fromstring(key[1:-1], dtype=np.int, sep=' ')
                label_string=np.hstack((preset_label,label_string[-num_class_per_task:]))
                # label_string=np.binary_repr(key)[::-1]
                # if len(label_string)<len(complete_label):
                #     add_length=len(complete_label)-len(label_string)
                #     label_string=label_string+'0'*add_length

                # label_string=np.array(list(label_string), dtype=int)

                verified_super_key.append(label_string)

            # print(key)
            all_keys.append(key)
            all_probs.append(prob)
            # print(prob)
            #### later print all of them for comparison
        return verified_super_key,all_keys,all_probs, all_pos_super_key


def eval_superclass_in_training(model, dataloader, config, metadata, test_mode=False, gpu=None, Threshold=0.6, num_class_per_task=5):
    valid_loss = 0
    valid_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0.}
    data_len = 0
    class_names_to_idx = metadata["class_names_to_idx"]
    num_seen_classes = len(model.seen_classes)
    model.net.eval()

    with torch.no_grad():
        super_pred_dict={}

        for minibatch in dataloader:
            labels_names = list(zip(minibatch[1], minibatch[2]))
            labels = transform_labels_names_to_vector(
                labels_names, num_seen_classes, class_names_to_idx
            )

            if gpu is None:
                images = minibatch[0].to(config["device"], non_blocking=True)
                labels = labels.to(config["device"], non_blocking=True)
            else:
                images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
                labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

            if len(minibatch) > 3:
                if gpu is None:
                    in_buffer = minibatch[3].to(config["device"], non_blocking=True)
                else:
                    in_buffer = minibatch[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            else:
                in_buffer = None

            if not test_mode:
                predictions, loss = model.observe(images, labels, in_buffer, train=False)
                valid_loss += loss * images.shape[0]
            else:
                predictions, output = model(images, return_output=True)

            outputs = output[:, :(num_seen_classes-num_class_per_task)]
            super_predictions = outputs > 0.0
            # super_labels=labels[:, :10].bool()

            for index in range(len(labels)):
                complete_label=labels[index]
                complete_label=complete_label.data.cpu().numpy()
                # value=bool2int(complete_label.astype(np.int32))
                value=np.array2string(complete_label.astype(np.int32))
                # value=hot_to_value(complete_label)
                if not value in super_pred_dict.keys():
                    super_pred_dict[value]=[]

                super_pred_dict[value].append(super_predictions[index].data.cpu().detach().numpy())

            # np.vstack(super_pred_dict[16386]).astype(np.int64).sum(axis=0)  16386, 4098, 1024, 8192, 2048

        verified_super_key=[]
        all_pos_super_key=[]
        all_keys=[]
        all_probs=[]
        for key in super_pred_dict.keys():
            probs=np.vstack(super_pred_dict[key]).astype(np.int64).sum(axis=0)
            prob= float(probs.max()) / float(len(super_pred_dict[key]))

            max_label=np.zeros(len(probs),dtype=np.int64)
            max_label[probs.argmax()]=1
            max_string=np.fromstring(key[1:-1], dtype=np.int, sep=' ')
            max_string=np.hstack((max_label,max_string[-num_class_per_task:]))
            all_pos_super_key.append(max_string)

            if prob > Threshold:
                preset_label=np.zeros(len(probs),dtype=np.int64)
                preset_label[probs.argmax()]=1
                ########### ::-1 reverse
                label_string=np.fromstring(key[1:-1], dtype=np.int, sep=' ')
                label_string=np.hstack((preset_label,label_string[-num_class_per_task:]))
                verified_super_key.append(label_string)

            all_keys.append(key)
            all_probs.append(prob)
            # print(prob)
            #### later print all of them for comparison
        return verified_super_key,all_keys,all_probs, all_pos_super_key



def hot_to_value(hot_vector):
    value=0
    for j in range(len(hot_vector)):
        value+=hot_vector[j]*pow(2,j)
    return int(value)


def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y


def modified_tasks_eval(model, dataset, cur_task_id, config, metadata, logbook, dataset_type="valid", dist_args=None,
               look_table=None):
    """log the accuracies of the new model on all observed tasks
    :param metadata:
    """
    assert dataset.complete_information_mode is True

    distributed = dist_args is not None
    if distributed:
        gpu = dist_args["gpu"]
        rank = dist_args["rank"]
    else:
        gpu = None
        rank = 0

    metrics_dict = {}
    # for task_id in range(cur_task_id + 1):
    #     # for task_id in range(1,cur_task_id + 1):
    #     dataset.choose_task(task_id)
    #     dataloader = data.DataLoader(
    #         dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
    #     )
    #     _, metrics = evaluate(model, dataloader, config, metadata, test_mode=True, gpu=gpu)
    #     for metric in metrics.keys():
    #         metrics_dict[f"task_{task_id}_{dataset_type}_{metric}"] = metrics[metric]

    dataset.load_tasks_up_to(cur_task_id)
    dataloader = data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )
    _, metrics,  original_pred_list, modified_pred_list, GT = modified_evaluate(model, dataloader, config, metadata, test_mode=True, gpu=gpu,
                          look_table=look_table)

    for metric in metrics.keys():
        metrics_dict[f"average_{dataset_type}_{metric}"] = metrics[metric]

    if rank == 0:
        utils.log_task(cur_task_id, metrics_dict, logbook)
        if distributed:
            dist.barrier()
            metrics_dict["rank"] = rank
            print_msg(metrics_dict)
    else:
        dist.barrier()
        metrics_dict["rank"] = rank
        print_msg(metrics_dict)

    return metrics_dict, original_pred_list, modified_pred_list, GT



def modified_evaluate(model, dataloader, config, metadata, test_mode=False, \
                gpu=None,look_table=None,super_class_index=None,sub_class_index=None):
    valid_loss = 0
    valid_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0.}
    data_len = 0
    class_names_to_idx = metadata["class_names_to_idx"]
    num_seen_classes = len(model.seen_classes)
    model.net.eval()
    #########################################
    # valid_metrics['super_modified_jaccard']=0
    #########################################
    with torch.no_grad():
        original_pred_list=[]
        modified_pred_list=[]
        GT=[]
        for minibatch in dataloader:
            labels_names = list(zip(minibatch[1], minibatch[2]))
            labels = transform_labels_names_to_vector(
                labels_names, num_seen_classes, class_names_to_idx
            )
            #####################
            # if verified_super_key is not None:
            #     for lbl_ind in range(len(labels)):
            #         lbl = labels[lbl_ind]
            #         for ind in range(len(verified_super_key)):
            #             logic_add = lbl.numpy().astype(np.int32) & verified_super_key[ind]
            #             if np.delete(logic_add, super_class_index).sum() > 0:
            #             # if (lbl.numpy().astype(np.int32) & np.delete(verified_super_key[ind],super_class_index) ).sum() > 0:
            #                 labels[lbl_ind] = torch.from_numpy(verified_super_key[0]).float()

            if gpu is None:
                images = minibatch[0].to(config["device"], non_blocking=True)
                labels = labels.to(config["device"], non_blocking=True)
            else:
                images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
                labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

            if len(minibatch) > 3:
                if gpu is None:
                    in_buffer = minibatch[3].to(config["device"], non_blocking=True)
                else:
                    in_buffer = minibatch[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            else:
                in_buffer = None

            if not test_mode:
                predictions, loss = model.observe(images, labels, in_buffer, train=False)
                valid_loss += loss * images.shape[0]
            else:
                predictions = model(images)
            
            original_pred = deepcopy(predictions)
            original_pred_list.append(original_pred)
            GT.append(labels)

            if look_table is not None:
                for item_id in range(len(predictions)):
                    # for look_id in range(len(look_table)):
                    pred_dup=predictions[item_id].view(1, -1).repeat(len(look_table), 1)

                    xor_matrix = pred_dup ^ look_table

                    if (xor_matrix.sum(1) == 0).sum() == 0: # no match
                        # most_match_table_item=xor_matrix.sum(1).argmin()
                        # #############################################
                        min_value=xor_matrix.sum(1).min()
                        min_indexes=torch.where(xor_matrix.sum(1)==min_value)[0]
                        ##########################
                        most_match_table_item=min_indexes[torch.randperm(len(min_indexes))[0]]
                        ###############################################
                        # most_match_table_item=min_indexes[0]
                        ################################################
                        
                        predictions[item_id]=look_table[most_match_table_item]
                    # xor_matrix.sum(1).cpu().detach().numpy().argmin()
                    # if (predictions[item_id] ^ look_table[look_id]).sum() > 0:
                    # predictions[item_id] & look_table[look_id]
            labels = labels.bool()

            modified_pred_list.append(predictions)
            ###################
            # mask=(labels[:,:10].sum(dim=1)==1)
            # if torch.sum(mask)>0:
            #     valid_metrics['super_modified_jaccard']+= \
            #         metrics.super_modified_jaccard_sim\
            #             (predictions[mask,:10], labels[mask,:10]) \
            #             * images.shape[0]
            ##################
            valid_metrics['jaccard_sim'] += metrics.modified_jaccard_sim(original_pred, labels) * images.shape[0]
            valid_metrics['modified_jaccard'] += metrics.modified_jaccard_sim(predictions, labels) * images.shape[0]

            valid_metrics['strict_acc'] += metrics.strict_accuracy(predictions, labels) * images.shape[0]
            valid_metrics['recall'] += metrics.recall(predictions, labels) * images.shape[0]
            data_len += images.shape[0]
        valid_loss /= data_len
        valid_metrics['jaccard_sim'] /= data_len
        valid_metrics['modified_jaccard'] /= data_len
        valid_metrics['strict_acc'] /= data_len
        valid_metrics['recall'] /= data_len
        # valid_metrics['super_modified_jaccard'] /=data_len
    return valid_loss, valid_metrics, original_pred_list, modified_pred_list, GT

