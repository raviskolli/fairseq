import os
import onnx
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription
from onnxruntime.capi.ort_trainer import LossScaler
import torch
from fairseq.ort_supplement.azureml_adapter import set_environment_variables_for_nccl_backend, get_local_rank, get_local_size, get_global_size, get_world_size, get_world_rank 

def setup_onnxruntime_with_mpi(args):
    '''
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    has_aml = 'AZ_BATCH_MASTER_NODE' in os.environ.keys() or 'AZ_BATCHAI_MPI_MASTER_NODE' in os.environ.keys()
    if not has_aml:
        print('Detected local run')
        args.local_rank = comm.Get_rank() % torch.cuda.device_count()
        args.world_rank = comm.Get_rank()
        args.world_size = comm.Get_size()

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1

    else:
        print('Detected Azure batch run')
        set_environment_variables_for_nccl_backend(get_local_size() == get_global_size(), IB = args.use_ib)
        args.local_rank = get_local_rank()
        args.local_size = get_local_size()
        args.world_rank = get_world_rank()
        args.world_size = get_global_size()

        print('Local rank: {}'.format(args.local_rank))
        print('Local size: {}'.format(args.local_size))
        print('World rank: {}'.format(args.world_rank))
        print('World size: {}'.format(args.world_size))
        print('CUDA device: {}'.format(args.local_rank))

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1

        torch.distributed.init_process_group(backend='nccl')
    '''

    torch.cuda.set_device(args.distributed_rank)
    device = torch.device("cuda", args.distributed_rank)

    from onnxruntime.capi._pybind_state import set_cuda_device_id 
    set_cuda_device_id(args.distributed_rank)

    from onnxruntime.capi._pybind_state import set_arena_extend_strategy, ArenaExtendStrategy
    set_arena_extend_strategy(ArenaExtendStrategy.kSameAsRequested)

    return device

def bert_model_description(args):
    vocab_size = 50349

    # allow variable input sizes:
    # input_ids_desc = IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = vocab_size)
    # segment_ids_desc = IODescription('segment_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = 2)
    # input_mask_desc = IODescription('input_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = 2)
    # masked_lm_labels_desc = IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = vocab_size)
    # next_sentence_labels_desc = IODescription('next_sentence_labels', ['batch',], torch.int64, num_classes = 2)

    # set concrete input sizes to permit optimization
    src_tokens_desc = IODescription('src_tokens', ['batch', 'max_tokens_valid'], torch.int64, num_classes = vocab_size)
    src_lengths_desc = IODescription('src_lengths', ['batch'], torch.int64, num_classes = args.max_tokens_valid)
    prev_output_tokens_desc = IODescription('prev_output_tokens', ['batch', 'max_tokens_valid'], torch.int64, num_classes = vocab_size)
    target_desc = IODescription('target', ['batch', 'max_tokens_valid'], torch.int64, num_classes = vocab_size)

    loss_desc = IODescription('loss', [], torch.float32)
    return ModelDescription([src_tokens_desc, src_lengths_desc, prev_output_tokens_desc, target_desc], [loss_desc])

# for opset 10
from fairseq.ort_supplement.onnx_transforms.model_transform import add_name, fix_transpose, add_expand_shape, process_concat, process_dropout, handle_expand_input_is_not_constant_case, fix_dim, fix_expand

from fairseq.ort_supplement.onnx_transforms.layer_norm_transform import layer_norm_transform

def postprocess_model(model):
    add_name(model)
    # for opset 10 ..
    handle_expand_input_is_not_constant_case(model)
    fix_expand(model)
    fix_dim(model)
    process_dropout(model)
    # --- 
    add_expand_shape(model)
    layer_norm_transform(model)

def create_ort_trainer(args, device, model):
    # set GPU memory limitation
    from onnxruntime.capi._pybind_state import set_cuda_mem_limit
    ort_cuda_mem_limit_in_gbs = 16
    set_cuda_mem_limit(int(ort_cuda_mem_limit_in_gbs * 1024 * 1024 *1024))

    def map_optimizer_attributes(name):
        no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
        no_decay = False
        for no_decay_key in no_decay_keys:
            if no_decay_key in name:
                no_decay = True
                break
        if no_decay:
            return {"alpha": 0.9, "beta": 0.999,
                    "lambda": 0.0,
                    "epsilon": 1e-8, #if self.optimizer == 'LambOptimizer' else 1e-8,
                    # Adam optimizer mode
                    # 0: pytorch's Adamw
                    # 1: Huggface's Adamw
                    "weight_decay_mode" : 0,
                    "do_bias_correction" : 1}
        else:
            return {"alpha": 0.9, "beta": 0.999,
                    "lambda": 0.01,
                    "epsilon": 1e-8, #if self.optimizer == 'LambOptimizer' else 1e-8,
                    # Adam optimizer mode
                    # 0: pytorch's Adamw
                    # 1: Huggface's Adamw
                    "weight_decay_mode" : 0,
                    "do_bias_correction" : 1}

    # we request ORTTrainer to create a LambOptimizer with given optimizer_attributes. 
    # train_step does forward, backward, and optimize step.
    model = ORTTrainer(model, None, bert_model_description(args), "AdamOptimizer", 
        map_optimizer_attributes,
        IODescription('Learning_Rate', [1,], torch.float32),
        device, postprocess_model=postprocess_model, 
        gradient_accumulation_steps=args.update_freq[0],
        world_rank=args.distributed_rank, world_size=args.distributed_world_size,
        use_mixed_precision = True if args.fp16 else False,
        allreduce_post_accumulation = False, #if args.allreduce_post_accumulation else False,
        partition_optimizer = False, #if args.partition_optimizer else False,
        _opset_version = 12)

    if args.fp16:
        setattr(args, 'ort_loss_scale', LossScaler(model.loss_scale_input_name, True, up_scale_window=2000))

    return model

def get_lr(args, update_num):
    warmup_end_lr = args.lr[0]
    if args.warmup_init_lr < 0:
        args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

    # linearly warmup for the first args.warmup_updates
    self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

    # then, decay prop. to the inverse square root of the update number
    self.decay_factor = warmup_end_lr * args.warmup_updates**0.5

    # initial learning rate
    self.lr = args.warmup_init_lr
    
    if num_updates < self.args.warmup_updates:
        self.lr = self.args.warmup_init_lr + num_updates*self.lr_step
    else:
        self.lr = self.decay_factor * num_updates**-0.5
    return self.lr

def ort_train_step(args, update_num, model, sample):
    src_tokens = sample['src_tokens']
    src_lengths = sample['src_lengths']
    prev_output_tokens = sample['prev_output_tokens']
    target = sample['target']
    target = target.view(-1)

    lr = get_lr(args, update_num)
    learning_rate = torch.tensor([lr])
    if args.fp16:
        loss_scale = torch.tensor([args.ort_loss_scale.loss_scale_])
        loss = model.train_step(src_tokens, src_lengths, prev_output_tokens, target, learning_rate, loss_scale)
        all_finite = 1
        if isinstance(loss, (list, tuple)):
            assert len(loss) == 2
            loss, all_finite = loss
    else:
        loss = model(src_tokens, src_lengths, prev_output_tokens, target, learning_rate, learning_rate)

    if training_steps % args.gradient_accumulation_steps == 0:
        if args.fp16:
            args.ort_loss_scale.update_loss_scale(all_finite.item())
        global_step += 1

    sample_size = sample['ntokens']
    logging_output = {
        'loss': loss.data,
        'ntokens': sample['ntokens'],
        'nsentences': sample['target'].size(0),
        'sample_size': sample_size,
    }

    return loss, sample_size, logging_output
 