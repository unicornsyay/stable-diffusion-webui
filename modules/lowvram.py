import torch
from modules import devices
## if you want to reduce memory requirements, use model simplification, quantization, gradient checkpointing, mixed precision training, or working with smaller batch sizes

module_in_gpu = None
cpu = torch.device("cpu")
# Set the GPU memory threshold in bytes
threshold = 0.80 * torch.cuda.get_device_properties(0).total_memory  # 80% of total GPU memory, provides a 20% buffer
# Perform GPU computations
tensor_gpu = torch.randn(10, 3, 1600, 1600, device='cuda')
#10 = batch_size = number of images being processed // may need to refine this. 
#3 = RGB colors used by MOST models minus scientific ones which may have more data - but this should be kept as '3'
#width - the width of the  images
#height = the height of the image.
## This is a basic memory check and won't handle spikes during training

def check_memory_and_allocate(shape):
    # Get current memory usage
    current_memory = torch.cuda.memory_allocated()
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # Calculate the size of the new tensor
    new_tensor_memory = torch.Tensor(*shape).numel() * 4  # 4 bytes per float32
    # Check if there is enough memory left for the new tensor
    if current_memory + new_tensor_memory > total_memory * 0.9:  # leave 10% buffer
        print("Insufficient memory to create new tensor. Reducing batch size or model size might be necessary.")
        #return torch.randn(*shape, device='cuda') #can uncomment if you want to use CPU when getting warning
    else:
        return torch.randn(*shape, device='cuda')

if torch.cuda.memory_allocated() > threshold:
    print("Switching to CPU...")
    tensor_cpu = tensor_gpu.to('cpu')
else:
    print("Continuing with GPU computations...")
    result_gpu = tensor_gpu * 2


def send_everything_to_cpu():
    global module_in_gpu

    if module_in_gpu is not None:
        module_in_gpu.to(cpu)

    module_in_gpu = None


def setup_for_low_vram(sd_model, use_medvram):
    parents = {}

    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu  
        # Monitor GPU memory usage and switch to CPU if threshold is exceeded
        threshold = torch.cuda.max_memory_allocated() *0.85 #15% buffer - not to be used when training models
        if threshold > 0:
            if torch.cuda.memory_allocated() > threshold:
                print("Switching to CPU...")
                send_everything_to_cpu()
                module = parents.get(module, module)

        if module_in_gpu == module:
            return

        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        module.to(devices.device)
        module_in_gpu = module

    # see below for register_forward_pre_hook;
    # first_stage_model does not use forward(), it uses encode/decode, so register_forward_pre_hook is
    # useless here, and we just replace those methods

    first_stage_model = sd_model.first_stage_model
    first_stage_model_encode = sd_model.first_stage_model.encode
    first_stage_model_decode = sd_model.first_stage_model.decode

    def first_stage_model_encode_wrap(x):
        send_me_to_gpu(first_stage_model, None)       
        return first_stage_model_encode(x)

    def first_stage_model_decode_wrap(z):
        send_me_to_gpu(first_stage_model, None)          
        return first_stage_model_decode(z)

    # for SD1, cond_stage_model is CLIP and its NN is in the tranformer frield, but for SD2, it's open clip, and it's in model field
    if hasattr(sd_model.cond_stage_model, 'model'):
        sd_model.cond_stage_model.transformer = sd_model.cond_stage_model.model

    # remove several big modules: cond, first_stage, depth/embedder (if applicable), and unet from the model and then
    # send the model to GPU. Then put modules back. the modules will be in CPU.
    stored = sd_model.cond_stage_model.transformer, sd_model.first_stage_model, getattr(sd_model, 'depth_model', None), getattr(sd_model, 'embedder', None), sd_model.model
    sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.depth_model, sd_model.embedder, sd_model.model = None, None, None, None, None
    sd_model.to(devices.device)
    sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.depth_model, sd_model.embedder, sd_model.model = stored

    # register hooks for those the first three models
    sd_model.cond_stage_model.transformer.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.encode = first_stage_model_encode_wrap
    sd_model.first_stage_model.decode = first_stage_model_decode_wrap
    if sd_model.depth_model:
        sd_model.depth_model.register_forward_pre_hook(send_me_to_gpu)
    if sd_model.embedder:
        sd_model.embedder.register_forward_pre_hook(send_me_to_gpu)
    parents[sd_model.cond_stage_model.transformer] = sd_model.cond_stage_model

    if hasattr(sd_model.cond_stage_model, 'model'):
        sd_model.cond_stage_model.model = sd_model.cond_stage_model.transformer
        del sd_model.cond_stage_model.transformer

    if use_medvram:
        sd_model.model.register_forward_pre_hook(send_me_to_gpu)
        if torch.cuda.memory_allocated() > threshold:
            print("Switching to CPU...")
            send_everything_to_cpu()
    else:
        diff_model = sd_model.model.diffusion_model

        # the third remaining model is still too big for 4 GB, so we also do the same for its submodules
        # so that only one of them is in GPU at a time
        stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = None, None, None, None
        sd_model.model.to(devices.device)
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = stored

        # install hooks for bits of third model
        diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.input_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
        diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.output_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
        # Monitor GPU memory usage and switch to CPU if threshold is exceeded
            if torch.cuda.memory_allocated() > threshold:
                print("Switching to CPU...")
                send_everything_to_cpu()

