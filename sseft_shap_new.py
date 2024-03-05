

import shap
import torch
import numpy as np
import math
import random
from tqdm import tqdm
from fairseq import checkpoint_utils, options, progress_bar, utils

## This file contains the explainability framework merged with the model 'sse-ft', it is a work in progress, however the masking logic is tested and is correct

## Main for now parses the args from the command prompt and passes them to load_models()




# Load_models() gets the model from the fairseq library and sets the checkpoint, it initialises a data iterator and loops over the samples in the dataset. 
# For now I need to test the masking functions together with the model, later i will adjust the code so that it can handle data from Sound and Vision
global text_tensor
global audio_tensor
global video_tensor
global text_length
global audio_length
global padded_amount_value
global samples_per_token
global task
global model
global criterion
global use_cuda
def load_models(args, override_args=None):
    global task
    global model
    global criterion
    global use_cuda
    utils.import_user_module(args)
    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if override_args is not None:
            overrides = vars(override_args)
            overrides.update(eval(getattr(override_args, 'model_overrides', '{}')))
    else:
        overrides = None
    ## Load ensemble and extract model
    print('| loading model(s) from {}'.format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        )
    model = models[0]

        ## Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()
    print(model_args)

    ## Build criterion
    criterion = task.build_criterion(model_args)
    criterion.eval()

    ## Load valid dataset 
    for subset in args.valid_subset.split(','):
        try:
            task.load_dataset(subset, combine=False, epoch=0)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception('Cannot find dataset: ' + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )


    results_dict = {} ## This dict will gatter the shap values for each data sample key: sample ID value: shap-values for each combination of tokens

    ## This loop iterates over every sample and creates the tokens representing the data input for each modality (shap needs tokens instead if the real input data)
    for index, sample in enumerate(progress):
        print(sample)
        #DATA TEST, the code below is made for testing
        #x = {'net_input': {'audio': audio_tensor, 'text': text_tensor, 'video': video_tensor, 'padded_amount':padded_amount_value }}
        #audio_tensor = torch.randn(1, 150000)  # Example dimensions (batch_size, channels, audio_length)
        #text_tensor = torch.randn(1, 30)  # Example dimensions (batch_size, seq_length, embedding_size)
        #video_tensor = torch.randn(1, 3, 300, 256, 256)  # Example dimensions (batch_size, channels, frames, height, width)
        #padded_amount_value = 100
        global audio_tensor
        global text_tensor
        global video_tensor
        global padded_amount_value
        global text_length
        global audio_length
        global samples_per_token
        audio_tensor = sample['net_input']['audio']
        text_tensor = sample['net_input']['text']
        video_tensor = sample['net_input']['video']
        padded_amount_value = sample['net_input']['padded_amount']

        ## Text tokens
        text = text_tensor.squeeze(0).cpu().numpy()
        last_non_zero_index = np.max(np.nonzero(text))
        text_length  = len(text[:last_non_zero_index + 1])
        text_tokens_ids = torch.tensor(range(1, text_length+1)).unsqueeze(0)
        #text_input = TextInputData(input_ids=text_tokens_ids, text_length=text_length, text_tensor = text_tensor)
        ## Audio tokens, each token represents a segment of a second in spectogram values (the sanmple rate is used to calculate the amount of values per token)
        audio = audio_tensor.squeeze(0).cpu().numpy()
        last_index = len(audio) - np.argmax(np.flip(audio) != 1) - 1
        print(audio.shape)
        audio = audio[:last_index + 1]
        print(audio.shape)
        sample_rate = 16000
        sample_duration = 1.0 / sample_rate
        token_duration = 1.0
        samples_per_token = int(token_duration / sample_duration)
        audio_length = math.ceil((len(audio)/samples_per_token))
        audio_tokens_ids = torch.tensor(range(1, audio_length+1)).unsqueeze(0) 
        #audio_input = AudioInputData(input_ids=audio_tokens_ids, audio_length=audio_length, audio_tensor = audio_tensor)
        print(audio_tokens_ids.shape)
        ## Video tokens, each pixel is represented as a token (256 x 256 is an image) 
        video_token_ids = torch.tensor(range(1, 257)).unsqueeze(0)
        #video_input = VideoInputData(input_ids=video_token_ids, video_tensor = video_tensor)
        ## All tokens are concatenated together
        All_token_ids = torch.cat((text_tokens_ids, audio_tokens_ids, video_token_ids), 1).unsqueeze(1)
        print(All_token_ids)

        ## A SHAP explainer object is made with the custom masker and the model prediction function
        explainer = shap.Explainer(
        get_model_prediction, custom_masker, silent=True)
        shap_values = explainer(All_token_ids)

        ## Shap values are stored
        results_dict[index] = shap_values


## The custom masker recieves the input tokens and a boolean tensor of the same shape (what need to be masked)
def custom_masker(mask, x):
    masked_X = x.clone()
    mask = torch.tensor(mask).unsqueeze(0)
    masked_X[~mask] = 0  # ~mask !!! to zero
    # never mask out CLS and SEP tokens (makes no sense for the model to work without them) -> this is handled in the load_models range(1,...) instead of range*(0,....)
    return masked_X

## This function recieves an array with rows that represent combinations of masked tokens for one sample
## From the masked tokens, based on the token generaton logic, the input is reconstructed to the form needed by the model
def get_model_prediction(x):
    with torch.no_grad():
        print(x)
        print(x.shape)
        masked_text_tokens_ids = torch.tensor(x[:, :text_length]) 
        print(masked_text_tokens_ids)
        masked_audio_tokens_ids = torch.tensor(x[:, text_length: text_length + audio_length])
        masked_video_tokens_ids = torch.tensor(x[:, text_length + audio_length:])
        result = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            masked_text_tensor = text_tensor.clone()
            for k in range(1, len(masked_text_tokens_ids)):
                if masked_text_tokens_ids[0,k].item() == 0:
                    masked_text_tensor[0,k] = 0 ## Here the real text masking happens
            masked_audio_tensor = audio_tensor.clone()
            print(len(masked_audio_tokens_ids))
            for k in range(masked_audio_tokens_ids.size(1)-1):
                print(k)
                if masked_audio_tokens_ids[0,k].item() == 0:
                    masked_audio_tensor[k:k+samples_per_token] = 0 ## Here the real audio masking happens
            masked_video_tensor = video_tensor.clone()
            num_patches = 256
            patch_size = 1
            num_frames = 300
            print(masked_video_tensor)
            print(masked_video_tensor.shape)
            for k in range(masked_video_tokens_ids.size(1)-1):
                print(masked_video_tokens_ids)
                if masked_video_tokens_ids[0,k].item() == 0 :
                    start_row = (k // (256 // patch_size)) * patch_size
                    start_col = (k % (256 // patch_size)) * patch_size
                    for frame_idx in range(masked_video_tensor.shape[2]):
                      if torch.any(masked_video_tensor[0, :, frame_idx] == -1):
                        continue  # Skip padded frames
                      masked_video_tensor[:,:, frame_idx, start_row:start_row+patch_size, start_col:start_col+patch_size] = 0 ## Here the real video masking happens
            ## Input in the right shape fir model
            #masked_input = {'net_input': {'audio': masked_audio_tensor, 'text': masked_text_tensor, 'video': masked_video_tensor, 'padded_amount':padded_amount_value }}
            print(masked_input)
            masked_model_input = {'id': tensor([0]), 'net_input': {'audio': masked_audio_tensor, 'text': masked_text_tensor, 'video': masked_video_tensor, 'padded_amount':padded_amount_value}, 'target': tensor([0.])} #TODO change id
            masked_model_input = utils.move_to_cuda(masked_model_input) if use_cuda else masked_model_input
            ## Collect the prediction andthe probabilities, let the model infer
            prediction, probabilities = task.valid_step(masked_input, model, criterion)
            ## Store the outcome of the model
            result[i] = (prediction, probabilities)
    return result



def cli_main():

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    load_models(args, override_args)
if __name__ == '__main__':
    cli_main()
