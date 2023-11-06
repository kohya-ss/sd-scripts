__There may be errors in the description as the document is being updated.__

# Learning, Common Edition

In this repository, we support the learning of model fine-tuning, DreamBooth, and LoRA and Textual Inversion (including [XTI:P+](https://github.com/kohya-ss/sd-scripts/pull/327)). This document explains the common methods for preparing training data and options.

# Overview

Please refer to the README of this repository in advance and set up the environment.

We will explain the following:

1. About preparing training data (new format using a configuration file)
2. A brief explanation of terms used in learning
3. Previous specification format (specified from the command line without using a configuration file)
4. Generation of sample images during learning
5. Commonly used options in each script
6. Preparation of metadata for fine-tuning: captioning, etc.

You can start learning by just doing 1 (please refer to the documentation of each script for learning). Please refer to 2 and onwards as needed.

# About Preparing Training Data

Prepare image files for training data in any folder (multiple folders are also acceptable). We support `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`. Basic preprocessing such as resizing is not necessary.

However, we recommend that you do not use images that are extremely smaller than the learning resolution (described later), or that you enlarge them in advance with super-resolution AI. Also, images that are extremely large (about 3000x3000 pixels?) may cause errors, so please reduce them in advance.

During training, you need to organize the image data to be learned by the model and specify it to the script. Depending on the number of training data, learning targets, whether or not captions (description of the image) can be prepared, etc., you can specify the training data in several ways. Here are the ways (the names are not general ones, but are defined uniquely for this repository). Regularized images will be described later.

1. DreamBooth, class+identifier method (regularization image can be used)

    Learn to associate the learning target with a specific word (identifier). There is no need to prepare a caption. For example, when you teach a specific character, you don't need to prepare a caption, which is handy, but all elements of the learning data such as hairstyle, clothing, and background are associated with the identifier and learned, so you may not be able to change clothes in the prompt when generating.

2. DreamBooth, caption method (regularization image can be used)

    Prepare a text file recorded with captions for each image and learn. For example, when you teach a specific character, by describing the details of the image in the caption (character A in white clothes, character A in red clothes, etc.), the character and other elements are separated, and the model can be expected to learn only the character more strictly.

3. Fine-tuning method (regularization image cannot be used)

    Collect the captions in advance in a metadata file. Supports features such as managing tags and captions separately and pre-caching latents to speed up learning (both are explained in other documents). (Although it is called the fine-tuning method, it can be used for other than fine-tuning.)

The combination of what you want to learn and the available specification methods are as follows:

| Learning target or method | Script | DB / class+identifier | DB / caption | Fine tuning |
| ----- | ----- | ----- | ----- | ----- |
| Fine-tuning the model | `fine_tune.py`| x | x | o |
| DreamBooth the model | `train_db.py`| o | o | x |
| LoRA | `train_network.py`| o | o | o |
| Textual Inversion | `train_textual_inversion.py`| o | o | o |

## Which to Choose

When it comes to LoRA and Textual Inversion, if you want to train without preparing caption files easily, DreamBooth class+identifier might be a good choice, if you can prepare them, DreamBooth caption method may be preferred. If you have a large amount of training data and do not use regularized images, you should also consider the fine-tuning method.

The same applies to DreamBooth, but the fine-tuning method cannot be used. For fine tuning, only the fine tuning method is used.

# How to Specify Each Method

Here, I will only explain typical patterns for each specification method. For more detailed specification methods, please see [Dataset Setup](./config_README-en.md).

# DreamBooth, class+identifier method (Regularized images can be used)

In this method, each image becomes the same as it was trained with a caption called `class identifier` (like `shs dog`).

## step 1. Decide identifier and class

Determine the word identifier that connects the object you want to learn and the class to which the object belongs.

(There are various terms such as instance, but for now, I will stick to the original paper.)

I will explain very briefly (please research for more details).

Class is a general type of learning target. For example, if you want to teach a specific dog breed, the class will be dog. For anime characters, it would probably be boy or girl, 1boy or 1girl, depending on the model.

The identifier is for identifying and learning the learning target. Any word is fine, but according to the original paper, it is good to have "a rare word of 3 characters or less that becomes one token in tokenizer".

By using the identifier and class, you can identify the object you want to teach from the class and learn it, for example, "shs dog".

When generating images, if you say "shs dog", an image of the dog breed you taught will be generated.

(For reference, here are some of the identifiers that I have been using recently, such as ``shs sts scs cpc coc cic msm usu ici lvl cic dii muk ori hru rik koo yos wny``. Ideally, it would be better not to be included in the Danbooru Tag.)

## step 2. Decide whether to use regularized images, and if so, generate regularized images

Regularized images are images used to prevent the entire class mentioned above from being pulled to the learning target (language drift). If you do not use regularized images, if you teach a specific character with `shs 1girl`, even if you generate with a simple `1girl` prompt, it will resemble that character. This is because `1girl` is included in the caption during learning.

By teaching the target image and regularized image at the same time, the class remains as a class, and only when you add an identifier to the prompt, the learning target will be generated.

If you only need a specific character to come out in LoRA or DreamBooth, you may not need to use regularized images.

Textual Inversion does not need to be used (nothing will be learned if the token string to be learned is not included in the caption).

As a regularized image, it is common to use an image generated with just the class name of the learning target model (for example, `1girl`). However, if the quality of the generated image is poor, you can devise a prompt or use an image downloaded separately from the internet.

(Since the regularized image is also learned, its quality affects the model.)

Generally, it seems desirable to prepare about several hundred

 (if the number is small, the class image is not generalized and it will learn their characteristics).

If you use a generated image, normally, please adjust the size of the generated image to the learning resolution (more precisely, the resolution of the bucket, mentioned later).

## step 2. Writing the Configuration File

Create a text file and change the extension to `.toml`. For example, write as follows.

(The parts starting with `#` are comments, so you can copy and paste them as they are, or delete them without any problem.)

```toml
[general]
enable_bucket = true                        # Whether to use Aspect Ratio Bucketing or not

[[datasets]]
resolution = 512                            # Learning resolution
batch_size = 4                              # Batch size

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # Specify the folder where the training images are placed
  class_tokens = 'hoge girl'                # Specify the identifier class
  num_repeats = 10                          # Number of repetitions for training images

  # Describe below only when using regularized images. If not used, delete it
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # Specify the folder where the regularized images are placed
  class_tokens = 'girl'                     # Specify the class
  num_repeats = 1                           # Number of repetitions for regularized images, basically 1 is fine
```

Basically, you can learn by rewriting only the following places.

1. Learning Resolution

    If you specify one number, it will be a square (if `512` then 512x512), if you specify two numbers separated by a comma in brackets, it will be width x height (if `[512,768]` then 512x768). The original learning resolution in SD1.x series is 512. If you specify a larger resolution such as `[512,768]`, you might be able to reduce the distortion when generating long vertical or horizontal images. In SD2.x 768 series, it is `768`.

1. Batch Size

    Specify how many data are learned at the same time. It varies depending on the GPU VRAM size and learning resolution. We will explain in detail later. Also, it varies depending on fine-tuning/DreamBooth/LoRA etc., so please also refer to the explanation of each script.

1. Folder Specification

    Specify the folder for training images and regularized images (only when used). Specify the folder itself that contains image data.

1. Identifier and Class Specification

    It's just like the sample mentioned earlier.

1. Number of Repetitions

    We will explain later.

### About the Number of Repetitions

The number of repetitions is used to adjust the number of regularized images and training images. Since there are more regularized images than training images, the training images are repeated to match the number, so that they can be learned at a 1:1 ratio.

Please specify the number of repetitions so that " __the number of repetitions of training images × the number of training images ≥ the number of repetitions of regularized images × the number of regularized images__ ".

(The number of data for 1 epoch (1 epoch when the data goes around once) is "the number of repetitions of training images × the number of training images". If the number of regularized images is greater than that, the extra regularized images will not be used.)

## step 3. Training

Please conduct training by referring to each document.

# DreamBooth, Caption Method (Regularization Images Can Be Used)

In this method, each image is trained with a caption.

## step 1. Prepare the Caption File

In the folder with training images, place a file with the same name as the image and with the extension `.caption` (which can be changed in the settings). Each file should contain only one line. The encoding should be `UTF-8`.

## step 2. Decide Whether to Use Regularization Images and Generate Them if Needed

This is the same as the class+identifier method. Note that although it's possible to add captions to regularization images, it's usually not necessary.

## step 2. Write the Configuration File

Create a text file and change the extension to `.toml`. For example, write it as follows:

```toml
[general]
enable_bucket = true                        # Whether to use Aspect Ratio Bucketing or not

[[datasets]]
resolution = 512                            # Training resolution
batch_size = 4                              # Batch size

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # Specify the folder where the training images are stored
  caption_extension = '.caption'            # Extension for the caption file; change this if you're using .txt
  num_repeats = 10                          # Number of repetitions for the training images

  # The following should be written only if you're using regularization images. If not, remove it.
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # Specify the folder where the regularization images are stored
  class_tokens = 'girl'                     # Specify the class
  num_repeats = 1                           # Number of repetitions for the regularization images, typically 1
```

You can start training by simply modifying the following areas. Areas not specifically mentioned are the same as the class+identifier method.

1. Training resolution
1. Batch size
1. Folder specification
1. Caption file extension

    You can specify any extension.
1. Number of repetitions

## step 3. Training

Please conduct training by referring to each document.

# Fine Tuning Method

## step 1. Prepare Metadata

We refer to the management file that consolidates captions and tags as metadata. It is in json format with the extension `.json`. The method to create it is written at the end of this document as it is lengthy.

## step 2. Write the Configuration File

Create a text file and change the extension to `.toml`. For example, write it as follows:

```toml
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = 512                                    # Training resolution
batch_size = 4                                      # Batch size

  [[datasets.subsets]]
  image_dir = 'C:\piyo'                             # Specify the folder where the training images are stored
  metadata_file = 'C:\piyo\piyo_md.json'            # Metadata file name
```

You can start training by simply modifying the following areas. Areas not specifically mentioned are the same as the DreamBooth, class+identifier method.

1. Training resolution
1. Batch size
1. Folder specification
1. Metadata file name

    Specify the metadata file created with the method described later.

## step 3. Training

Please conduct training by referring to each document.

# A Simple Explanation of Terms Used in Training

I've omitted the finer details, and I don't fully understand all of these terms myself, so please look up any specifics on your own.

## Fine Tuning

Refers to training and refining the model. Depending on how it is used, the meaning can differ. In a narrow sense, fine tuning in the case of Stable Diffusion means training the model with images and captions. DreamBooth can be seen as a special method of this narrow sense of fine tuning. In a broader sense, fine tuning includes LoRA, Textual Inversion, Hypernetworks, etc., and encompasses all ways of training a model.

## Step

Roughly speaking, one calculation with the training data is one step. "Flow the training data's caption into the current model, compare the resulting image with the image from the training data, and slightly modify the model to bring it closer to the training data." That's one step.

## Batch Size

Batch size specifies how many pieces of data are computed together in one step. Because calculations are performed in batches, speed is relatively improved. Also, it is generally said that accuracy also increases.

The number of data used for learning is `batch size x number of steps`. Therefore, it would be better to decrease the number of steps by the amount you increase the batch size.

(However, for example, "batch size 1 with 1600 steps" and "batch size 4 with 400 steps" do not produce the same results. In general, the latter tends to be under-learned with the same learning rate. Try to devise a strategy, such as slightly increasing the learning rate (e.g., `2e-6`), or setting the number of steps to 500, for example.)

Increasing the batch size will consume more GPU memory. If there is not enough memory, it will result in an error, and if you're just barely avoiding an error, the learning speed will decrease. It would be a good idea to adjust while checking the amount of memory used with the Task Manager or the `nvidia-smi` command.

By the way, a batch means "a chunk of data".

## Learning Rate

Roughly speaking, it represents how much to change per step. If you specify a large value, learning will progress that much faster, but the model may break from changing too much, or it may not reach the optimal state. If you specify a small value, the learning speed will be slow, and it may also not reach the optimal state.

The learning rate differs greatly between fine tuning, DreamBooth, and LoRA, and it also varies depending on the training data, the model to be trained, the batch size, and the number of steps. Start with a general value and increase or decrease it while watching the learning state.

By default, the learning rate is fixed throughout the training. How to change the learning rate can be determined by specifying a scheduler, so the results will change depending on that.

## Epoch

When all the training data has been learned (i.e., one round of data), it is one epoch. If you specify the number of repetitions, one round of data after that repetition is one epoch.

The number of steps per epoch is basically `number of data / batch size`, but it will slightly increase when using Aspect Ratio Bucketing (since data from different buckets cannot be in the same batch, the number of steps increases).

## Aspect Ratio Bucketing

In addition to being trained at 512\*512, Stable Diffusion v1 also trains at resolutions like 256\*1024 and 384\*640. This reduces the part that is trimmed, and it is expected that the relationship between captions and images will be learned more accurately.

Also, because it learns at any resolution, there is no need to unify the aspect ratio of the image data in advance.

It can be enabled or disabled in the settings, but it is enabled in the examples of the configuration file described so far (`true` is set).

The training resolution is adjusted and created in increments of 64 pixels (default, changeable) vertically and horizontally, within the range not exceeding the area (memory usage) of the given resolution.

In machine learning, it is common to unify all input sizes, but there is no particular constraint, and it is actually okay if they are unified within the same batch. Bucketing, as NovelAI refers to it, seems to mean pre-classifying the training data into learning resolutions according to the aspect ratio. Then, by creating batches with images within each bucket, the image size of the batch is unified.

# Previous Specification Format (Specifying from the Command Line without Using a Configuration File)

This is a method of specifying from command-line options without specifying a `.toml` file. There are DreamBooth class+identifier method, DreamBooth caption method, and fine tuning method.

## DreamBooth, class+identifier method

The number of repetitions is specified by the folder name. The `train_data_dir` option and `reg_data_dir` option are also used.

### Step 1. Preparation of Training Images

Create a folder to store the training images. __Within this__, create directories with the following names:

```
<number of repetitions>_<identifier> <class>
```

Don't forget the underscore (`_`) between them.

For example, for a prompt of "sls frog" and 20 repetitions of data, the name would be "20_sls frog". It would look like this:

![image](https://user-images.githubusercontent.com/52813779/210770636-1c851377-5936-4c15-90b7-8ac8ad6c2074.png)

### Training with Multiple Classes, Multiple Targets (Identifiers)

The method is simple, just prepare multiple folders named ``number of repetitions_<identifier> <class>`` in the training image folder, and similarly, prepare multiple folders named ``number of repetitions_<class>`` in the regularization image folder.

For example, if you want to learn "sls frog" and "cpc rabbit" at the same time, it would look like this:

![image](https://user-images.githubusercontent.com/52813779/210777933-a22229db-b219-4cd8-83ca-e87320fc4192.png)

If there is only one class and multiple targets, one regularization image folder is fine. For example, if there are character A and character B in 1girl, do as follows:

- train_girls
  - 10_sls 1girl
  - 10_cpc 1girl
- reg_girls
  - 1_1girl

### Step 2. Preparation of Regularization Images

This is the procedure for using regularization images.

Create a folder to store the regularization images. __Within this__, create directories named ``<number of repetitions>_<class>``.

For example, for a prompt of "frog" and no data repetition (only once), it would look like this:

![image](https://user-images.githubusercontent.com/52813779/210770897-329758e5-3675-49f1-b345-c135f1725832.png)

### Step 3. Execution of Training

Run each training script. Specify the parent folder of the aforementioned training data with the `--train_data_dir` option (__not the folder containing the images, but its parent__), and specify the folder for the regularization images with the `--reg_data_dir` option (__not the folder containing the images, but its parent__).

## DreamBooth, Caption Method

If you place a file with the same filename as the image and an extension .caption (changeable by option) in the training image and regularization image folders, the script will read the caption from that file and use it as a prompt for training.

* The folder name (identifier class) will no longer be used for training those images.

The default extension for the caption file is .caption. You can change this with the `--caption_extension` option of the training script. With the `--shuffle_caption` option, you can shuffle each part of the caption separated by commas during training.

## Fine Tuning Method

Up to the creation of metadata is the same as when using a configuration file

. Specify the metadata file with the `in_json` option.

# Sample Output During Training

You can check the progress of the training by generating images with the model during training. Specify the following options in the training script.

- `--sample_every_n_steps` / `--sample_every_n_epochs`

    Specify the number of steps or epochs for sample output. A sample is output for each specified number. If both are specified, the number of epochs is given priority.

- `--sample_prompts`

    Specify the file for sample output prompts.

- `--sample_sampler`

    Specify the sampler to use for sample output. You can choose from `'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'`.

You need to prepare a text file with prompts written in advance to perform sample output. Write one prompt per line.

For example, it looks like this:

```txt
# prompt 1
masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

Lines starting with `#` are comments. You can specify options for the generated image with `--` + lower-case English letters. The following can be used:

- `--n` Everything up to the next option is treated as a negative prompt.
- `--w` Specify the width of the generated image.
- `--h` Specify the height of the generated image.
- `--d` Specify the seed of the generated image.
- `--l` Specify the CFG scale of the generated image.
- `--s` Specify the number of steps during generation.


# Commonly Used Options in Each Script

There may be times when the documentation hasn't caught up with the script updates. In those cases, please check the available options with the `--help` option.

## Specifying the Model to be Used for Training

- `--v2` / `--v_parameterization`
    
    If you are using Hugging Face's stable-diffusion-2-base or a fine-tuning model derived from it (i.e., models that are instructed to use `v2-inference.yaml` during inference) as your training target model, specify the `--v2` option. If you're using stable-diffusion-2 or 768-v-ema.ckpt and their fine-tuning models (i.e., models that use `v2-inference-v.yaml` during inference), please specify both `--v2` and `--v_parameterization` options.

    Stable Diffusion 2.0 has made significant changes in the following areas:

    1. The tokenizer used
    2. The Text Encoder used and the output layer used (2.0 uses the second-to-last layer)
    3. The output dimension of the Text Encoder (768->1024)
    4. The structure of U-Net (e.g., the number of heads in CrossAttention)
    5. v-parameterization (apparently the sampling method has changed)

    Of these, 1-4 are adopted

 in base, and 1-5 are adopted in the version without base (768-v). The v2 option enables 1-4, and the v_parameterization option enables 5.

- `--pretrained_model_name_or_path` 
    
    Specify the model to be used as the basis for additional training. You can specify Stable Diffusion's checkpoint files (.ckpt or .safetensors), model directories on Diffusers' local disk, or Diffusers' model IDs (such as "stabilityai/stable-diffusion-2").
## Training Settings

- `--output_dir` 

    Specifies the folder to save the model after training.
    
- `--output_name` 
    
    Specifies the filename of the model excluding the extension.
    
- `--dataset_config` 

    Specifies the `.toml` file describing the dataset configuration.

- `--max_train_steps` / `--max_train_epochs`

    Specifies the number of steps or epochs to train. If both are specified, the number of epochs is prioritized.

- `--mixed_precision`

    To save memory, train using mixed precision. Specify like `--mixed_precision="fp16"`. Compared to no mixed precision (default), the accuracy may be lower, but the amount of GPU memory required for training is significantly reduced.
    
    (In the RTX30 series and later, you can also specify `bf16`. Please match the setting you made in accelerate during environment setup).
    
- `--gradient_checkpointing`

    Reduces the amount of GPU memory required for training by calculating weights during training bit by bit instead of all at once. Turning this on or off does not affect the accuracy, but since turning it on allows for a larger batch size, there is an impact in that regard.
    
    Also, while generally turning it on will slow down the speed, since it allows for a larger batch size, the total training time may actually be faster.

- `--xformers` / `--mem_eff_attn`

    If you specify the xformers option, it will use xformers' CrossAttention. If you have not installed xformers or if an error occurs (depending on the environment, such as when `mixed_precision="no"`), you can specify the `mem_eff_attn` option to use the memory-efficient version of CrossAttention (it is slower than xformers).

- `--clip_skip`
    
    If you specify `2`, it will use the output from the second-to-last layer of the Text Encoder (CLIP). If you specify 1 or omit the option, it will use the last layer.

    *Please do not specify this during training of SD2.0 as it uses the second to last layer by default.

    If the model you are training has originally been trained to use the second layer, it would be good to specify 2.

    If it has been using the last layer instead, the whole model has been trained on that assumption. Therefore, re-training using the second-to-last layer may require a certain amount of teacher data and a longer training period to achieve the desired training results.

- `--max_token_length`

    The default is 75. By specifying `150` or `225`, you can extend the token length and train. Please specify this when training with long captions.
    
    However, the specification of token extension during training is slightly different from that of Mr. Automatic1111's Web UI (such as the division specification), so if not necessary, it is recommended to train with 75.

    Similar to clip_skip, if you train with a length different from the training state of the model, a certain amount of teacher data and a longer training period may be required.
- `--weighted_captions`

    If specified, weighted captions equivalent to those in Automatic1111's Web UI will be activated. This can be used for training except for "Textual Inversion and XTI". It is also effective for token strings of the DreamBooth technique, not just captions.

    The notation for weighted captions is almost the same as the Web UI, where (abc), [abc], (abc:1.23), etc. can be used. Nesting is also possible. If a comma is included in the parentheses, the correspondence of parentheses will be strange in the shuffle/dropout of the prompt, so please do not include a comma in the parentheses.

- `--persistent_data_loader_workers`

    If specified in a Windows environment, the waiting time between epochs will be significantly shortened.

- `--max_data_loader_n_workers`

    Specifies the number of processes for data loading. The more processes, the faster the data reading and the more efficient use of the GPU, but it consumes main memory. The default is "the smaller of `8` or `the number of CPU concurrent threads -1`", so if there is not enough main memory, or if the GPU usage rate is around 90%, please lower it to about `2` or `1` while looking at those values.

- `--logging_dir` / `--log_prefix`

    Options related to saving the training log. Please specify the log storage folder in the logging_dir option. Logs in TensorBoard format will be saved.

    For example, if you specify --logging_dir=logs, a logs folder will be created in the working folder, and the log will be saved in the date folder in it.
    Also, if you specify the --log_prefix option, the specified string will be added before the date. Please use it for identification, such as "--logging_dir=logs --log_prefix=db_style1_".

    To check the log in TensorBoard, open another command prompt and enter as follows in the working folder.

    ```
    tensorboard --logdir=logs
    ```

    (I think tensorboard will be installed along with the environment setup, but if it is not installed, please enter `pip install tensorboard`.)

    After that, open the browser and access http://localhost:6006/ to see it.

- `--log_with` / `--log_tracker_name`

    These are options related to saving the training log. In addition to `tensorboard`, saving to `wandb` is also possible. For details, please see [PR#428](https://github.com/kohya-ss/sd-scripts/pull/428).

- `--noise_offset`

    This is an implementation of the article here: https://www.crosslabs.org//blog/diffusion-with-offset-noise
    
    The generation results of overall dark, bright images may improve. It seems to be effective in LoRA training as well. A value of about `0.1` seems to be good.

- `--adaptive_noise_scale` (Experimental option)

    This option automatically adjusts the Noise offset value according to the absolute value of the average value of each channel of latents. It is activated by specifying it along with `--noise_offset`. The Noise offset value is calculated as `noise_offset + abs(mean(latents, dim=(2,3))) * adaptive_noise_scale`. Since the latent is close to a normal distribution, it might be good to specify a value around 1/10 to the same degree as the noise offset.

    Negative values can also be specified, and in that case, the noise offset is clipped to 0 or more.

- `--multires_noise_iterations` / `--multires_noise_discount`
    


    This is the configuration for Multi resolution noise (pyramid noise). For details, please refer to [PR#471](https://github.com/kohya-ss/sd-scripts/pull/471) and this page [Multi-Resolution Noise for Diffusion Model Training](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2).
    
    If you specify a numerical value for `--multires_noise_iterations`, it will be activated. A value of about 6~10 seems to be good. Please specify a value of about 0.1~0.3 for `--multires_noise_discount` (recommended by the PR author in the case of relatively small datasets such as LoRA training), or about 0.8 (recommended in the original article) (default is 0.3).

- `--debug_dataset`

    By attaching this option, you can check what kind of image data and captions will be used for training before starting the training. Press the Esc key to exit and return to the command line. Press the `S` key to go to the next step (batch), and the `E` key to proceed to the next epoch.

    * Please note that images will not be displayed in a Linux environment (including Colab).

- `--vae`

    If you specify either the Stable Diffusion checkpoint, VAE checkpoint file, Diffuses model or VAE (either local or Hugging Face model ID can be specified) in the vae option, you will use that VAE for training (at the time of caching latents or obtaining latents during training).

    In DreamBooth and fine tuning, the saved model will be one that incorporates this VAE.

- `--cache_latents` / `--cache_latents_to_disk`

    Caches the output of VAE in main memory to reduce VRAM usage. You will not be able to use augmentations other than `flip_aug`. Also, the overall training speed will be slightly faster.

    If you specify cache_latents_to_disk, the cache will be saved to disk. The cache will be valid even if you terminate the script and start it again.

- `--min_snr_gamma`

    Specifies the Min-SNR Weighting strategy. For details, please see [here](https://github.com/kohya-ss/sd-scripts/pull/308). In the paper, `5` is recommended.
    
## Settings for Saving the Model

- `--save_precision`

    Specifies the data precision at the time of saving. If you specify either float, fp16, or bf16 in the save_precision option, the model will be saved in that format (this is not valid when saving the model in Diffusers format for DreamBooth or fine tuning). Use this option if you want to reduce the size of the model.

- `--save_every_n_epochs` / `--save_state` / `--resume`

    If you specify a number in the save_every_n_epochs option, the model will be saved at every specified epoch during training.

    If you also specify the save_state option, it will save the training state, including the state of the optimizer (although you can resume learning from the saved model, you can expect to improve accuracy and reduce learning time compared to that). The saved location will be a folder.
    
    The learning state is output in a folder named `<output_name>-??????-state` (?????? is the number of epochs) in the save location folder. Please use it during long-term learning.

    To resume learning from the saved learning state, use the resume option. Please specify the folder of the learning state (not `output_dir`, but the state folder inside it).

    Please note that due to the specifications of the Accelerator, the number of epochs and the global step are not saved, and they will start from 1 even when you resume.

- `--save_every_n_steps`

    If you specify a number in the save_every_n_steps option, the model will be saved at every specified step during training. You can specify this along with save_every_n_epochs.

- `--save_model_as` (Only for DreamBooth, fine tuning)

    You can choose the model's save format from `ckpt, safetensors, diffusers, diffusers_safetensors`.
    
    Specify like `--save_model_as=safetensors`. When loading in Stable Diffusion format (ckpt or safetensors) and saving in Diffusers format, the missing information is supplemented by dropping the v1.5 or v2.1 information from Hugging Face.

- `--huggingface_repo_id` and others

    If huggingface_repo_id is specified, it will be uploaded to HuggingFace at the same time as saving the model. Please be careful about handling access tokens (please refer to HuggingFace's documentation).

    Please specify other arguments like this:

    -   `--huggingface_repo_id "your-hf-name/your-model" --huggingface_path_in_repo "path" --huggingface_repo_type model --huggingface_repo_visibility private --huggingface_token hf_YourAccessTokenHere`

    If you specify `public` in huggingface_repo_visibility, the repository will be published. If omitted or if you specify `private` (or anything other than public), it will be private.

    If you specify `--save_state_to_huggingface` when the `--save_state` option is specified, the state will also be uploaded.

    If you specify `--resume_from_huggingface` when the `--resume` option is specified, you can download the state from HuggingFace and resume it. The --resume option at that time will be `--resume {repo_id}/{path_in_repo}:{revision}:{repo_type}`.
    
    Example: `--resume_from_huggingface --resume your-hf-name/your-model/path/test-000002-state:main:model`

    If you specify the `--async_upload` option, the upload will be done asynchronously.

## About the Optimizer

- `--optimizer_type`

    Specifies the type of optimizer

. The following can be specified.
    - AdamW: [torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
    - Same as when no option is specified in past versions
    - AdamW8bit: The arguments are the same as above
    - Same as when --use_8bit_adam was specified in past versions
    - Lion: https://github.com/lucidrains/lion-pytorch
    - Same as when --use_lion_optimizer was specified in past versions
    - Lion8bit: The arguments are the same as above
    - SGDNesterov: [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html), nesterov=True
    - SGDNesterov8bit: The arguments are the same as above
    - DAdaptation(DAdaptAdam): https://github.com/facebookresearch/dadaptation
    - DAdaptAdaGrad: The arguments are the same as above
    - DAdaptAdan: The arguments are the same as above
    - DAdaptSGD: The arguments are the same as above
    - AdaFactor: [Transformers AdaFactor](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
    - Any optimizer

- `--learning_rate`

    Specifies the learning rate. The appropriate learning rate varies depending on the learning script, so please refer to the individual explanations.

- `--lr_scheduler` / `--lr_warmup_steps` / `--lr_scheduler_num_cycles` / `--lr_scheduler_power`
  
    These are settings related to the learning rate scheduler.

    With the lr_scheduler option, you can choose the learning rate scheduler from linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, or any scheduler. The default is constant.
    
    With lr_warmup_steps, you can specify the number of warm-up steps for the scheduler.
    
    lr_scheduler_num_cycles is the number of restarts for the cosine with restarts scheduler, and lr_scheduler_power is the polynomial power for the polynomial scheduler.

    For details, please investigate on your own.

    When using any scheduler, please specify the option arguments with `--scheduler_args` just like for any optimizer.
### About specifying the optimizer

Specify the option arguments for the optimizer using the `--optimizer_args` option. You can specify multiple values in the form of key=value. Also, multiple values can be specified for value, separated by commas. For example, to specify arguments for the AdamW optimizer, it would be `--optimizer_args weight_decay=0.01 betas=.9,.999`.

Please check the specifications of each optimizer when specifying option arguments.

Some optimizers have mandatory arguments that are automatically added if omitted (such as the momentum of SGDNesterov). Please check the console output.

The D-Adaptation optimizer automatically adjusts the learning rate. The value specified in the learning rate option is not the learning rate itself but the application rate of the learning rate determined by D-Adaptation, so please normally specify 1.0. If you want to specify half the learning rate for the Text Encoder as U-Net, specify `--text_encoder_lr=0.5 --unet_lr=1.0`.

The AdaFactor optimizer can automatically adjust the learning rate by specifying relative_step=True (it is added by default if omitted). When auto-adjusting, the adafactor_scheduler is forcibly used for the learning rate scheduler. It seems good to specify scale_parameter and warmup_init.

Option specification in the case of auto-adjustment will be like `--optimizer_args "relative_step=True" "scale_parameter=True" "warmup_init=True"`.

Add the option argument `relative_step=False` if you do not want to automatically adjust the learning rate. In that case, it seems recommended to use constant_with_warmup for the learning rate scheduler and not to clip the gradient. Therefore, the arguments will be like `--optimizer_type=adafactor --optimizer_args "relative_step=False" --lr_scheduler="constant_with_warmup" --max_grad_norm=0.0`.

### Using any optimizer

When using an optimizer from `torch.optim`, just specify the class name (`--optimizer_type=RMSprop`, etc.). When using an optimizer from another module, please specify "module name.class name" (`--optimizer_type=bitsandbytes.optim.lamb.LAMB`, etc.).

(This only imports internally, and operation has not been confirmed. Please install the package as needed if required.)

<!--
## Learning with arbitrary size images --resolution
You can train with non-square images. Please specify the resolution as "width, height" like "448,640". Both width and height must be divisible by 64. Please match the size of the training images and regularization images.

Personally, I often generate tall images, so I sometimes train with "448,640".

## Aspect Ratio Bucketing --enable_bucket / --min_bucket_reso / --max_bucket_reso
It becomes valid when you specify the enable_bucket option. Stable Diffusion is trained at 512x512, but in addition to that, it also learns at resolutions like 256x768 and 384x640.

When you specify this option, you do not need to unify the training images and regularization images to a specific resolution. You select the optimal one from several resolutions (aspect ratios) and learn at that resolution. The resolution is in units of 64 pixels, so if the aspect ratio of the original image does not perfectly match, the part that sticks out will be slightly trimmed.

You can specify the minimum size of the resolution with the min_bucket_reso option and the maximum size with the max_bucket_reso. The defaults are 256 and 1024 respectively.
For example, if you specify 384 for the minimum size, you will not use resolutions like 256x1024 and 320x768.
If you increase the resolution to 768x768, you might want to

 specify something like 1280 for the maximum size.

When you enable Aspect Ratio Bucketing, you might want to prepare various resolutions with similar trends to the training images for the regularization images as well.

(This is because the images in one batch will not be biased towards training images and regularization images. I don't think it has much impact though.)

## Augmentation --color_aug / --flip_aug
Augmentation is a method of improving the performance of a model by dynamically changing data during learning. It learns while subtly changing the hue with color_aug and flipping left and right with flip_aug.

Because data is dynamically changed, it cannot be specified simultaneously with the cache_latents option.


## Training with gradients as fp16 (experimental feature) --full_fp16
By specifying the full_fp16 option, you change the gradient from the normal float32 to float16 (fp16) and learn (it seems to become full fp16 learning, not mixed precision).
This seems to allow you to learn with less than 8GB of VRAM for the 512x512 size of SD1.x and less than 12GB of VRAM for the 512x512 size of SD2.x.

Please specify fp16 in the accelerate config in advance and set ``mixed_precision="fp16"`` in the option (it does not work with bf16).

To minimize memory usage, specify the options xformers, use_8bit_adam, cache_latents, gradient_checkpointing, and set train_batch_size to 1.

(If you have some leeway, you should be able to slightly increase the accuracy by gradually increasing train_batch_size.)

# Creating Metadata Files

## Preparing Training Data

Prepare the image data you want to train with, as mentioned earlier, and place it in a folder of your choice.

For example, store images as follows.

![Screenshot of training data folder](https://user-images.githubusercontent.com/52813779/208907739-8e89d5fa-6ca8-4b60-8927-f484d2a9ae04.png)

## Automatic Captioning

Please skip this section if you want to learn only with tags without using captions.

Also, if you are preparing captions manually, please prepare captions in the same directory as the training data images, with the same file name and extensions such as .caption. Each file should be a text file with only one line.

### Captioning with BLIP

In the latest version, there is no need to download BLIP, download weights, or add a virtual environment. It works as it is.

You run make_captions.py in the finetune folder.

```
python finetune\make_captions.py --batch_size <batch_size> <training_data_folder>
```

If you set the batch size to 8 and put the training data in the parent folder's train_data, it will look like this.

```
python finetune\make_captions.py --batch_size 8 ..\train_data
```

Caption files are created in the same directory as the training data images, with the same file name, and the extension .caption.

Please increase or decrease the batch_size according to the VRAM capacity of your GPU. The larger the size, the faster it will be (I think you can increase it a bit even with VRAM 12GB).
You can specify the maximum length of the caption with the max_length option. The default is 75. If you are training the model with a token length of 225, you may want to make it longer.
You can change the extension of the caption with the caption_extension option. The default is .caption (if you set it to .txt, it will conflict with DeepDanbooru described later).

If you have multiple training data folders, please run it for each folder.

Please note that because there is randomness in the inference, the results will change each time you run it. To fix it, please specify a random seed with the --seed option, like `--seed 42`.

For other options, please refer to the help with `--help` (it seems that there is no document summarizing the meaning of parameters, so it seems that you have to look at the source).

By default, the caption file is generated with the extension .caption.

![Folder with generated captions](https://user-images.githubusercontent.com/52813779/208908845-48a9d36c-f6ee-4dae-af71-9ab462d1459e.png)

For example, the following caption is attached.

![Caption and image](https://user-images.githubusercontent.com/52813779/208908947-af936957-5d73-4339-b6c8-945a52857373.png)

## Tagging with DeepDanbooru

Please proceed to "Preprocessing of Caption and Tag Information" if you do not perform the tagging of danbooru tags itself.

Tagging is done with DeepDanbooru or WD14Tagger. WD14Tagger seems to be more accurate. If you are tagging with WD14Tagger, please proceed to the next chapter.

# Setting Up the Environment

Clone DeepDanbooru from https://github.com/KichangKim/DeepDanbooru to your workspace, or download and unzip the zip file. I personally unzipped the file.
Also, from the DeepDanbooru Releases page at https://github.com/KichangKim/DeepDanbooru/releases, download `deepdanbooru-v3-20211112-sgd-e28.zip` from the Assets of "DeepDanbooru Pretrained Model v3-20211112-sgd-e28" and unzip it into the DeepDanbooru folder.

Download from the link below. Click on Assets to open, and download from there.

![DeepDanbooru Download Page](https://user-images.githubusercontent.com/52813779/208909417-10e597df-7085-41ee-bd06-3e856a1339df.png)

Please arrange the directory structure as follows:

![DeepDanbooru Directory Structure](https://user-images.githubusercontent.com/52813779/208909486-38935d8b-8dc6-43f1-84d3-fef99bc471aa.png)

Install the necessary libraries for the Diffusers environment. Move to the DeepDanbooru folder to install (I believe it's essentially just adding tensorflow-io).

```
pip install -r requirements.txt
```

Next, install DeepDanbooru itself.

```
pip install .
```

With this, the tag assignment environment setup is complete.

### Implementing Tagging
Move to the DeepDanbooru folder and run deepdanbooru to perform the tagging.

```
deepdanbooru evaluate <training data folder> --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

If you placed the training data in the parent folder's `train_data`, it would look like this:

```
deepdanbooru evaluate ../train_data --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

Tag files are created in the same directory as the training data images, with the same filename and a .txt extension. It is quite slow because it processes one item at a time.

Execute for each folder if you have multiple training data folders.

This is how it is generated:

![Generated Files of DeepDanbooru](https://user-images.githubusercontent.com/52813779/208909855-d21b9c98-f2d3-4283-8238-5b0e5aad6691.png)

This is how the tags are attached (such a great amount of information...).

![DeepDanbooru Tags and Image](https://user-images.githubusercontent.com/52813779/208909908-a7920174-266e-48d5-aaef-940aba709519.png)

## Tagging with WD14Tagger

This is the procedure to use WD14Tagger instead of DeepDanbooru.

We use the tagger used in Automatic1111's WebUI. We referred to the information on this github page (https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger).

The necessary modules have already been installed during the initial environment setup. Also, the weights will be automatically downloaded from Hugging Face.

### Implementing Tagging

Execute the script to perform tagging.
```
python tag_images_by_wd14_tagger.py --batch_size <batch size> <training data folder>
```

If you placed the training data in the parent folder's `train_data`,

 it would look like this:
```
python tag_images_by_wd14_tagger.py --batch_size 4 ..\train_data
```

On the first launch, the model file will be automatically downloaded to the `wd14_tagger_model` folder (the folder can be changed by option). It will look like this:

![Downloaded Files](https://user-images.githubusercontent.com/52813779/208910447-f7eb0582-90d6-49d3-a666-2b508c7d1842.png)

Tag files are created in the same directory as the training data images, with the same filename, and a .txt extension.

![Generated Tag Files](https://user-images.githubusercontent.com/52813779/208910534-ea514373-1185-4b7d-9ae3-61eb50bc294e.png)

![Tags and Image](https://user-images.githubusercontent.com/52813779/208910599-29070c15-7639-474f-b3e4-06bd5a3df29e.png)

The `thresh` option allows you to specify how much confidence (certainty) is required to apply a tag. The default is 0.35, the same as the WD14Tagger sample. Lowering this value will result in more tags being assigned, but at the cost of lower accuracy.

Please increase or decrease the `batch_size` according to your GPU's VRAM capacity. The larger the size, the faster it will be (I think you can increase it a bit even with 12GB of VRAM). You can change the extension of the tag file with the `caption_extension` option. The default is .txt.

The `model_dir` option allows you to specify the folder where the model will be saved.

Also, if you specify the `force_download` option, the model will be re-downloaded even if there is a save folder.

If you have multiple training data folders, please execute for each folder.


It is forcibly realized by patching the source of PyTorch (confirmed with PyTorch 1.12.1 and 1.13.0). The accuracy drops considerably, and the probability of learning failure during the process increases.
The setting of learning rate and step number also seems to be severe. Please use at your own risk with that in mind.

-->

## Preprocessing of Caption and Tag Information

To make it easier to process from scripts, we will compile captions and tags into a single file as metadata.

### Preprocessing of Captions

To include captions in metadata, please run the following in the working folder (it's not necessary if you're not using captions for training). The `--full_path` option is specified to store the location of the image file in the metadata as a full path. If you omit this option, it will be recorded as a relative path, but you will need to specify the folder separately in the `.toml` file.

```python
python merge_captions_to_metadata.py --full_path <training data folder>
    --in_json <metadata filename to read> <metadata filename>
```

The metadata filename can be anything.
If the training data is train_data, there is no metadata file to read, and the metadata file is meta_cap.json, it would look like this.

```python
python merge_captions_to_metadata.py --full_path train_data meta_cap.json
```

The `caption_extension` option allows you to specify the extension of the caption.

If you have multiple training data folders, please run for each folder with the full_path argument specified.

```python
python merge_captions_to_metadata.py --full_path 
    train_data1 meta_cap1.json
python merge_captions_to_metadata.py --full_path --in_json meta_cap1.json 
    train_data2 meta_cap2.json
```

If you omit in_json, it will read from the metadata file to be written to and overwrite it there.

__*To keep it safe, change the in_json option and destination each time to write to a different metadata file.__

### Preprocessing of Tags

Similarly, we will also compile the tags into metadata (not necessary if you're not using tags for training).
```python
python merge_dd_tags_to_metadata.py --full_path <training data folder> 
    --in_json <metadata filename to read> <metadata filename to write>
```

If you have the same directory structure as before and you're reading from meta_cap.json and writing to meta_cap_dd.json, it would look like this.
```python
python merge_dd_tags_to_metadata.py --full_path train_data --in_json meta_cap.json meta_cap_dd.json
```

If you have multiple training data folders, please run for each folder with the full_path argument specified.

```python
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap2.json
    train_data1 meta_cap_dd1.json
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap_dd1.json 
    train_data2 meta_cap_dd2.json
```

If you omit in_json, it will read from the metadata file to be written to and overwrite it there.

__*To keep it safe, change the in_json option and destination each time to write to a different metadata file.__

### Cleaning of Captions and Tags

By this point, the captions and DeepDanbooru tags have been compiled into the metadata file. However, the captions created by automatic captioning have some inconsistencies and subtleties, and the tags may contain underscores or ratings (in the case of DeepDanbooru), so it might be better to clean the captions and tags using a text editor's replacement function.

For example, if you're training on anime-style girls, the captions may have variations like girl/girls/woman/women. Also, "anime girl" might be more appropriately just "girl".

There is a script prepared for cleaning, so please edit the contents of the script as needed and use it.

(You no longer need to specify the training data folder. All data

 in the metadata will be cleaned.)

```python
python clean_captions_and_tags.py <metadata filename to read> <metadata filename to write>
```

Please note that there's no --in_json. For example, it would look like this.

```python
python clean_captions_and_tags.py meta_cap_dd.json meta_clean.json
```

With this, the preprocessing of captions and tags is complete.
## Pre-acquisition of Latents

*This step is not mandatory. Even if you skip it, you can still acquire latents during the training process. Furthermore, if you are conducting operations like `random_crop` or `color_aug` during training, pre-acquisition of latents is not feasible (as the image changes with each training iteration). Without pre-acquisition, you can still conduct training with the metadata accumulated so far.

We will pre-acquire and save the latent representations of the images onto the disk. This allows us to expedite the training process. Additionally, we will perform bucketing (classifying training data according to aspect ratio).

In the working directory, enter the following:

```python
python prepare_buckets_latents.py --full_path <training data folder>  
    <metadata filename to read> <metadata filename to write> 
    <fine-tuning model name or checkpoint> 
    --batch_size <batch size> 
    --max_resolution <resolution width,height> 
    --mixed_precision <precision>
```

For example, if the model is `model.ckpt`, the batch size is 4, the training resolution is 512*512, the precision is `no` (float32), reading metadata from `meta_clean.json`, and writing to `meta_lat.json`, it would look like this:

```python
python prepare_buckets_latents.py --full_path 
    train_data meta_clean.json meta_lat.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no
```

The latents will be saved in the training data folder in numpy's npz format.

You can specify the minimum and maximum sizes of the resolution with the `--min_bucket_reso` and `--max_bucket_reso` options, respectively. The defaults are 256 and 1024, respectively. For instance, if you set the minimum size to 384, resolutions such as 256*1024 or 320*768 will no longer be used. If you increase the resolution to something like 768*768, it's a good idea to set the maximum size to 1280 or so.

Specifying the `--flip_aug` option will perform left-right flip augmentation, doubling the amount of data. However, if the data is not symmetric (like character appearances or hairstyles), the training may not go well.

(This is a simple implementation that acquires latents for flipped images and saves *_flip.npz files. No specific options need to be specified in `fline_tune.py`. If there are *_flip files, it will randomly load flip and non-flip files.)

You might be able to increase the batch size a bit even with VRAM 12GB.
The resolution is a number divisible by 64 and is specified in "width,height". The resolution is directly related to the memory size during fine tuning. With VRAM 12GB, it is believed that 512,512 is the limit (*). With 16GB, you might be able to raise it to 512,704 or 512,768. However, even at 256,256, it seems tough with VRAM 8GB (because a certain amount of memory is necessary for parameters and optimizer, regardless of resolution).

(* There have also been reports that it worked with 12GB VRAM at a training batch size of 1 and 640,640.)

The results of bucketing will be displayed like this:

![Results of bucketing](https://user-images.githubusercontent.com/52813779/208911419-71c00fbb-2ce6-49d5-89b5-b78d7715e441.png)

If you have multiple training data folders, please run for each

If you have multiple teacher data folders, run it for each folder specifying the full_path argument.
```
python prepare_buckets_latents.py --full_path  
    train_data1 meta_clean.json meta_lat1.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

python prepare_buckets_latents.py --full_path 
    train_data2 meta_lat1.json meta_lat2.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

```
It is possible to set the source and destination to be the same, but it is safer to separate them.

__*It is safer to rewrite the arguments each time and write to a different metadata file.__
