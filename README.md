# S13ERAV3 Project

## Overview
S13ERAV3 is a reverse-engineered language model based on the configuration extracted from SmoLLM-135M. This repository provides the implementation, training scripts, and utilities to run and fine-tune the language model efficiently. The model is designed to handle various natural language processing tasks, including text generation and tokenization.

## Features
- Reverse-engineered implementation based on SmoLLM-135M.
- Supports RoPE (Rotary Position Embedding) for positional encoding.
- Configurable transformer blocks with custom attention mechanisms.
- Modularized training and evaluation scripts for easy customization.
- Checkpointing system to save and resume training progress.

## Model Summary
```plaintext
Model Name: S13ERAV3
Base Architecture: Transformer-based
Total trainable weights : 134,515,008 (Emeddinng & lm_head have shared weights hence, 162,826,560 - 28,311,552 = 134,515,008)

# Share weights between embedding and lm_head
self.lm_head.weight = self.embedding.weight

===================================================================================================================
Layer (type:depth-idx)                   Output Shape              Param #                   Trainable
===================================================================================================================
SmollM                                   [1, 2048, 49152]          --                        True
├─Embedding: 1-1                         [1, 2048, 576]            28,311,552                True
├─LlamaRotaryEmbedding: 1-2              [1, 2048, 64]             --                        --
├─ModuleList: 1-3                        --                        --                        True
│    └─TransformerBlock: 2-1             [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-2             [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-3             [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-4             [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-5             [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-6             [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-7             [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-8             [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-9             [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-10            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-11            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-12            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-13            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-14            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-15            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-16            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-17            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-18            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-19            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-20            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-21            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-22            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-23            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-24            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-25            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-26            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-27            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-28            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-29            [1, 2048, 576]            3,540,096                 True
│    └─TransformerBlock: 2-30            [1, 2048, 576]            3,540,096                 True
├─LlamaRMSNorm: 1-4                      [1, 2048, 576]            576                       True
├─Linear: 1-5                            [1, 2048, 49152]          28,311,552                True
===================================================================================================================
Total params: 162,826,560
Trainable params: 162,826,560
Non-trainable params: 0
Total mult-adds (M): 162.83
===================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 3938.45
Params size (MB): 651.31
Estimated Total Size (MB): 4589.77
===================================================================================================================
```

## Huggingface Demo
[Space Demo](https://huggingface.co/spaces/piyushgrover/SmoLLM-135M)
## Training Logs
Sample logs from training the model:
```Step 0, Loss: 11.318713188171387
Checkpoint saved at checkpoints/checkpoint_0.pth
Validation: (Step 0), Generated text: This is a, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat, mat
Step 1, Loss: 10.951064109802246
Step 2, Loss: 9.568182945251465
Step 3, Loss: 9.793399810791016
Step 4, Loss: 9.114611625671387
Step 5, Loss: 9.143531799316406
Step 6, Loss: 9.042845726013184
Step 7, Loss: 9.735605239868164
Step 8, Loss: 8.973363876342773
Step 9, Loss: 8.684163093566895
Step 10, Loss: 8.158095359802246
Step 11, Loss: 8.261303901672363
Step 12, Loss: 8.213305473327637
.
.
.
Step 300, Loss: 6.2050557136535645
Step 301, Loss: 6.36290168762207
Step 302, Loss: 5.793313503265381
Step 303, Loss: 6.272426128387451
Step 304, Loss: 6.5965094566345215
Step 305, Loss: 6.027985095977783
Step 306, Loss: 6.154433250427246
Step 307, Loss: 6.288233280181885
Step 308, Loss: 5.979233741760254
Step 309, Loss: 5.937126159667969
Step 310, Loss: 6.1712327003479
.
.
Step 496, Loss: 6.061275482177734
Step 497, Loss: 5.81633996963501
Step 498, Loss: 5.850150108337402
Step 499, Loss: 5.806085109710693
Step 500, Loss: 6.1612443923950195
Checkpoint saved at checkpoints/checkpoint_500.pth
Validation: (Step 500), Generated text: This is a to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to to
Step 501, Loss: 6.126720905303955
Step 502, Loss: 6.442710876464844
Step 503, Loss: 6.062013626098633
Step 504, Loss: 5.968945026397705
Step 505, Loss: 6.05215311050415
.
.
.
Step 776, Loss: 5.558218002319336
Step 777, Loss: 6.148859977722168
Step 778, Loss: 6.586211681365967
Step 779, Loss: 6.240291118621826
Step 780, Loss: 6.07138729095459
Step 781, Loss: 5.9661865234375
Step 782, Loss: 5.762390613555908
.
.
.
Step 992, Loss: 6.257589340209961
Step 993, Loss: 6.392139911651611
Step 994, Loss: 6.0093231201171875
Step 995, Loss: 6.030836582183838
Step 996, Loss: 5.812777996063232
Step 997, Loss: 5.853862285614014
Step 998, Loss: 5.840930938720703
Step 999, Loss: 6.427679538726807
Step 1000, Loss: 6.027270793914795
Checkpoint saved at checkpoints/checkpoint_1000.pth
Validation: (Step 1000), Generated text: This is a..................................................
Step 1001, Loss: 6.439802646636963
Step 1002, Loss: 6.065145492553711
Step 1003, Loss: 5.90217924118042
Step 1004, Loss: 6.3704609870910645
Step 1005, Loss: 5.978586196899414
Step 1006, Loss: 6.24245023727417
.
.
.
Step 1498, Loss: 5.971211910247803
Step 1499, Loss: 5.689746856689453
Step 1500, Loss: 5.946279525756836
Checkpoint saved at checkpoints/checkpoint_1500.pth
Validation: (Step 1500), Generated text: This is a two,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Step 1501, Loss: 5.783288478851318
Step 1502, Loss: 6.136028289794922
Step 1503, Loss: 6.184974193572998
Step 1504, Loss: 6.25680685043335
Step 1505, Loss: 6.021946430206299
Step 1506, Loss: 6.625786304473877
Step 1507, Loss: 6.101223945617676
Step 1508, Loss: 6.461009502410889
.
.
.
Step 1998, Loss: 6.259542942047119
Step 1999, Loss: 6.046116352081299
Step 2000, Loss: 5.944657325744629
Checkpoint saved at checkpoints/checkpoint_2000.pth
Validation: (Step 2000), Generated text: This is a..................................................
Step 2001, Loss: 6.1490349769592285
Step 2002, Loss: 6.15787935256958
Step 2003, Loss: 6.399116516113281
Step 2004, Loss: 5.692773342132568
Step 2005, Loss: 5.862493515014648
Step 2006, Loss: 6.071348667144775
Step 2007, Loss: 6.427814483642578
Step 2008, Loss: 6.118124485015869
Step 2009, Loss: 5.78250789642334
Step 2010, Loss: 5.872506618499756
Step 2011, Loss: 6.302607536315918
Step 2012, Loss: 6.24782657623291
Step 2013, Loss: 5.500602722167969
Step 2014, Loss: 5.817261219024658
Step 2015, Loss: 6.2115631103515625
Step 2016, Loss: 6.1064372062683105
Step 2017, Loss: 5.977082252502441
Step 2018, Loss: 6.142667293548584
.
.
.
Step 2497, Loss: 5.906391143798828
Step 2498, Loss: 5.681766033172607
Step 2499, Loss: 5.827874660491943
Step 2500, Loss: 5.777903079986572
Checkpoint saved at checkpoints/checkpoint_2500.pth
Validation: (Step 2500), Generated text: This is a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a
Step 2501, Loss: 5.989109516143799
Step 2502, Loss: 6.165022850036621
Step 2503, Loss: 5.925332546234131
Step 2504, Loss: 5.984160423278809
Step 2505, Loss: 5.533101558685303
Step 2506, Loss: 5.6322855949401855
Step 2507, Loss: 5.835229873657227
Step 2508, Loss: 5.951961994171143
Step 2509, Loss: 5.861036777496338
Step 2510, Loss: 6.093997001647949
.
.
.
Step 2996, Loss: 5.997471332550049
Step 2997, Loss: 5.998309135437012
Step 2998, Loss: 5.981715679168701
Step 2999, Loss: 5.993240833282471
Step 3000, Loss: 6.113542556762695
Checkpoint saved at checkpoints/checkpoint_3000.pth
Validation: (Step 3000), Generated text: This is a was've upon a was've upon a was've upon a was've upon a was've upon a was've upon a was've upon a was've upon a was've upon a was've upon a was've upon a was've upon a was've
Step 3001, Loss: 5.934725284576416
Step 3002, Loss: 5.815080165863037
Step 3003, Loss: 5.989375114440918
Step 3004, Loss: 5.482710361480713
Step 3005, Loss: 5.966794967651367
Step 3006, Loss: 6.2525954246521
Step 3007, Loss: 5.977765083312988
Step 3008, Loss: 5.942745208740234
Step 3009, Loss: 6.181905746459961
Step 3010, Loss: 5.8756303787231445
Step 3011, Loss: 5.7347798347473145
.
.
.
Step 3497, Loss: 5.382612228393555
Step 3498, Loss: 6.023163795471191
Step 3499, Loss: 5.571175575256348
Step 3500, Loss: 5.683298110961914
Checkpoint saved at checkpoints/checkpoint_3500.pth
Validation: (Step 3500), Generated text: This is a of lived few time of lived few time of lived few time of lived few time of lived few time of lived few time of lived few time of lived few time of lived few time of lived few time of lived few time of lived few time of lived
Step 3501, Loss: 6.165890693664551
Step 3502, Loss: 6.059831619262695
Step 3503, Loss: 5.936408042907715
Step 3504, Loss: 6.249509334564209
Step 3505, Loss: 5.887147426605225
Step 3506, Loss: 6.206728458404541
Step 3507, Loss: 6.132350921630859
Step 3508, Loss: 6.010180950164795
Step 3509, Loss: 5.85306978225708
Step 3510, Loss: 6.130023956298828
.
.
.
Step 3996, Loss: 6.289129734039307
Step 3997, Loss: 6.150471210479736
Step 3998, Loss: 5.6754655838012695
Step 3999, Loss: 6.160754680633545
Step 4000, Loss: 5.6676483154296875
Checkpoint saved at checkpoints/checkpoint_4000.pth
Validation: (Step 4000), Generated text: This is a time upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon land upon
Step 4001, Loss: 6.013797283172607
Step 4002, Loss: 5.99492883682251
Step 4003, Loss: 6.2151384353637695
Step 4004, Loss: 6.324347972869873
Step 4005, Loss: 5.7890849113464355
Step 4006, Loss: 5.827409267425537
.
.
.
Step 4497, Loss: 5.76957893371582
Step 4498, Loss: 5.8743062019348145
Step 4499, Loss: 5.867910385131836
Step 4500, Loss: 6.294216632843018
Checkpoint saved at checkpoints/checkpoint_4500.pth
Validation: (Step 4500), Generated text: This is a into, going big in to the through to the through to the through to the through to the through to the through to the through to the through to the through to the through to the through to the through to the through to the through to the through
Step 4501, Loss: 6.257639408111572
Step 4502, Loss: 6.179189682006836
Step 4503, Loss: 6.471646785736084
Step 4504, Loss: 6.023241996765137
.
.
.
Step 4652, Loss: 6.321015357971191
Step 4653, Loss: 6.364404678344727
Step 4654, Loss: 5.880470275878906
Step 4655, Loss: 6.003664016723633
Step 4656, Loss: 6.003395080566406
Step 4657, Loss: 6.217809677124023
Step 4658, Loss: 5.961635112762451
.
.
.
Step 4995, Loss: 5.691064834594727
Step 4996, Loss: 5.791865348815918
Step 4997, Loss: 6.42966365814209
Step 4998, Loss: 5.75705623626709
Step 4999, Loss: 6.206450462341309
Step 5000, Loss: 6.107337474822998
Checkpoint saved at checkpoints/checkpoint_5000.pth
Validation: (Step 5000), Generated text: This is a time: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title: Title:
Reached maximum training steps.
Training complete.
```

## Setup Instructions

### Requirements
- Python 3.8+
- PyTorch 1.13+
- Transformers Library (Hugging Face)
- CUDA (Optional, for GPU support)

Install the required packages using:
```bash
pip install -r requirements.txt
```

### Training the Model
1. Configure the `config.yaml` file with your desired model and training parameters.
2. Run the training script using:
```bash
python train.py --config config.yaml
```

#### Checkpointing
The training script supports checkpointing to save and resume training progress. Checkpoints include the model state, optimizer state, scheduler state, and current training step.

To resume training:
```bash
python train.py --config config.yaml --resume checkpoint_<step>.pt
```

#### Example: Saving Checkpoints
During training, checkpoints are saved using the following code:
```python
checkpoint_path = os.path.join(config['checkpoints']['checkpoints_path'], f"checkpoint_{step}.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'step': step
}, checkpoint_path)
```

### Generating Text
To use the model for text generation:
```python
from transformers import AutoTokenizer
from model import SmollM
from utils import generate_tokens

# Load the model
model = SmollM(config['model']['model_config'])

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['tokenizer_name_or_path'])

# Generate text
sample_prompt = "This is the demo prompt for"
generated_text = generate_tokens(model, tokenizer, sample_prompt, max_length=50, device=device)

print("Generated Text:", generated_text)
```

## Usage
The project supports the following tasks:
- Text generation
- Fine-tuning on custom datasets
- Evaluation on standard benchmarks

### Running Inference
Inference can be performed using the pre-trained model weights:
```bash
python inference.py --config config.yaml --input "This is a sample prompt"
```

### Fine-Tuning
Fine-tune the model on your custom dataset:
1. Prepare the dataset in the required format.
2. Update `config.yaml` with dataset paths and training parameters.
3. Run:
```bash
python train.py --config config.yaml
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

Contributions and feedback are welcome! If you encounter issues or have feature requests, feel free to open an issue or submit a pull request.
