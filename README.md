## Multimodal genre prediction using OCR

<img width="400" alt="demo screenshot" src="https://i.imgur.com/d8jPKiQ.png">

#### CRAFT text extraction code: 

https://github.com/clovaai/CRAFT-pytorch

https://github.com/clovaai/deep-text-recognition-benchmark


### Install dependencies
#### Requirements
- PyTorch>=0.4.1
- torchvision>=0.2.1
- opencv-python>=3.4.2
- Flask >= 1.1.2
- requirements.txt
```
pip install -r requirements.txt
```

### Instructions
#### Load pre-trained weights
- Insert [craft_mlt_25k.pth](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view) into the CRAFT-pytorch folder
- Insert [TPS-ResNet-BiLSTM-Attn.pth](https://drive.google.com/file/d/1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9/view?usp=sharing) into deep-text-recognition-benchmark folder
- Insert [best_gmu.pth](https://drive.google.com/file/d/1-bibnWb-pSrl9Io6kNTCJr8tNzeUVfPW/view), [best_model_mmu](https://drive.google.com/file/d/1FoACqjLxUwkXbvrYNIe-bkz2UPaptGMs/view), and [best_img_model](https://drive.google.com/file/d/13BLlR_XnIVk5RUJtKDl1rQz3VIDqAUIC/view) into genre-demo folder

#### Run demo: 

```shell
cd genre-demo
python3 main.py
```
