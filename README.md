# SMAConvFormer: A fault diagnosis model for rotating machinery with high noise and variable operating conditions
* Core codes for the paper:
<br> SMAConvFormer: a fault diagnosis model for rotating machinery with high noise and variable operating conditions
* Journal: Measurement Science and Technology

## Reference
This project is based on the framework from [LiConvFormer](https://github.com/yanshen0210/LiConvFormer-a-lightweight-fault-diagnosis-framework)


## Our operating environment
* Python 3.8
* pytorch  1.10.1
* numpy  1.22.0 (If you get an error when saving data, try lowering your numpy version!)
* and other necessary libs

## Datasets
* [Case1: XJTU gearbox](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)
* [Case2: XJTU spurgear](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)
* [Case3: OU bearing](https://drive.google.com/file/d/1PQnIBKzAu098SAl3DUw0n8AHONynpdb7/view?usp=sharing)
* [Case4: BJTU-Rao](https://drive.google.com/drive/folders/1RlZvFw-v07VvsL2Ni9cS7iFrTPDIhn2r?usp=sharing)
* [Save dataset](https://drive.google.com/file/d/10XQDVN9YqbM7--X3dB55Io1eRLsLmruI/view?usp=sharing)  
  
  
## Guide 
* This repository provides a lightweight fault diagnosis framework. 
* It includes the pre-processing for the data and the model proposed in the paper. 
* `train_val_test.py` is the train&val&test process of all methods.
* You need to load the data in above Datasets link at first, and put them in the `data` folder. Then run in `args_diagnosis.py`
<br> Pay attention to that if you want to run the data pre-process, you need to load [Case1](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing),
[Case2](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing) and [Case3](https://drive.google.com/file/d/1PQnIBKzAu098SAl3DUw0n8AHONynpdb7/view?usp=sharing) in Datasets,
<br> and set --save_dataset (in `args_diagnosis.py`) to True; or you can just load the [Save dataset](https://drive.google.com/file/d/10XQDVN9YqbM7--X3dB55Io1eRLsLmruI/view?usp=sharing), and set --save_dataset to False.
* You can also choose the modules or adjust the parameters of the model to suit your needs.

## Initial learning rate
* Liconvformer: Case1--0.01;  Case2--0.001;  Case3--0.01；Case4--0.01
* SMAConvformer: Case1--0.01;  Case2--0.001;  Case3--0.01;  Case4--0.01
* EWSNet: Case1--0.01;  Case2--0.001;  Case3--0.01;  Case4--0.01
* CLFormer: Case1--0.01;  Case2--0.001;  Case3--0.01;  Case4--0.01
* convoformer_v1_small: Case1--0.001;  Case2--0.001;  Case3--0.001
* mcswint: Case1--0.001;  Case2--0.001;  Case3--0.01;  Case4--0.01
* MobileNet: Case1--0.01;  Case2--0.001;  Case3--0.001;  Case4--0.01
* MobileNetV2: Case1--0.01;  Case2--0.001;  Case3--0.001;  Case4--0.01
* ResNet18: Case1--0.001;  Case2--0.001;  Case3--0.001;  Case4--0.01
* MSResNet: Case1--0.001;  Case2--0.001;  Case3--0.001;  Case4--0.01
## Pakages
* `data` needs loading the Datasets in above links
* `datasets` contians the pre-processing process for the data
* `models` contians 8 methods including the proposed method
* `utils` contians train&val&test processes

## Citation
If our work is useful to you, please cite the following paper, it is the greatest encouragement to our open source work, thank you very much!
```
@paper{
  title = {SMAConvFormer: a fault diagnosis model for rotating machinery with high noise and variable operating conditions},
  author = {Jingwen Wei, Jianhai Yue},
  journal = {Measurement Science and Technology},
  year = {2026},
}
```
