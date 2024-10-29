# SA-RCL
Source code for the paper "SA-RCL: State-Aware Root Cause Localization Method for Anomalies Based on Metrics Data"

### Get Started
1. Download data to `dataset` folder. For the AIOps dataset, you can obtain it from https://github.com/NetManAIOps/AIOps-Challenge-2020-Data. For the WADI and SWAT dataset, you can apply for it by following its official tutorial.
2. Preprocess data to anomaly samples, where each sample contains normal part metrics and abnormal part metrics according to our paper.
2. Train and evaluate. run `python main.py`. The result records are in the `res` directory.


### The overview of SA-RCL

![](https://github.com/lhysgithub/SA-RCL/blob/main/resource/ov.png "")

### Main Result

![](https://github.com/lhysgithub/SA-RCL/blob/main/resource/res.png "")

