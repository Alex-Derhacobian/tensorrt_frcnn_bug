# Converting Pytorch Faster-RCNN model to TensorRT
This repository is for debugging the process of porting a Pytorch Torchvision Faster-RCNN model to a TensorRT engine. Please follow the steps below to replicate the errors that we encounter.

**Requirements/Versions**<br />
- pytorch=1.9.0
- torchvision=0.10.0
- CUDA=11.0
- TensorRT=8.0.1

## Replicating Errors
#### Step 1. Save Faster-RCNN model as ONNX model: 
Run `get_frcnn_onnx.py` to save a Torchvision Faster-RCNN model as an ONNX file. If you would like to load a model from a specific path, use the argument `--model_path`. The argument `--dummy_img` is a path to a dummy input used for generating the ONNX file. The dummy input in our case is a 1440 × 1080 JPEG image from our dataset. The dummy image that we used is included in this repository, but a custom with these dimensions will also work.  

We use an opset number of 11. 

#### Step 2. Validate ONNX model
Validate that the onnx model was saved successfully by running `check_model.py`. The code for this script is from the following forum post by Nvidia moderators: https://forums.developer.nvidia.com/t/convert-faster-rcnn-tensorflow-model-to-tensorrt/173071/3

#### Step 3. Build TensorRT Engine using `trtexec` 

`trtexec` is used to build TensorRT engines from the command line. To compile `trtexec`, run `make` in the `<TensorRT root directory>/samples/trtexec` directory. The binary named `trtexec` will be created in the `<TensorRT root directory>/bin` directory.
```
cd <TensorRT root directory>/samples/trtexec
make
```
After you have compiled `trtexec`, go to `<TensorRT root directory>/bin` and run 
```
./trtexec --onnx=<onnx_model_path> --verbose
```
`<onnx_model_path>` is the ONNX model generated in Step 1. Once you run this command, you should see that the model fails on layer **Resize_62**

#### Note: See `stderr.txt` and `stdout.txt` for our standard error and standard output
