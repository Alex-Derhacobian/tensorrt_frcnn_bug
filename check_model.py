import sys
import argparse
import onnx

def check_model(onnx_path)
    model = onnx.load(filename)
    onnx.checker.check_model(model)
    print("Check successfull")
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', default = 'frcnn_opset=11.onnx')
    args = parser.parse_args()
    check_model(args.onnx_path)
