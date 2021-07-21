# pytorch quantization workflow with TensorFlow model output
This repo implements quantization-aware training (QAT) under the PyTorch framework.
It requires the following inputs:
- PyTorch model builder
- trained floating PyTorch checkpoint 

## Batch normalization folding
If the model has batch normalization layers, then it needs to be folded into its preceding convolutional layer first.
More details about the folding can be found in quant-with-bn.md.
efn-r192-fold.ipynb contains the technical details about batch normalization folding.


## Quantizaion aware training 
r192_qat.py demonstrates how to perform QAT after batch normalization folding. 
There are two key differences between the QAT and normal PyTorch training.
Firstly, a *backup model* is created in line 232 of r192_qat.py.
It was used to save the model states before inference quantization.
Secondly, there is model *quantization* process using model_quant() function in line 94.
The inference and gradient propagation are performed with the quantized model.
The weight updates are performed on the original backup model before quantization.
The *backup model* is essential to accumulate the small updates,  which could be smaller than the quantization gap.

## Convert to TensorFlow format for deployment
dpu-efn-qat.ipynb shows how to convert the PyTorch model after QAT to Tensorflow format for deployment.
Basically, one needs to rebuild the model in TensorFlow then assign the PyTorch model weights to the blank TF model.
Pytorch and TF have different *memory layouts*(NHWC and NCHW, respectively).
dpu-efn-qat.ipynb took care of the difference with np.transpose() at cell [5].