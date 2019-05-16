# ClassActivationMapping
the implementation of Class Activation Mapping(CAM), Grad-CAM and Grad-CAM++ with pytorch

## Requirements
* python 3.x
* pytorch >= 0.4
* pillow
* numpy
* opencv
* matplotlib

## How to use
You can use the CAM, GradCAM and GradCAMpp class as a model wrapper described in `cam.py`.
Please see `cam_demo.ipynb` for the detail.

## References
* Learning Deep Features for Discriminative Localization, 
  Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba [[paper](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)]
* Grad-CAM: Visual explanations from deep networks via gradient-based localization,
  Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra, [[arXiv](https://arxiv.org/abs/1610.02391)]
* Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks,
  Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader and Vineeth N Balasubramanian[[arXiv](https://arxiv.org/pdf/1710.11063.pdf)]
  
