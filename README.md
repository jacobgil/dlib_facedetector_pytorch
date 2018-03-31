## Dlib's face detector ported to PyTorch


![](https://raw.githubusercontent.com/jacobgil/dlib_facedetector_pytorch/master/positive_images/13_4.jpg)![](https://raw.githubusercontent.com/jacobgil/dlib_facedetector_pytorch/master/positive_images/18_3.jpg)![](https://raw.githubusercontent.com/jacobgil/dlib_facedetector_pytorch/master/positive_images/18_6.jpg)![](https://raw.githubusercontent.com/jacobgil/dlib_facedetector_pytorch/master/positive_images/1_5.jpg)![](https://raw.githubusercontent.com/jacobgil/dlib_facedetector_pytorch/master/negative_images/0_4.jpg)

This repository contains code to:

 - Convert a Dlib model in XML format, to a PyTorch sequential object.
 This happens in dlib_torch_converter.py, in get_model.
 - Run multi scale face detection in a webcam.
``` python webcam_example.py face.xml```
 - Hallucinate faces using activation maximization.
``` python webcam_example.py face.xml```

