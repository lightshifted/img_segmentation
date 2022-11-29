# Image Segmentation for Shoe Detection

## 1. Project Overview
The aim of this project is to pre-train an instance segmentation model for accurately detecting and delineating each distinct shoe present in an image. 

## 2. Requirements
* Linux or macOS with Python $\geq$ 3.7
* PyTorch $\geq$ 1.8 and torchvision that matches the PyTorch installation.
* OpenCV for visualizations

## 3. Data
We collected 250 license-free images of varying sizes from Flikr and annotated them using VGG Image Annotator. The images were then divided into train and validation sets using a 70/20 split.

## 4. Model
We utilize an implementation of Mask RCNN using a RESNET50 + FPN backbone from Facebook AI Research's [detectron2](https://github.com/facebookresearch/detectron2) library.

![RESNET50 + FPN Architecture](https://i.ibb.co/dWDnb75/Detailed-architecture-of-the-backbone-of-Res-Net-50-FPN-Basic-Stem-down-samples-the-input.png)

The model was trained using 600 iterations with a ROI batch size of 512 per image on a single RTX 3080 GPU.

## 5. Evaluation
### Metrics used
**total_loss:** Weighted sum of the individual losses calculated each iteration.
**loss_cls:** Classification loss in the ROI head. Measures how well the model is at labeling a predicted box with the correct class.
**loss_box_reg:** Measures the loss for box localisation.
**loss_mask:** Measures correctness of predicted binary masks.
**loss_rpn_cls:** Measures how well the RPN is at labelling anchor boxes as foreground or background.
**loss_rpn_loc:** Measures the loss for localisation of the predicted regions in the RPN.

### Results and Performance
| Metric  | Score  |
|---|---|
|total_loss   |0.5602   |
| loss_cls  |  0.1416 |
|loss_box_reg | 0.2277  |
|loss_mask   |  0.1859 |
|loss_rpn_cls   | 0.01095  |
|loss_rpn_loc   |  0.01462 |

### Example Output
![example_1](https://i.ibb.co/zsgwJ6j/results-1.jpg)
![example_2](https://i.ibb.co/Nsxs2dP/results-2.jpg)

## 6. Conclusion
The aim of this project was to pre-train an instance segmentation model for accurately detecting and delineating each distinct shoe present in an image. Future work involves generating image masks for use with Stable Diffusion in-painting model. By combining these two methods, we can then replace detected shoes with the objects specified in a text prompt fed to the text-to-image model. 

## 7. References
### Relevant papers and tutorials
1. [Detectron2: A PyTorch-based modular object detection library](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
2. [Detectron2 Repository](https://github.com/facebookresearch/detectron2)
3. Mask R-CNN [paper](https://arxiv.org/abs/1703.06870)