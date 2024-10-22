curl -o models/model_final_162be9.pkl https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl
curl -o model_configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/detectron2/configs/DensePose_R_50_FPN_s1x.yaml
curl -o model_configs/Base-DensePose-RCNN-FPN.yaml https://dl.fbaipublicfiles.com/detectron2/configs/Base-DensePose-RCNN-FPN.yaml

curl -o models/model_final.pkl https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl
curl -o model_configs/faster_rcnn_R_50_FPN_3x.yaml https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml