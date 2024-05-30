# -HW2
训练方法：
T1:
下载dataset.py、transform.py（数据预处理文件），dlhw3.py（预训练模型训练文件）dlhw4.py（未预训练模型训练文件）以及CUB2011数据集，将三个py文件放在同一文件夹下；需要修改的地方有（两个文件一致）：
数据集文件路径：代码第25行，换为电脑上保存的数据集路径
Tensorboard写出路径：第15行，换为自己设置的路径
断点载入文件：可以将其注释掉，在代码的第89行
断点写出路径和模型保存路径：代码144行左右，可以删除

T2：
Faster r-cnn：复制faster-rcnn_r50_fpn_1x_voc0712.py中的代码粘贴到同名文件下（mmdetection\configs\pascal_voc\faster-rcnn_r50_fpn_1x_voc0712.py）；准备数据集voc2007；复制voc0712.py（数据载入文件，一定要复制！因为显存限制我只使用了voc2007数据集，没有使用2012，不修改里面的设置会报错！）粘贴到同名文件下（mmdetection\configs\_base_\datasets\voc0712.py）；数据集按正常架构组织即可
训练命令：python tools/train.py configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py
测试命令：python tools/test.py configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py work_dirs/faster-rcnn_r50_fpn_1x_voc0712/epoch_4.pth –show（需下载epoch_4.pth）

Yolo：复制yolov3_d53_8xb8-ms-608-273e_coco.py中的代码粘贴到同名文件下（mmdetection\configs\yolo\yolov3_d53_8xb8-ms-608-273e_coco.py），复制schedule_1x.py到mmdetection\configs\_base_\schedules\schedule_1x.py下（否则会报错）
训练命令：python tools/train.py configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py
测试命令：python tools/test.py configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py work_dirs/yolov3_d53_8xb8-ms-608-273e_coco/epoch_14.pth –show


 

