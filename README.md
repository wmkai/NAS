## Usage
Please see [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md) for dataset prepare.

For supernet pre-training, run:
```
cd classification;
```
then run:
```
bash run_supernet_fix_backbone.sh;
```
and
```
bash run_supernet_fix_trans.sh;
```

For supernet fine-tuning, run:
```
cd segmentation;
```
then run:
```
bash tools/dist_train_bignas.sh configs/hess_dynamic_supernet/base/hess_4x8_ade20k_fix_backbone.py.py 8 --work-dir /path;
```
and
```
bash tools/dist_train_bignas.sh configs/hess_dynamic_supernet/base/hess_4x8_ade20k_fix_trans.py.py 8 --work-dir /path;
```

To search for the optimal subnet, run:
```
python tools/search_alternately.py configs/hess_dynamic_supernet/base/hess_4x1_ade20k_search.py --eval mIoU;
```

To retrain the subnet on ImageNet, run:
```
cd classification;
```
then run:
```
bash retrain_subnet_base.sh
```

Then run:
```
cd retrain; sh tools/dist_train.sh local_configs/HESS/base/<config-file> <num-of-gpus-to-use> --work-dir /path/to/save/checkpoint
```
to fine-tune the subnet on segmentation dataset.

To evaluate, run:
```
sh tools/dist_test.sh local_configs/HESS/<config-file> <checkpoint-path> <num-of-gpus-to-use>
```


To test the inference speed in mobile device, please refer to [tnn_runtime](tnn_runtime.md).

## Acknowledgement

The implementation is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

The checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/1G1FEkT5zWl6kfOGHwMX5xxaydKAIWmYE?usp=share_link)