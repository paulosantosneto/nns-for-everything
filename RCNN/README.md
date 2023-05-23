# Regions Convolutions Neural Networks (R-CNN)

`Warning` This is my implementation of the article 'Rich feature hierarchies for accurate object detection and semantic segmentation'. Therefore, it is not an official repository of it. If you want to access it, feel free: [Paper](https://arxiv.org/pdf/1311.2524.pdf).

## Code Installation

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

or with conda:

```
conda env create -f enviroment.yml
conda activate rcnn-model
```

## Model Training

```
python3 main.py --mode train --epochs 20 --data_root_dir /path/to/data_root --output_model_dir /path/to/output_model
```
