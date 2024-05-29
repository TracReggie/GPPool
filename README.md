# GPPool

The source code of **Training Large-Scale Graph Neural Networks Via Graph Partial Pooling**


## Training
To reproduce the results of GPPool on the Cora dataset, please run the following commands.
```
Conda activate your_conda_env_name
python test.py 
```
If you need to apply GPPool to other datasets, modify the dataset section in test.py and use the hyperparameters suggested in the paper for training.

## Citing

Please cite our work if you find it is useful for you:
```
@article{TrainingLargeScaleGraph,
  title = {Training {{Large-Scale Graph Neural Networks Via Graph Partial Pooling}}},
  author = {Zhang, Qi and Sun, Yanfeng and Wang, Shaofan and Gao, Junbin and Hu, Yongli and Yin, Baocai},
  year = {2024},
  journal = {IEEE Transactions on Big Data},
  note = {Early Access, DOI: 10.1109/TBDATA.2024.3403380}
}

```
