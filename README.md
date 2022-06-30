# Understanding Robust Overfitting of Adversarial Training and Beyond

This branch is developed for understanding robust overfitting in adversarial training, the related paper is as follows:

    Understanding Robust Overfitting of Adversarial Training and Beyond [C]
    Chaojian Yu, Bo Han, Li Shen, Jun Yu, Chen Gong, Mingming Gong, Tongliang Liu
    ICML. 2022.

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.7.3
- torch = 1.2.0
- torchvision = 0.4.0

## What is in this repository
Codes for our Minimum Loss Constrained Adversarial Training (MLCAT):
- In `train_step1_epsilon_2.py` and `train_step1_epsilon_8.py`, the codes for the data distribution of non-overfit (weak adversary) and overfitted (strong adversary) adversarial training, respectively.
- In `train_step2_remove_0_15.py` and `train_step3_remove_transformed_0_15.py`, the codes for data ablation adversarial training.
- In `train_step4_scale_15.py`, the codes for the realization of MLCAT through loss scaling.
- In `train_step5_weight_15.py`, the codes for the realization of MLCAT through weight perturbation.
- In `eval_aa.py`, the codes for robustness evaluation on AA.

## How to use it

For the data distribution of non-overfit and overfitted AT, run codes as follows, 
```
python train_step1_epsilon_2.py

python train_step1_epsilon_8.py
``` 

For the causes of robust overfitting, run codes as follows,
```
python train_step2_remove_0_15.py

python train_step3_remove_transformed_0_15.py
```

For the realizations of MLCAT, run codes as follows,
```
python train_step4_scale_15.py

python train_step5_weight_15.py
```

For robustness evaluation on AA, you first need installation `pip install git+https://github.com/fra31/auto-attack`, and then run codes as follows,
```
python eval_aa.py
```

## Citation
If you find our code useful, please consider citing our work:

    @article{yu2022understanding,
      title={Understanding Robust Overfitting of Adversarial Training and Beyond},
      author={Yu, Chaojian and Han, Bo and Shen, Li and Yu, Jun and Gong, Chen and Gong, Mingming and Liu, Tongliang},
      journal={arXiv preprint arXiv:2206.08675},
      year={2022}
    }


## Reference Code
[1] AT: https://github.com/locuslab/robust_overfitting

[2] AWP: https://github.com/csdongxian/AWP

[3] RWP: https://github.com/ChaojianYu/Robust-Weight-Perturbation

[4] AutoAttack: https://github.com/fra31/auto-attack
