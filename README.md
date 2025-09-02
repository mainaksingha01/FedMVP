# FedMVP: Federated Multi-modal Visual Prompt Tuning for Vision-Language Models [ICCV 2025]
> [Mainak Singha](https://scholar.google.com/citations?user=DvIe72QAAAAJ&hl=en), [Subhankar Roy](https://scholar.google.com/citations?user=YfzgrDYAAAAJ&hl=en), [Sarthak Mehrotra](https://scholar.google.com/citations?user=87yQ-vQAAAAJ&hl=en), [Ankit Jha](https://sites.google.com/view/jha-ankit/), [Moloud Abdar](https://scholar.google.com/citations?user=PwgggdIAAAAJ&hl=en), [Biplab Banerjee](https://biplab-banerjee.github.io/), [Elisa Ricci](https://eliricci.eu/)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2404.00710)

Official implementation of the paper "[FedMVP: Federated Multi-modal Visual Prompt Tuning for Vision-Language Models](https://arxiv.org/pdf/2504.20860)"

## How to install

### Create your environment:

```bash
$ conda create -n fedmvp python=3.8
$ conda activate fedmvp
$ conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch
$ pip install -r requirements.txt
```

### Data preparation:
Please refer to [CoOP](https://github.com/KaiyangZhou/CoOp/tree/main) for data preparation. 

### Training and Evaluation
Please run the command for `training` the model:

```bash
python Launch_FL.py --root YOUR_DATA_PATH --exp_name $1 --model_name $2
```
`--exp_name` specifies the generalization setting e.g. `cross_cls` = base-to-new generalization, `multisource_singletarget_office` = MSST task in the domains of OfficeHome dataset. Please refer to the `config/utils.py` file for more details.
`--model_name` refers to the training model e.g. `fedmvp`

`Testing` is also be done in the same run. The above command can be used to reproduce the test results.


## Citation
If you use our work, please consider citing:
```bibtex
@article{singha2025fedmvp,
  title={FedMVP: Federated Multi-modal Visual Prompt Tuning for Vision-Language Models},
  author={Singha, Mainak and Roy, Subhankar and Mehrotra, Sarthak and Jha, Ankit and Abdar, Moloud and Banerjee, Biplab and Ricci, Elisa},
  journal={arXiv preprint arXiv:2504.20860},
  year={2025}
}
```

## Acknowledgements

Our implementation builds upon the [CoOp](https://github.com/KaiyangZhou/CoOp), [FedTPG](https://github.com/boschresearch/FedTPG) and [classify_by_description](https://github.com/sachit-menon/classify_by_description_release) repositories, and we sincerely thank the authors for making their code publicly available.

