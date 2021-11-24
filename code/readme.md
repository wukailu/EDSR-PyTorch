# Introduction

This is the demo code for CVPR2022 paper #5768. 

# Environments

* Codes are based on Pytorch-Lightning framework.
* You can setup the environments as follows:
  1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 
  2. Create a new environment with `python=3.7`.
  3. Install packages in `requirements.txt`.
* To reproduce the main experiments, you need to download `DIV2K`, `Set5`, `Set14`, `B100`, and `Urban100` datasets.
* And place them to `/data/DIV2K`, `/data/Set5`, `/data/Set14`, and `/data/B100`, `/data/Urban100`. The prefix '/data' can be changed at `code/datasets/super_resolution/__init__.py:45`.
# Training the student.

* `teacherx4_div2k_69068.ckpt` is a teacher model pretrained on DIV2K.
* Modify the `path_to_teacher` in `code/frameworks/distillation/start_jobs.py` to the path where above file is.
* Run `python frameworks/distillation/start_jobs.py` at the folder `code` to start the training. This step train a student on DIV2K dataset with a scale factor of 4.
