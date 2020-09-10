## Exploring Auxiliary Reasoning Tasks for Task-oriented Dialog Systems with Meta Cooperative Learning
Requirements: Code is written in Python 3 and requires Pytorch.

## Preparation
For quick start, please download the dataset.

## Code Explanation
The source/inputter implements the functions for data processing.
The source/model implements all functions of MCL, including basic model, and MCL algorithm.
The source/module implements all functions of basic module.
The source/utils implements all functions of evaluation functions, and other utils functions.

## Quick start
CUDA_VISIBLE_DEVICES=0 python main.py

## Test
./test_camrest.sh
