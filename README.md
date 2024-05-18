# Language Modeling Tasks

## Overview

This repository contains the scripts and data for running the various components of this language modeling assignment. The `src` directory contains all the model definition/creation/training scripts, and the `data` dir contains all of the training/testing data, training metrics, and results.

## Usage

Run the different parts of this assignment (make sure you are in the project root directory) as follows:

`python src/main.py part1` (speech classification task)

`python src/main.py part2` (language modeling task)

`python src/main.py part3` (architectural exploration for speech classifier)

`python src/main.py tune_part3` (hyperparam tuning on speech classifier)

## Code

* `src/transformer.py` contains the various model layers
* `src/part1.py` contains the code run to generate the results of part 1
* `src/part2.py` contains the code run to generate the results of part 2
* `src/part3.py` contains the code run to generate the results of part 3
* `src/main.py` runs the preceding modules


