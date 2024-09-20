
MAMA MIA Experiments
(September 9, 2024)
--------------------


The experiments designed in this codebase run in two stages.

1. Shadow Modelling

During this stage, all of the focal-points of the specified SDG
algorithm will be chosen. By default, it simulates all three implemented;
MST, PrivBayes, and Private-GSD. It will save off results in a dictionary into a
pickle file.

This code can be ran using the file and parameters:

    mamamia_experiments.py shadowmodel {experiment} {SDG algorithm} {parameter}

where the experiment can be one of {"A", "B", "C", "D" (i.e. additional results)},
described in the paper, the SDG algorithm can be one of {"mst", "priv", "gsd"},
corresponding to the SDG algorithms described in the paper,
and the parameter can be one of {0.1, 0.32, 1, 3.16, 10, 31.62, 100, 316.23, 1000},
denoting values of epsilon if the experiment is "A", "C", or "D", or the parameter
can be one of {100, 316, 1000, 3162, 10000, 31623}, denoting training dataset sizes
if the experiment is "B".

You will need to specify a directory to save and read intermediate results as 'DIR' in
mamamia_experiments.py and another as 'DATA_DIR' in encode_data.py.

Running these experiments will also require installing all the appropriate
libraries, and msy require running Python 3.12.



2. Conduct Attacks

Here, the focal-points have already been chosen during the first stage. Now
these experiments will use those focal-point counts to use in an attack.

Similar to running step 1, this step is ran by running the file and parameters

    mamamia_experiments.py attack {experiment} {SDG algorithm} {parameter} {overlap} {set MI}

by using the same possible values for "experiment", "algorithm", "parameter" given in step 1,
and by setting overlap (i.e. whether or not to use training data that overlaps with the
auxiliary data) to 'True' or 'False', and set MI to 'True' or 'False'.

The code generates random samples from the auxiliary data to use as the training
data, passes this into the synthetic data generators, which will then create
synthetic data. From there, the focal points are aggregated into a density estimation of
the synthetic data and the population data in order to conduct MIA on a similarly
sampled set of holdout targets. A subset of these targets were used in the training data.

The code then scores the predictions on the targets, using AUC, and stored in a results file
to the specified directory from above.

This process is repeated over n_runs, specified at the top of util.py, among
other experimental configurations.


Datasets:

These experiments are created to operate over two datasets. One is the California
Housing Dataset, which is included in the sklearn library, and will load into memory
simply if sklearn is installed and connected to the internet.

The other dataset, 'base.parquet', and 'meta.json', are the SNAKE data files described
in the technical paper accompanying this submission. Both will be included in the
zip file containing this code. They must be organized into a file structure
for this code to recognize.


Reprosyn, Private-GSD and Relaxed-Adaptive-Projection modules:

Our experiments use default implementations of MST, PrivBayes from the Reprosyn Library
(https://github.com/alan-turing-institute/reprosyn), an implementation of Private-GSD
from the authors' repository (https://github.com/giusevtr/private_gsd) and an
implementation RAP from the authors' repository
(https://github.com/amazon-science/relaxed-adaptive-projection). However, in order
to measure the behaviors needed for shadow modelling, in the firststep, we
need to infiltrate the code, not to modify behavior, but to record internal
behavior. To be clear, we are not learning internal behavior when training on the 'hidden'
training data in our experiments. That would violate our threat model. But rather, we
assume the ability to observe ~recreated~ internal behavior on random samples of the
auxiliary data, to learn what ~might~ have happened when operating over the hidden data.

So to this end, we copy these three repositories into this code base, and they are in
separate folders at the root directory of this zip file.

Lastly, godspeed to you. It is regretful that we cannot be at your side to aid you in
running this code as issues arise. But the utmost care was taken in keeping things neat,
modularized, readable, and reproducible. However, this code base was developed by a team
of few. And surely, habitual behavior eventually became required to execute this code to
its fullest potential. Thank you for reading.

