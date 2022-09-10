## WWParRecDL
[![Build Status](https://travis-ci.org/uwescience/wwparrecdl.svg?branch=master)](https://travis-ci.org/uwescience/wwparrecdl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

WWParRecDL is a library for generating samples of the Wong & Wang 2006 model, with the possibility of changing the default parameters, and recover them using Deep Learning. The code for sample generation is written in Cython to leverage the speed of C language. This repository has been created using the Shablona template https://github.com/uwescience/shablona.

### Organization of the  project

The project has the following structure:

    wwparrecdl/
      |- README.md
      |- wwparrecdl/
         |- __init__.py
	 |- cfast
	    |- parrec.pyx
	    |- setup.py
	    |- ...
         |- wwparrecdl.py
         |- due.py
         |- data/
            |- ...
         |- tests/
            |- ...
      |- doc/
         |- Makefile
         |- conf.py
         |- sphinxext/
            |- ...
         |- _static/
            |- ...
      |- scripts
         |- ...
      |- setup.py
      |- .travis.yml
      |- .mailmap
      |- appveyor.yml
      |- LICENSE
      |- Makefile
      |- ipynb/
         |- ...
      

The core of the project is the code inside of `wwparrecdl/wwparrecdl.py`. Here are the functions for running samples of the models and compute statistics on the outputs like accuracy and mean reaction times. There is also the code of the simple MLP to recover the parameters and two functions (in beta) for data loading and fine-tuning. Note that the sample generation here uses pure Python. The fast Cython implementation can be found in `wwparrecdl/cfast/parrec.pyx'. Please note that after editing, the .pyx file needs to be compiled using:

```
python setup.py build_ext --inplace
```

In the scripts folder there are some .py scripts to run the sampling algorithm, extract features from the output and training/tuning the model.

### Module code

We place the module code in a file called `wwparrecdl.py` in directory called
`wwparrecdl`. This structure is a bit confusing at first, but it is a simple way
to create a structure where when we type `import wwparrecdl as pr` in an
interactive Python session, the classes and functions defined inside of the
`wwparrecdl.py` file are available in the `pr` namespace. For this to work, we
need to also create a file in `__init__.py` which contains code that imports
everything in that file into the namespace of the project:

    from .wwparrecdl import *

### Project Data

You can create a `wwparrecdl/data` folder in which you can
organize the data for example the output of the sampling algorithm using different parameters.

### Citation
If you find this repository useful and would like to use the code, please consider cite the work

```
@article{SZ22,
 title = {Deep Learning for parameter recovery from a neural mass model of perceptual decision-making},
 author = {Sicurella, E. and Zhang, J},
 journal = {2022 Conference on Cognitive Computational Neuroscience},
 year = {2022},
 doi = {10.32470/CCN.2022.1095-0}
}
```

