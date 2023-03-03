# com.mcast.res.poseDetection
Research into pose detection

# Setting up
- [GIT](https://git-scm.com/): Git libraries for collaboration.
- [GitHub Desktop (Optional but recommended)](https://desktop.github.com/): a git client with UI if you prefer.
- [Python](https://www.python.org/): either through package manager (Windows Store, Apple App Store, Ubuntu Repository) or directly from site.
- [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html): this is optional if using CPU processing but convenient. Mandatory if using GPU processing.
- [VSCode (Optional but recommended)](https://code.visualstudio.com/): recommended IDE for development.

Post installation of application use the scripts found under the setup folder to create a conda environment and to install the recommended VSCode extensions. You may want to consider setting up Run Configurations in VSCode accordingly.

# Setting up CUDA, TensorFlow/Keras with GPU support (NVIDIA-CUDA)
Adapted from Jeff Heaton's guide on [YouTube](https://www.youtube.com/watch?v=OEFKlRSd8Ic)/[GitHub](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/manual_setup2.ipynb)

- **Step 01** - Install [Nvidia Video Driver](https://www.nvidia.com/download/index.aspx)

**P.S.** Restart PC after this step.

- **Step 02** - Install [Visual C++ via Microsoft Visual Studio Community](https://visualstudio.microsoft.com/vs/community/)

**P.S.** After downloading Visual Studio go to [Worloads](https://learn.microsoft.com/en-us/visualstudio/install/modify-visual-studio?view=vs-2022) and select Desktop development with C++

- **Step 03** - Install Python 3.10 from [homepage](https://www.python.org/) or via Windows Store
- **Step 04** - Install [Cuda](https://developer.nvidia.com/cuda-downloads)
- **Step 05** - Install [Anaconda](https://anaconda.org/)/[MiniConda](https://docs.conda.io/en/latest/miniconda.html), setup conda environment, Jupyter, create Jupter Kernel, [Tensorflow/Keras](https://www.tensorflow.org/install/pip)

Launch Anaconda Prompt and **run as administrator**. Run the following code.

```
conda create --name rdi python=3.10
conda activate rdi
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow<2.11"
conda install -c conda-forge numpy
conda install -c conda-forge jupyter
ipython kernel install --name "rdi-kernel" --user
conda install -c conda-forge matplotlib
conda install -c conda-forge pandas
conda install -c conda-forge scikit-learn
conda clean --all
conda env export > rdi.yaml
conda deactivate
```

- **Step 06** - Test if tensorflow version is correct and if Cuda is using GPU

Launch Anaconda Prompt and **run as administrator**. Run the following code.<br/>

```
conda activate rdi
python -c "import tensorflow as tf; print(tf.__version__)"
```

**P.S.** Should show 2.10.1.<br />

`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`<br />

**P.S.** Should give you a list of supported graphic cards.

- **Step 07** - Testing tensorflow and verifying GPU usage

Open a command prompt and type the following code:

`nvidia-smi`

This should show a table with the list of supported graphic cards. Note the device number given such as GPU 0 and the GPU-Util value. Run the `tensorflow_get_started.ipynb` notebook using the tf conda environment. This notebook processes the mnist dataset. During model fitting run the `nvidia-smi` command again and note the GPU-Util. This is a more accuracte representation than the task manager performance visual. For more information check the TensorFlow [Get Started](https://www.tensorflow.org/tensorboard/get_started) page.

# Useful links
- [Anaconda repository](https://anaconda.org/anaconda): to determine correct command for installation of package via conda repository.
- [scikit-learn](https://scikit-learn.org/stable/install.html): for further research on machine learning related modules.
- [Cuda, tensorflow, keras setup - video](https://www.youtube.com/watch?v=OEFKlRSd8Ic): Non Anaconda guide, low level installation of Cuda, tensorflow, keras for machine learning with GPU 
- [Cuda, tensorflow, keras setup - GIT](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/manual_setup2.ipynb)

# Yoga Pose Classifier Reference content
- [Yoga Pose Classifier Article](https://medium.com/@leahnagy/yoga-pose-classification-with-tensorflows-movenet-model-3e5771fda292)
- [Yoga Pose Classifier Git](https://github.com/leahnagy/yoga_pose_classifier)
- [Yoga Pose Dataset](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset)

The reference code has a number of errors which have been fixed in this repository. Also note that the dataset has a few errors:
- TRAIN/tree/00000114.jpg
- TEST/plank/00000084.jpg
