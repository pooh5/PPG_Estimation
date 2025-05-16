This project uses a ResNet-based deep learning model to estimate Tissue Perfusion Pressure (TPP) from photoplethysmography (PPG) signals. It's designed for biomedical signal researchers and engineers working in critical care, especially where noninvasive hemodynamic monitoring is important.


What’s Included in the Project

Loads PPG signals and associated features like mean arterial pressure (MAP), diastolic pressure (DBP), and heart rate (HR)

Trains a custom ResNet model to predict TPP from these signals

Evaluates predictions using metrics like RMSE, MAE, median absolute deviation (MeAD), and R²

Generates figures such as training loss curves and prediction vs ground-truth histograms

Folder and File Structure

training_header.py: Includes utility functions for loading data and defining the training loop

MyFirstResNet: The neural network model tailored for 1D time-series PPG inputs

TPP_PPG_dataset: A dataset class that prepares waveform data from HDF5 files

runs/: Stores the trained model and its training trajectory

figures_resnet/: Contains saved visualizations

Using this code, would involve access to the dataset that is not public. The code however serves as blueprint to train on one dimension time-series PPG to predict Tissue Perfusion Pressure



Installation Requirements

You’ll need the following tools and packages:
Python 3.8 or higher, PyTorch, NumPy, Matplotlib, scikit-learn
Custom utility files: tk_utils, training_header, etc.
To install dependencies: pip install numpy matplotlib scikit-learn torch

After running the training:A trained model saved as runs/M1.5.pth, A 2D histogram showing predicted vs actual TPP values, A line plot showing loss and learning rate over training epochs
Model Evaluation Criteria, RMSE – How far off predictions are on average, MAE – Average absolute error across samples, MeAD – Median of absolute deviations, R² Score – How well the predictions explain the variance

Notes Input PPG segments are 60 seconds long, sampled at 60 Hz, Model inference runs on GPU if CUDA is available (check DEVICE), You can customize the ResNet architecture via get_model_v3()

Author

Pooja Shukla, PhD currently a reseach fellow at Mass General Hospital and Harvard Medical School. My work focuses on Biomedical engineer focused on machine learning, signal processing, and critical care applications.

Feel free to use this as a base for your own experiments. I will not be able to share the dataset.  Contributions and feedback are welcome!