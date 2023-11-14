# Feature Extraction and classification of EEG data based on Motor Imagery using Meta heuristic optimisation techniques and novel Deep learning methods
![Model Overview](img1.001.jpeg)

## BCI_EEG_ML-classification-models
 ConvNETs based ML models for feature extracted 64 channel EEG data


# EEG Motor Imagery Preprocessing Pipeline

## 1. Bandpass FIR Filtering
- **Objective:** Retain EEG signal components within µ (8-12 Hz) and β (16-30 Hz) bands.
- **Method:** Use a Finite Impulse Response (FIR) bandpass filter with cut-off frequencies at 7 Hz and 32 Hz.

## 2. Epoching
- **Process:** Segment EEG data into epochs centered around specific events.
- **Epoch Details:** Starts 0.5 seconds before the event onset and ends 2 seconds post-event.

## 3. Baseline Correction and Normalization
- **Correction:** Perform baseline correction by subtracting the mean value of the pre-event segment.
- **Normalization:** Obtain normalized signal \(E_{\text{norm}} = \frac{E - \mu_{\text{baseline}}}{\sigma_{\text{baseline}}}\).

## 4. Resampling
- **Purpose:** Reduce computational load and focus on the frequency range of interest.
- **Method:** Downsample from 500 Hz to 250 Hz by selecting every alternate sample.

## 5. Common Spatial Pattern (CSP) Filtering
- **Application:** Enhance class separability for multiclass tasks in EEG data.
- **Configuration:** Extract 2 filters per class for a 6-class classification, resulting in 12 filters.

## 6. Continuous Wavelet Transform
- **Transform:** Process CSP-transformed epochs using the Continuous Wavelet Transform (CWT) with a Complex Morlet wavelet.

## 7. Formation of RGB Images
- **Creation:** Concatenate sets of three features from CWT output to form an RGB image.
- **Size and Interpolation:** Reshape and interpolate data to fit the size (3, 224, 224) using bicubic interpolation.

## 8. Model Architecture and Training: MobileNet
- **Architecture:** Utilize MobileNet for its efficient computation suitable for mobile and embedded vision applications.
- **Features:** Depthwise and Pointwise Convolution, Batch Normalization, ReLU, Global Average Pooling.
- **Classification Layer:** Customize the last layer to have six output units for the six classes in the grasp-and-lift task.

## 9. Training Process
- **Fine-tuning:** Newly added linear layer with the rest of the model layers frozen.
- **Optimization:** Use Cross-Entropy Loss and the Adam optimizer.
- **Evaluation:** Monitor model’s accuracy in both training and evaluation phases.
- **Cross-Validation:** Include a 5-fold CV test-train split for generalizability and robustness.

## 10. Quantization of MobileNet Model to 16-bit Integers
- **Objective:** Reduce model’s memory footprint while maintaining accuracy.
- **Steps:** Scaling, Quantization, and Clipping to 16-bit integer range.
- **Deployment:** Enables deployment on resource-constrained devices for MI decoding tasks.

## Dependencies Required
* Python 3.7
* Tensorflow 2.1.0
* MATLAB 9.10.0
* SciKit-learn 0.22.1
* SciPy 1.4.1
* Numpy 1.18.1
