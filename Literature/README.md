# Literature to use

### [Temporal Convolution for Real-time Keyword Spotting on Mobile Devices](https://github.com/ivaste/KeyWordSpotting/blob/master/Literature/To%20Use/Temporal_Convolution_for_Real_time_Keyword_Spotting_on_Mobile_Devices.pdf)
 - Abstract
 - Our dataset
 - 96.6% Acc, 305K params, 5.7ms
 - 96.1% Acc, 66K params, 1.1ms
 - è un miglioramento del paper DEEP RESIDUAL LEARNING FOR SMALL-FOOTPRINT KEYWORD...
 - Metrics: Accuracy, #All Params, time(ms) on Google pixel 1
 - INTRO
 - We should not consider Multiplication as metrics because is an indirect alternative for the direct metric such as latency.
 - Temporal Convolutional neural network. Empirically demostrated to reduce computation and increased performance in term of accuracy compared to 2Dconvs in KWS.
 - There has been an increase in the use of 1D convolutions in acustics and speech domain.
 - 1D conv along temporal axis of time-frequency rappresentations instead of convolving along th frequency axis or processing raw audio signals.
 - Github code
 - Data augmentation and preprocessing (background noise)
 - Training parameters
 - Include FLOPs computed by tensorflow profiling tool
 - ROC curves for models with corresponding values of AUC

### [A neural attention model for speech command recognition](https://github.com/ivaste/KeyWordSpotting/blob/master/Literature/To%20Use/A%20neural%20attention%20model%20for%20speech%20command%20recognition.pdf)
 - python library kapre for mel-scale
 - Different tasks: 20 commands, 12 commands, 35 words, left-right
 - Training parameters
 - Accuracy 94%, 202K parameters
 - Attention plots (log scale) to visualize what parts of the audio were most relevant
 - Confusion matrices
 - Try augmenting audio samples with background noise
 - Try using pretrained models
 - Try stack pair of words form more complex commands and use sequence to sequence model or multiple attention layers
 
### [DEEP RESIDUAL LEARNING FOR SMALL-FOOTPRINT KEYWORD SPOTTING](https://github.com/ivaste/KeyWordSpotting/blob/master/Literature/To%20Use/DEEP%20RESIDUAL%20LEARNING%20FOR%20SMALL-FOOTPRINT%20KEYWORD%20SPOTTING%20.pdf)
 - Intro
 - trade-off accuracy small net
 - MFCC
 - Accuracy ROC AUC
 - Training Parameters
 
### [Small-Footprint Keyword Spotting on Raw Audio Data with Sinc-Convolutions](https://github.com/ivaste/KeyWordSpotting/blob/master/Literature/To%20Use/Small-Footprint%20Keyword%20Spotting%20on%20Raw%20Audio%20Data%20with%20Sinc-Convolutions.pdf)
 - works directly on raw audio signals. NO MFCC.
 - 97.4% Accuracy, 62k parameters. SincConv+DSConv
 - 97.3% Accuracy, 122k parameters. SincConv+GDSConv
 - Sinc Convolution
 - Depth-wise separable conv nets
 - 3.1 per intro
 - not use residual connections in our network architecture, considering the memory overhead and added difficulty for hardware acceleration modules.

### [Speech Commands - A Dataset for Limited-Vocabulary Speech](https://github.com/ivaste/KeyWordSpotting/blob/master/Literature/To%20Use/Speech%20Commands-%20A%20Dataset%20for%20Limited-Vocabulary%20Speech.pdf)
 - Describes the dataset
 - 2 INTRO
 - 4 Motivation
 - Dataset description 5.1, 5.2, 5.8
 - 6
 - 7.1, 7.2

### [Efficient keyword spotting using dilated convolutions and gating](https://github.com/ivaste/KeyWordSpotting/blob/master/Literature/To%20Use/Efficient_keyword_spotting_using_dilated_convolutions_and_gating.pdf)
 - Gated activation units (works well with audio signals)
 - Residual learning startegies like skip connections. Used to speedup convergence and adress the issue of vanishing gradients.
 - Dilateted convolutions

### [Attention-Based Models for Speech Recognition](https://github.com/ivaste/KeyWordSpotting/blob/master/Literature/To%20Use/Attention-Based%20Models%20for%20Speech%20Recognition.pdf)
 - 40mel scale filterbank features togheter with energy in each frame, and first and second temporal differences, for a total of 123 feature per frame
 - Each feature rescaled to have zeromean and unit variance over tht training set
 - MAYBE other details on attention (Chapter 2)

### [Convolutional_Neural_Networks_for_Small_footprint_Keyword_Spotting](https://github.com/ivaste/KeyWordSpotting/blob/master/Literature/To%20Use/Convolutional_Neural_Networks_for_Small_footprint_Keyword_Spotting.pdf)
 - MAYBE performance measure
 - MAYBE limiting multiplications
 - MAYBE limiting parameters
 
### [Key-Word Spotting - The Base Technology for Speech Analytics](https://github.com/ivaste/KeyWordSpotting/blob/master/Literature/To%20Use/Key-Word%20Spotting%20-%20The%20Base%20Technology%20for%20Speech%20Analytics.pdf)
 - Chapter 5 on performance measurement
 
### [Hello Edge Keyword Spotting on Microcontrollers]()
 - Depth-wise separable CNN
 - 2.1 Explanation of Input->frames->MFCC features
 - Convolutional Recurrent Neural networks
 - Our Dataset
 









