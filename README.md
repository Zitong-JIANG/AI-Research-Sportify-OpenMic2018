# Music Instrument Recognition with CNN

## Introduction
This project focuses on identifying instruments in polyphonic music recordings using Convolutional Neural Networks (CNN). It's a fundamental challenge in the field of Music Information Retrieval with significant practical applications in areas like music recommendation, production, education, document archiving, and copyright detection.

## Data Source
The model is trained on the OpenMIC-2018 dataset, a collection of audio and crowd-sourced instrument labels. This dataset, a collaboration between Spotify and New York University, contains 20,000 Creative Commons-licensed music excerpts from the Free Music Archive, each partially labeled for 20 instrument classes.

### Source Reference
Humphrey, Eric J., et al. "OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition." ISMIR, 2018. [Dataset GitHub](https://github.com/cosmir/openmic-2018/blob/master/examples/modeling-baseline.ipynb)

## Model
Our CNN model is tailored to solve a multi-label classification problem, recognizing instruments in polyphonic recordings. CNNs are chosen for their proficiency in processing data with spatial hierarchies.

### Experiment and Methodology
- **Main Task**: Multi-label classification of musical instruments.
- **Outcome**: The CNN model efficiently recognizes instruments, slightly outperforming the random forest classifier in accuracy for some instruments.

### Performance
- Test Accuracy: 72.19%
- Test Precision (weighted average across classes): 93%
  - Saxophone: 93% vs 83%
  - Trumpet: 92% vs 78%
  - Ukulele: 96% vs 77%
- Test Recall (weighted average across classes): 100%

## Future Work
- **Improve CNN Model**: Enriching the training dataset, increasing model complexity.
- **Analysis**: Investigating the gap between training loss and validation loss.
- **Optimization**: Enhancing the selection and validation of optimal thresholds.
- **Other Directions**: Exploring real-time recognition, cross-cultural studies, and multimodal learning.

### Additional References
- Gavrikov, Paul. "VisualKeras," GitHub, 2020. [VisualKeras GitHub](https://github.com/paulgavrikov/visualkeras)
- Hongkun Yu, et al. "TensorFlow Model Garden," GitHub, 2020. [TensorFlow Models GitHub](https://github.com/tensorflow/models)
- Humphrey, Eric J., et al. "OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition," Spotify R&D Research, September 2018. [Spotify Research](https://research.atspotify.com/publications/openmic-2018-an-open-dataset-for-multiple-instrument-recognition/)
