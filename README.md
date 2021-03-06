# End-to-End Framework for Key-Word Spotting
Detect a set of predefined keywords in an audioclip.


<!-- ADD GIF demonstartion -->
![KeyWordSpotting](https://github.com/ivaste/KeyWordSpotting/blob/master/ReadmeImage.png)


[📈 Download Presentation PDF](https://github.com/ivaste/KeyWordSpotting/blob/master/Presentation/Presentation.pdf)

[📄 Download Paper PDF](https://github.com/ivaste/KeyWordSpotting/blob/master/Paper/Key%20Word%20Spotting.pdf)

[🕹️ Try Live Demo](https://colab.research.google.com/drive/15v66rkuL2hF0Ecg7gcD7RMVutVQCc0Nr)

[📺 Watch Presentation]()

[📺 Watch Live Demo]()


## Table of Contents
- [Overview](#overview)
- [Usage](#usage)
- [Contributing](#contributing)
- [Team](#team)
- [License](#license)

## Overview
The End-to-End framework for the KeyWordSpotting task is made of a sliding window of 1 second, a Voice Activity Detection module or a Silence Filter that select onfly the frames containig human voice, from those frames a feature extraction module will extract the Mel Spectogram or the Mel Cepstral Coefficients, this will be the input of the model. Finally a fusion rule aggregates all frames pedictions in a single one.

<p align="center">
<img src="https://github.com/ivaste/KeyWordSpotting/blob/master/Paper/End-To-End.png" width="500" />
</p>

## Usage
.......

### Download best model
If you just want to download and use the best model in your application you need to...


```python
import librosa
import numpy as np
import Models #Our models
import LoadAndPreprocessDataset
from tensorflow.keras.models import load_model
.....
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Team
| <a href="https://stefanoivancich.com" target="_blank">**Stefano Ivancich**</a> | <a href="https://github.com/TyllanDrake" target="_blank">**Luca Masiero**</a> |
| :---: |:---:|
| [![Stefano Ivancich](https://avatars1.githubusercontent.com/u/36710626?s=200&v=4)](https://stefanoivancich.com)    | [![Luca Masiero](https://avatars1.githubusercontent.com/u/48916928?s=200&v=4?s=200)](https://github.com/TyllanDrake) |
| <a href="https://github.com/ivaste" target="_blank">`github.com/ivaste`</a> | <a href="https://github.com/TyllanDrake" target="_blank">`github.com/TyllanDrake`</a> |


## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
