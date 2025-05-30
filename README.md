Multi-Layer Perceptron (stochastic gradient descent) solving the MNIST dataset implemented in Java. Single Hidden Layer of size 100 with momentum value of 0.5 and Epoch limited at 50 generated the best results: 97.92% Test Accuracy, 99.62% Training Accuracy. 

**Results with differing Hidden Units (Momentum set at 0)**

|  | Test Accuracy After 50 Epochs | Training Accuracy After 50 Epochs |
| ---         |      :---:      |       :---:  |
| **20 Hidden Units**  | 94.36%     | 95.68%    |
| **50 Hidden Units**     | 96.6%       | 98.51%      |
| **100 Hidden Units**     | 97.31%       | 99.22%      |

**Results with differing Momentum Value (Hidden Units set at 100)**

|  | Test Accuracy After 50 Epochs | Training Accuracy After 50 Epochs |
| ---         |      :---:      |       :---:  |
| **0 Momentum**  | 97.68%     | 99.58%    |
| **0.25 Momentum**     | 97.71%       | 99.6%      |
| **0.5 Momentum**     | 97.92%       | 99.62%      |
