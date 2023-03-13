# Multi-Task-Audio-Classification
Project for Advanced Audio Processing Course at Tampere University. In this project, we explored the effectiveness of multi-task learning for audio classification tasks. Our model was designed using a hard parameter sharing architecture, sharing all hidden layers but keeping task-specific output layers separate. We compared our multi-task model with two individual models trained separately for gender and digit classification. Results showed that our proposed model comparably to the individual single task models, as shown in the table below.

## Results from Cross-Validation:

|              | Model               | Accuracy  | Precision | Recall
|--------------|---------------------|-----------|-----------|----------
| Single-Task  |Gender Classification |  97.847% &plusmn;1.485% | 0.987 &plusmn;0.014 | 0.986 &plusmn;0.016
|              | Digit Classification  |  98.671% &plusmn; 0.862% | 0.987 &plusmn;0.009 | 0.987 &plusmn;0.009
| Multi-Task   | Gender Classification  |  95.84% &plusmn; 2.898% | 0.978 &plusmn;0.025 | 0.97 &plusmn;0.025
|              | Digit Classification  |  96.766% &plusmn; 1.805% | 0.968 &plusmn;0.018 | 0.968 &plusmn;0.018


### Dataset used: https://github.com/soerenab/AudioMNIST
