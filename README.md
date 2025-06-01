# FairRL: Fairness-Constrained Reinforcement Learning for Loan Approval Systems

## Overview
FairRL is a reinforcement learning framework designed to reduce algorithmic bias in automated decision-making systems such as loan approvals. By integrating fairness constraints into the learning process, FairRL aims to optimize both utility (e.g., reward, accuracy) and fairness (demographic parity and equal opportunity).

## Features
- Fairness-aware Double DQN agent for binary loan decisions  
- Integrated Gradients for post-hoc bias attribution  
- Fairness constraints: Demographic Parity & Equal Opportunity  
- Prioritized Experience Replay emphasizing fairness violations  
- Curriculum Learning to phase-in fairness during training

## Dataset
**UCI Adult Dataset**  
- Sensitive Attribute: `Sex`  
- Target Variable: `Income >50K`  
- Preprocessing: One-hot encoding, normalization, train/test split with balanced demographics

## Results
| Metric                | Baseline | FairRL  | Improvement |
|-----------------------|----------|---------|-------------|
| Demographic Parity    | 0.197    | 0.009   | 95% ↓       |
| Equal Opportunity     | 0.015    | 0.002   | 87% ↓       |
| Accuracy              | 0.939    | 0.742   | 21% ↓       |
| Reward                | 0.877    | 0.484   | 45% ↓       |

## Architecture
- **Agent**: Double DQN  
- **Replay Buffer**: Prioritized by fairness violations  
- **Constraints**: Applied via penalties in reward function  
- **Curriculum**: Gradual increase in constraint weights over training phases

## How to Run
1. Clone the repository and install requirements  
2. Prepare and preprocess the UCI Adult Dataset  
3. Train the baseline and FairRL models using `train.py`  
4. Evaluate results with fairness and utility metrics in `evaluate.py`

## Future Work
- Expand to multi-class sensitive attributes  
- Apply to other domains like hiring or criminal justice  
- Enhance interpretability for stakeholders  

## Contributors
Thomas Carr, Farsheed Haque, David Caballero, Subhash Krishna Veer Buddhi, Chinedu Ibeanu  
University of North Carolina at Charlotte

## License
MIT License
