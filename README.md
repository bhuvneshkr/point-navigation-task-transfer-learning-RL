# point-navigation-task-transfer-learning-RL

In this project, we analyze the possibilities that having any prior visual knowledge like ability to detect edges and corners
will facilitate agent for learning vision-based navigation at a faster speed and for generalized environment. We utilized the 
concept of transfer learning to incorporate the visual prior knowledge into the model and we adapted it to perform a point 
navigation task in the AI habitat environment. Our experimental results show that the agent with prior visual knowledge is able
to learn faster than the agent without. Our claim is also strongly supported by Lin Yen- Chen[1] et.al where they initially trained
the model on passive vision tasks before adapting to perform a manipulation task like pushing-gripping-task for objects. 
We used the RL (PPO) agent architecture discussed in Manolis Savva[2] et.al as the base model on which we performed visual transfer learning.

[1] Learning to See before Learning to Act - Visual Pre-training for Manipulation Lin Yen-Chen, Andy Zeng, Shuran Song, Phillip Isola, Tsung-Yi Lin, ICRA 2020.
[2] Habitat: A Platform for Embodied AI Research - Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao , Erik Wijmans, Bhavana Jain, Julian Straub, 
Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, Dhruv Batra
