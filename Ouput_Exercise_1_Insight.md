D:\Work\CSAI_DiemCong\exercise_21_9.py
======================================================================
EXERCISE 21.9: REINFORCE vs PEGASUS COMPARISON
======================================================================
Overall progress: 100%|████████████████████████████████| 3/3 [13:47<00:00, 275.96s/it]

======================================================================
SUMMARY
======================================================================

REINFORCE Final Return: -1.3729 ± 0.0728
PEGASUS Final Return:   -1.4253 ± 0.1174

Key Observations:
- REINFORCE has high variance due to Monte Carlo sampling
- PEGASUS has lower variance due to correlated sampling (fixed seeds)
- PEGASUS converges faster in terms of iterations
- Both converge to near-optimal policy for the 4x3 world

======================================================================
FINAL TRAINED POLICY (PEGASUS)
======================================================================
Training final PEGASUS agent...
Final training:   0%|▏                              | 1/200 [00:52<2:54:31, 52.62s/it]