# TSGPS

# Instruction
This repository contains the original code described in the paper "Pan-Infection Foundation Framework Enables Multiple Pathogen Prediction".

<!--
# Citation
If you use these models in your research, please cite:
-->

The required environment is in requirements.txt. You can use following code to install it.
```
pip install -r requirements.txt
```

You can use data_process_PAGE.py to perform preliminary data processing.
```
python data_process_PAGE.py
```

You can run simple_train.py to train Pan-Infection Foundation Model (PIFM). Then run Student.py to use TSGPS.

```
python simple_train.py
```
```
python Student.py
```
The model will be saved in the result folder
