# Generating a Diverse Rashomon Set of Good Sparse Generalized Additive Models

This project expands on the work done in the [FasterRisk paper](https://arxiv.org/abs/2303.16047) (published at NeurIPS'23) and is an extension of the code found in this [Github repository](https://github.com).
The Rashomon set is the set of all near-optimal models. 
This code constructs a Rashomon set of sparse GAMs by swapping out features of the optimal sparse solution. 
All generated models have a loss within the $\epsilon$ threshold from the optimal loss. 

## Background: Sparse GAMs

A GAM model has the form
$$g(E[y]) = \omega_0 + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p)$$
where $\omega_0$ is the intercept, $f_j$'s are the shape functions and $g$ is the link function, e.g., the identity function for regression or the logistic function for classification. Each shape function $f_j$ operates only on one feature $x_j$. 

In practice, a continuous feature is usually divided into bins, thereby its shape function can be viewed as a step function, i.e.,
$$f_j(x_j) = \sum_{k=0}^{B_j-1}\omega_{j,k}\cdot 1 [b_{j,k} < x_j \leq b_{j,k+1}]$$
where $\{b_{j,k}\}_{k=0}^{B_j}$ are the bin edges of feature $j$, leading to  $B_j$ total bins. This is a linear function on the binned dataset whose features are one-hot vectors.

**We consider the classification problem, where each shape function is a step function.** The sparsity is defined by the number of steps in the shape function. In our setting, we consider both $\ell_0$ penalty on the number of steps and $\ell_2$ penalty on the weights of steps. 

## Installation

Clone this repository, and then create a conda environment.
```bash
conda create -n FasterRisk python=3.9 # create a virtual environment
conda activate FasterRisk # activate the virtual environment
python -m pip install fasterrisk # pip install the fasterrisk package
```

## How to use this code to find the Rashomon set

This code can be used to find the Rashomon set when a sparse generalized additive model is provided. A sparse model should be a JSON object and has the following structure:
```
{
    "data_file": data_name, 
    "X": X, 
    "header_new": header,
    "p": X.shape[1]
    "sample_proportion": sample_proportion, 
    "lamb0": lambda0,
    "lamb2": lambda2, 
    "w_orig": w_orig, 
    "hessian": H,  
    "log_loss_orig": log_loss_orig,
    "multiplier": multiplier, 
    "rset_bound": eps
}
```

An example JSON object is available in ``diabetes_0.001_0.001_1.01.p``. 

**data_file**
 - Type: string 
 - Description: the name of the dataset. 

**X**
 - Type: (n, p) numpy array
 - Description: a matrix with binary values (0 or 1). The first column consists of all ones, used for intercept. For other columns, 1 indicates the sample falls into the respective bin. 
 - Example:
    ```
    1, 1, 0, 1, 0
    1, 0, 1, 0, 0
    1, 0, 0, 0, 0
    ```
    ``X[0, 1] = 1`` indicates that the first sample has ``bloodpressure <= 0``. 

**header_new** 
 - Type: a list of length p
 - Description: column names for the matrix X. 
 - Example: 
 ```['intercept', 'BloodPressure<=0.0', '0.0<BloodPressure<=61.0', '61.0<BloodPressure<=66.0', '66.0<BloodPressure<=86.0']```

**p**
 - Type: integer
 - Description: the number of columns in matrix X. 

**sample_proportion**
 - Type: numpy array of length p
 - Description: proportion of samples in each bin. 
 - Example:
```array([1, 0.33, 0.33, 0.33, 0])```

**lamb0**
 - Type: float
 - Description: regularizer for $\ell_0$ penalty. A larger value of ``lamb0`` penalizes more on the number of steps in the shape function, leading to a sparser model.  

**lamb2**
 - Type: float
 - Description: regularizer for $\ell_2$ penalty. A larger value of ``lamb2`` penalizes more on the weights of steps in the shape function

**w_orig**
 - Type: numpy array of length p
 - Description: weights of steps in shape functions. It is used as an initialization for finding the Rashomon set. 

**hessian**
 - Type: (p, p) numpy array
 - Description: the Hessian matrix. It is used as an initialization for finding the Rashomon set. 

**log_loss_orig**
 - Type: float
 - Description: log loss of the sparse GAM. 


**multiplier**
 - Type: float with value > 1
 - Description: a multiplier used to define the upper bound of the loss for the Rashomon set. 

**rset_bound** 
 - Type: float
 - Description: the upper bound of the loss for the Rashomon set, i.e. ```log_loss_orig * multiplier```


Once the sparse GAM is available as a JSON object, we can directly run ``RsetOPT`` to get its Rashomon set. Suppose we store the json object in a pickle file named ``diabetes_0.001_0.001_1.01.p``, we can run the following code to get the Rashomon set. 
```
import numpy as np
import pandas as pd
import pickle
from src.rset_opt import *

filepath = "diabetes_0.001_0.001_1.01.p"

model = RSetOPT(filepath)
model.finetune_ellipsoid()
model.get_precision()
H_opt = model.get_normalized_H()
w_opt = model.w_orig

# store optimized hessian and w to the file
model.update_file(H_opt, w_opt)
```


If the sparse GAM is not available, we can first use sparse GAM algorithms to get an initial model. In our experiments, we use [FastSparse](https://arxiv.org/abs/2202.11389)  to generate the sparse GAM and then postprocess the obtained model into the JSON object above. Please see the [prepare_gam.py](https://github.com/chudizhong/GAMsRashomonSet/blob/main/src/prepare_gam.py) for more details. Information about how to install and run FastSparse can be found in its repo (https://github.com/jiachangliu/fastSparse). Other sparse GAM algorithms where shape functions are step functions can be used to generate the initial model. 




## Applications of the Rashomon set

The GAM Rashomon set can be used for diverse applications. We show examples for 5 different applications in the [example_rset_application.ipynb](https://github.com/chudizhong/GAMsRashomonSet/blob/main/example_rset_application.ipynb). 

- Application 1: get different GAM models from the Rashomon set. 
- Application 2: get variable importance range. 
- Application 3: find the model that satisfies the monotonic constraints. 
- Application 4: find the users' preferred shape functions. 
- Application 5: explore sudden changes or abnormal patterns in the shape function. 

