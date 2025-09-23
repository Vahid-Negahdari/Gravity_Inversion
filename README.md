# Gravity Inversion : Physics-Guided Data-Driven Methods 

## :art: About the project
In this project, we introduce data-driven approaches to address the Gravity Inversion problem. Our work explores three main strategies: a purely data-driven approach (First Method), a hybrid approach that combines generative models with the underlying physics of the problem (Second Method), and iterative schemes designed to refine the initial guess (Third Method). The full paper for this source code can be found on [[1]](#1).\
The dataset needed for this project [[2]](#2) is automatically uploaded within the codes.
## :key: Getting Started
Please follow the guidelines we've provided to use the code effectively.
### Requirements
Please see the 
[requirements.txt](https://github.com/Vahid-Negahdari/Inverse_Elastic_Scattering/blob/main/requirements.txt) 
documentation for library and hardware requirements.
### Installing
1. Clone the repository to your local machine:
``` 
git clone https://github.com/Vahid-Negahdari/Elastic-Full-Waveforrm-Inversion.git
```

2. Change directory into the cloned repository:
``` 
cd Elastic-Full-Waveforrm-Inversion
```
### Executing program

1.To execute the First Method, which is **Direct Deep Learning Inversion**:
``` 
python3 First_Method_Direct_DL.py
```
2.To apply techniques within the Second Method, initially run:
``` 
python3 Create_Dataset.py
python3 Displacement_Approximation.py
```    
* To utilize the **Least Square** technique, execute:
  ``` 
  python3 Second_Method_Least_Square.py
  ```
* To utilize the **Linear-to-Nonlinear** technique, execute:
  ``` 
  python3 RhoU_Approximation.py
  python3 Second_Method_Linear_to_Nonlinear.py
  ```  
* To utilize the **Inverse Convolution** technique, execute:
  ``` 
  This code will be completed soon
  ```  
3.To utilize the Third Method, the **New-VAE** Method,
you first need to execute the linear-to-nonlinear process and then follow up with:
```
python3 Third_Method_New_VAE.py
```

## :books: References 
<a id="1">[1]</a> 
To be announced
. [arXiv](https://).\
<a id="2">[2]</a> 
To be announced, [doi:](https://)

## :relaxed: Author  
Vahid Negahdari, Shirin Bahrami

Email:  <vahid_negahdari@outlook.com>

Any discussions and contribution are welcomed!
