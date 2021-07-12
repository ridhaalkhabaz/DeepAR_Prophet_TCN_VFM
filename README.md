# VFM_ML

## Contributors
* Ridha Alkhabaz (ridhama2@illinois.edu/ridha.alkhabaz@gmail.com)
* Weichang Li (Weichang.Li@aramcoamericas.com)
* Tao Lin (tao.lin@aramcoamericas.com)



## Project overview
We used Temporal Convolutional Networks (TCN), Prophet algorithm from Facebook and Long Short-Term Memory (LSTM) to predict flowrates using pressure and temperature readings. Our data was from [Nikolai Andrianov's paper](https://arxiv.org/abs/1802.05698). 

### Relevant Publications
* https://doi.org/10.1016/j.ifacol.2018.06.376
* https://doi.org/10.1016/j.energy.2020.119708
* https://doi.org/10.1016/j.petrol.2019.106487


## File structure:

* **Data**: Our data was read and processed through *preprocessing.py*, from the *src* folder in order to apply machine learning algorithms. 


* **Analysis**
	* ML: 
		* Prophet_VFM.ipynb: Here, we apply prophet for the the original data set.
		* STANDARD_TCN_VFM.ipynb: Here, we apply TCN for the the original data set.
		* DEEPAR_VFM.ipynb: Here, we apply DEEPAR for the the original data set.


## Results
* TCN and DEEPAR are very capable to predict flowrates. 
