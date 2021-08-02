# VFM_ML

## Contributors
* Ridha Alkhabaz (ridhama2@illinois.edu/ ridha.alkhabaz@gmail.com)
* Weichang Li (Weichang.Li@aramcoamericas.com)
* Tao Lin (tao.lin@aramcoamericas.com)



## Project overview
We used Temporal Convolutional Networks (TCN), Prophet algorithm from Facebook and DeepAR from amazon to predict flowrates using pressure and temperature readings. Our data was from [Nikolai Andrianov's paper](https://arxiv.org/abs/1802.05698). 

### Relevant Publications
* https://doi.org/10.1016/j.ifacol.2018.06.376
* https://doi.org/10.1016/j.energy.2020.119708
* https://doi.org/10.1016/j.petrol.2019.106487


## File structure:

* **Data**: Our data was read and processed through *preprocessing.py*, from the *src* folder in order to apply machine learning algorithms. 


* **Analysis**
	* ML: 
		* Prophet-Slugging-Gas.ipynb: Here, we apply prophet in the slugging experiment to forecast gass mass flow.
		* Prophet-Slugging-Liquid.ipynb: Here, we apply prophet in the slugging experiment to forecast liquid mass flow.
		* TCN-Slugging-Gas.ipynb: Here, we apply TCN in the slugging experiment to forecast gass mass flow.
		* TCN-Slugging-Liquid.ipynb: Here, we apply TCN in the slugging experiment to forecast liquid mass flow.
		* DeepAR-Slugging-Gas.ipynb: Here, we apply DeepAR in the slugging experiment to forecast gass mass flow.
		* DeepAR-Slugging-Liquid.ipynb: Here, we apply DeepAR in the slugging experiment to forecast liquid mass flow.
		* Prophet-Well-GAS.ipynb: Here, we apply Prophet in the well's experiment to forecast the gas volume flow rate.
		* Prophet-Well-WATER.ipynb: Here, we apply Prophet in the well's experiment to forecast the water volume flow rate.
		* Prophet-Well-OIL.ipynb: Here, we apply Prophet in the well's experiment to forecast the oil volume flow rate.
		* TCN-Well-GAS.ipynb: Here, we apply TCN in the well's experiment to forecast the gas volume flow rate.
		* TCN-Well-WATER.ipynb: Here, we apply TCN in the well's experiment to forecast the water volume flow rate.
		* TCN-Well-OIL.ipynb: Here, we apply TCN in the well's experiment to forecast the oil volume flow rate.
		* DeepAR-Well-GAS.ipynb: Here, we apply DeepAR in the well's experiment to forecast the gas volume flow rate.
		* DeepAR-Well-WATER.ipynb: Here, we apply DeepAR in the well's experiment to forecast the water volume flow rate.
		* DeepAR-Well-OIL.ipynb: Here, we apply DeepAR in the well's experiment to forecast the oil volume flow rate.


## Results
* we show the behavior by plotting the forecasted data versus the original and provide statistical matrics, like Mean Squared Error.

