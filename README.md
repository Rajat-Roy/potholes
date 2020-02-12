## **Pothole Detection, Masking and Area Percentage Calculation**

The following code adapts from the [Mask_RCNN](https://arxiv.org/abs/1703.06870) implementation from the Facebook AI Research Team (FAIR).

*    Main source repository: https://github.com/matterport/Mask_RCNN

#### Run inference on images
Prepare a python 3.7 environment.

Open a terminal and execute the following lines:
```
git clone https://github.com/Rajat-Roy/potholes.git
cd potholes
pip install -r requirements.txt
wget https://github.com/Rajat-Roy/potholes/releases/download/v1.0/mask_rcnn_car_damage_0030.h5
wget https://github.com/Rajat-Roy/potholes/releases/download/v1.0/dataset.zip
```
Extract the dataset.zip archive

If jupyter notebook is not installed already:
```
pip install jupyter
```
Start the notebook server:
```
jupyter notebook
```
In the jupyter app directory listing click and run the `potholes_demo.ipynb` file.

Change the file paths in the notebook accordingly.

Execute the code cells.
