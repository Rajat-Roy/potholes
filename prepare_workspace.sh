#!/bin/bash
rm -r ./* && git clone https://github.com/Rajat-Roy/potholes.git
mv /content/potholes/* ./ && rm -r potholes
wget https://github.com/Rajat-Roy/potholes/releases/download/v1.0/mask_rcnn_pothole_0030.h5
wget https://github.com/Rajat-Roy/potholes/releases/download/v1.0/pothole_images.zip
unzip pothole_images.zip
