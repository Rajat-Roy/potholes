"""
Mask R-CNN
Train on the dataset and implement detection.
------------------------------------------------------------
"""

import os
import cv2
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class Config(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tissue"  

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class Dataset(utils.Dataset):
    
    class_name = "tissue"
    
    def set_ClassName(self, _class_name):
        self.class_name = _class_name;

    def load_dataset(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class(self.class_name, 1, self.class_name)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations["_via_img_metadata"].values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                self.class_name,
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a correct dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != self.class_name:
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == self.class_name:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Train
############################################################
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = Dataset()
    dataset_train.load_dataset(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Dataset()
    dataset_val.load_dataset(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

    
    
############################################################
#  Inference
############################################################
def Inference(weights_path, image_path):
    class InferenceConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    # config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)


    # Load weights
    model.load_weights(weights_path, by_name=True)

    # Run model detection and postprocess
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    
    # Post Process
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    # Copy color pixels from the original color image where mask is set
    mask = r['masks']
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        #=========================
        
        # create a CLAHE object (Arguments are optional).
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


        # convert the YUV image back to RGB format

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

        img_yuv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        hsv = cv2.cvtColor(img_yuv, cv2.COLOR_BGR2HSV)
        blur = cv2.medianBlur(hsv,51)
        mask_dry = cv2.inRange(blur,(0, 0, 0), (255, 255, 255))
        overlay = image.copy()
        outimg = image.copy()
        # overlay[mask_dry==0] = (200, 80, 200)
        # overlay[mask_dry==0] = (10, 180, 200)
        overlay[mask_dry>0] = (10, 180, 200)
        
        # apply the overlay
        cv2.addWeighted(overlay, 0.5, img_yuv, 0.5, 0.8, outimg)
            
            
        #=========================
        
        result = np.where(mask, outimg, img_yuv).astype(np.uint8)
        
        
        potholes = np.sum(mask)
        potholes = potholes*100/(np.prod(mask.shape)/3)
        
        
        print("\nTotal image area: {0}x{1} pixels\n".format(mask.shape[0], mask.shape[1] ))
        print("pothole area: {0:.2f}% (blue)\n".format(potholes))
        
    else:
        result = image.astype(np.uint8)
        
    return result

def Inference_Multi(weights_path, images_path, verbose=1, batch_size=1):
    class InferenceConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = batch_size

    config = InferenceConfig()
    # config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)


    # Load weights
    model.load_weights(weights_path, by_name=True)

    # Run model detection and postprocess
    # Read image
    # image = skimage.io.imread(image_path)
#     load_pattern = os.path.join(images_path, "*.jpg")
#     images = skimage.io.imread_collection(load_pattern)

    csv = open(os.path.join(ROOT_DIR, "output.csv"),"w")
    csv.write("name,result,total,pothole\n")
    image_paths=[]
    for filename in os.listdir(images_path):
        if filename!="via_region_data.json":
            image_paths.append(filename)
    
    batches = len(image_paths)//config.IMAGES_PER_GPU
    for batch in range(batches):
        
        # Detect objects
        start = batch*config.IMAGES_PER_GPU
        end = start+config.IMAGES_PER_GPU
        images=[]
        for image_path in image_paths[start:end]:
            images.append(skimage.io.imread(os.path.join(images_path, image_path)))
        
        rs = model.detect(images, verbose)

        # Post Process
        """Apply color splash effect.
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]
        """
        # Make a grayscale copy of the image. The grayscale copy still
        # has 3 RGB channels, though.
        # Copy color pixels from the original color image where mask is set

        results = []
        for i, r in enumerate(rs):
            mask = r['masks']
            image = images[i]
            if mask.shape[-1] > 0:
                # We're treating all instances as one, so collapse the mask into one layer
                mask = (np.sum(mask, -1, keepdims=True) >= 1)
                #=========================

                # create a CLAHE object (Arguments are optional).
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


                # convert the YUV image back to RGB format

                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))

                # equalize the histogram of the Y channel
                img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

                img_yuv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                hsv = cv2.cvtColor(img_yuv, cv2.COLOR_BGR2HSV)
                blur = cv2.medianBlur(hsv,51)
                mask_dry = cv2.inRange(blur,(0, 0, 0), (255, 255, 255))
                overlay = image.copy()
                outimg = image.copy()
                # overlay[mask_dry==0] = (200, 80, 200)
                # overlay[mask_dry==0] = (10, 180, 200)
                overlay[mask_dry>0] = (10, 180, 200)

                # apply the overlay
                cv2.addWeighted(overlay, 0.5, img_yuv, 0.5, 0.8, outimg)


                #=========================             

                potholes = np.sum(mask)
                potholes = potholes*100/(np.prod(mask.shape)/3)

                results.append({"image": np.where(mask, outimg, img_yuv).astype(np.uint8),
                                "name": image_paths[i+start],
                                "total": np.prod(mask.shape)/3,
                                "potholes": potholes
                               })

            else:
                results.append({"image": image.astype(np.uint8),
                                "name": image_paths[i+start],
                                "total": np.prod(image.shape)//3,
                                "potholes": 0.0
                               })
        for i, result in enumerate(results):
            file_name = "result_{0}".format(result["name"])
            skimage.io.imsave("./outputs/"+file_name, result["image"])          
            csv.write(result["name"] + ',' + 
                      file_name + ',' + str(result["total"]) + ',' + str(result["potholes"]) + "\n")
    csv.close() 
