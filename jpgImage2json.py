"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/vision/image_classification/resnet#pretrained-models

Typical usage example:

    resnet_client.py
"""

from __future__ import print_function

import base64
import io
import os
import json

import numpy as np
from PIL import Image
import requests

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

# Current Resnet model in TF Model Garden (as of 7/2021) does not accept JPEG
# as input
MODEL_ACCEPT_JPG = False

def main():
  # Download the image
  dl_request = requests.get(IMAGE_URL, stream=True)
  dl_request.raise_for_status()

  if MODEL_ACCEPT_JPG:
    # Compose a JSON Predict request (send JPEG image in base64).
    jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
  else:
    # Compose a JOSN Predict request (send the image tensor).
    jpeg_rgb = Image.open(io.BytesIO(dl_request.content))
    # Normalize and batchify the image
    jpeg_rgb = np.expand_dims(np.array(jpeg_rgb) / 255.0, 0).tolist()
    request = {'instances': jpeg_rgb}
    output_file = os.path.splitext('catImg')[0] + '.json'
    with open(output_file, 'w') as out:
        json.dump(request, out)

if __name__ == '__main__':
  main()