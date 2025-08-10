# Fish Record

## Project

### Applying Recurrent Neural Network (RNN) Variants in the Prediction Free-swimming Zebrafish Movement in 3D Space

This project focuses on applying 2D head-embedded virtual reality (VR) setups to 3D VR. To do this, I start with free-swimming fish: synchronized 2D videos are triangulated while correcting for air-water refraction, and per-frame pose is extracted and a sequence of poses is used to predict future trajectories ~0.2 sec in advance.

## Description
This repo includes Python codes and utilities I wrote for recording, syncing, and triangulating videos for 3D tracking of animals. Specifically, I use two Teledyne Blackfly S (BfS) USB3 cameras.

- [Configuring Synchronized Capture with Multiple Cameras](https://www.teledynevisionsolutions.com/support/support-center/application-note/iis/configuring-synchronized-capture-with-multiple-cameras/)
    - Teledyne documentation for setting up hardware synced capture, as well as supported models.
- [Acquiring Synchronized Multiple Camera Images with Spinnaker Python API + Hardware Trigger](https://justinblaber.org/acquiring-stereo-images-with-spinnaker-api-hardware-trigger/) (Cred. Justin Blaber)
    - Useful document on how to set up hardware and simple software implementation
- [Spinnaker SDK](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/)
    - How to set up Spinnaker SDK (need to download .whl file)

## Features

### A ton of versions
Will be removed in the future (hopefully)

### [capture_calibration_pairs_v3.py](capture_calibration_pairs_v3.py)
Initialize GUI for viewing synced feed, and press return to capture a pair of chessboard images. Get a chessboard ready.

### [calibration_toolbox_v2.py](calibration_toolbox_v2.py)
- Get intrinsic camera matrix + distortion coefficients for each camera provided calibration chessboard images.
- Get stereo geometry parameters for a stereo pair provided synced pairs of chessboard images.
- Get projection matrices (from 3D to 2D camera view), necessary for OpenCV triangulation.

### [find_origin.py](find_origin.py)
Enter GUI for selecting individual pixels on each image pair. Necessary for pre ray-tracing calibration. Also triangulates points for convenience.

### [ray_tracing_projection.py](ray_tracing_projection.py)
Correct points for air-water interface refraction and triangulate into a 3D point. My initial testing shows it is much more accurate than pure triangulation.
- Adapted from Murase et al., [2008](http://dx.doi.org/10.14358/PERS.74.9.1129) and Cao et al., [2020](https://doi.org/10.1016/j.jag.2020.102108)


### [capture_two_async_v9.py](capture_two_async_v9.py) (fallback to [capture_two_async_v5.py](capture_two_async_v5.py))
Record from both cameras. Drops a few frames sometimes (I don't know why), but handles it on the camera, video writer, and queue level. Very clunky codec functionality at the moment: you'll probably want a GPU. Fallback to [capture_two_async_v5.py](capture_two_async_v5.py), v9 was coded after I discovered I had Copilot Pro.
#### Helper files
- [capture_handler_v3.py](capture_handler_v3.py)
    - Wraps PySpin's *ImageEventHandler*. Writes to *SharedMemory* buffer on image. Functions for reading buffer.
- [capture_driver_v2.py](capture_driver_v2.py)
    - Wraps PySpin library. Functions for setting up as well as releasing sync resources. Edit class variables (serial number, exposure time, etc.) depending on setup.
- [select_dialog.py](select_dialog.py)
    - Utilities to get directory, file, or files using QDialog.


---
![pixelated zebrafish with thumbs up logo][fish]

[fish]: fishy_thumbsup.png "Logo Title Text 2"