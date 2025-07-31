# Fish Record

## Project

### Applying Recurrent Neural Network (RNN) Variants in the Prediction Free-swimming Zebrafish Movement in 3D Space

This project focuses on applying 2D head-embedded virtual reality (VR) setups to 3D VR. To do this, I start with free-swimming fish: synchronized 2D videos are triangulated, and per-frame pose is extracted and a sequence of poses is used to predict future trajectories ~0.2 sec in advance.

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

### [calibration_toolbox.py](calibration_toolbox.py)
Generate intrinsic camera matrix + distortion coefficients for cameras provided calibration images of chessboards.

### [capture_two_async_v9.py](capture_two_async_v9.py) (fallback to [capture_two_async_v5.py](capture_two_async_v5.py))
Record from both cameras. Drops a few frames sometimes (I don't know why), but handles it on the camera, video writer, and queue level. Very clunky codec functionality at the moment: you'll probably want a GPU. Fallback to [capture_two_async_v5.py](capture_two_async_v5.py), v9 was coded after I discovered I had Copilot Pro.
#### Helper files
- [capture_handler_v3.py](capture_handler_v3.py)
    - Wraps PySpin's *ImageEventHandler*. Writes to *SharedMemory* buffer on image. Functions for reading buffer.
- [capture_driver_v2.py](capture_driver_v2.py)
    - Wraps PySpin library. Functions for setting up as well as releasing sync resources. Edit class variables (serial number, exposure time, etc.) depending on setup.


---
![pixelated zebrafish with thumbs up logo][fish]

[fish]: fishy_thumbsup.png "Logo Title Text 2"