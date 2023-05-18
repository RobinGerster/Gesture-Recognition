# Gesture Recognition

## Generating the Data

Before starting the data generation process, please ensure that you have placed the original data from [this Google Drive link](https://drive.google.com/drive/folders/13KHZpweTE1vRGAMF7wqMDE35kDw40Uym) into the `train` and `test` folders located in the `orig_data` directory.

To generate the data for gesture recognition, execute the `embeddings.py` script. The code follows these steps:

1. **Extract Frames**: Frames are extracted from the video files. Each frame is resized to 224x224 pixels and converted into a PyTorch tensor with RGB color channels. The frames are normalized based on the ImageNet mean and standard deviation.

2. **Create Embeddings**: The `dinov2_vits14` model is used to create embeddings for the frames or the difference between consecutive frames. The frames are sent to the device (GPU if available, otherwise CPU), and the model generates an embedding of size `384` for each frame. 

   - **Frame Embeddings**: For frame embeddings, each frame's tensor is directly fed to the model.
   - **Frame Differences Embeddings**: For frame differences embeddings, the difference between consecutive frames is calculated and the resulting tensor is fed to the model. For the first frame, a blank frame (tensor of zeros) is subtracted.

3. **Store Embeddings**: The embeddings are stored as PyTorch tensors in the `frame_embeddings` or `diff_frame_embeddings` directory, depending on the type of embeddings created. Each video's embeddings are saved in a corresponding `images_embeddings.pt` file. Each file contains a tensor of shape `(num_frames_in_video, 384)`.

By following these steps, we prepare the data necessary for gesture recognition, either using direct frame embeddings or embeddings of frame differences.
