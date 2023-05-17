# Gesture Recognition

## Generating the Data

Before starting the data generation process, please ensure that you have placed the original data from [this Google Drive link](https://drive.google.com/drive/folders/13KHZpweTE1vRGAMF7wqMDE35kDw40Uym) into the `train` and `test` folders located in the `orig_data` directory.

To generate the data for gesture recognition simply execute the `embeddings.py`. The code follows these steps:

1. **Extract Frames**: We extract frames from the video files. Each frame is resized to 224x224 pixels and converted into a PyTorch tensor with RGB color channels. The frames are normalized based on the ImageNet mean and standard deviation. 

2. **Create Embeddings**: We use the `dinov2_vits14` model to create embeddings for the frames. The frames are sent to the device (GPU if available, otherwise CPU), and the model generates an embedding of size `384` for each frame. The embeddings are stored as PyTorch tensors in the embedding_data directory, with each video's embeddings saved in a corresponding images_embeddings.pt file. Each of the files has tensor of shape `(num_frames_in_video, 384)`.

By following these steps, we prepare the data necessary for gesture recognition.
