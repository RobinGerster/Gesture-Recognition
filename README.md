# Gesture Recognition

## Generating the Data

Before starting the data generation process, please ensure that you have placed the original data from [this Google Drive link](https://drive.google.com/drive/folders/13KHZpweTE1vRGAMF7wqMDE35kDw40Uym) into the `train` and `test` folders located in the `orig_data` directory.

To generate the data for gesture recognition, we follow these steps:

1. **Extract Frames**: We extract frames from the video files. Each frame is resized to 224x244 pixels and converted to a numpy array with RGB color channels. We save an array of frames for each video file in the `frame_data` directory.

2. **Create Embeddings**: We use the `dinov2_vits14` model to create embeddings for the frames. Before generating the embeddings, we normalize the color channels. For each frame, we generate an embedding of size 384. The embeddings are stored as numpy arrays in the `embedding_data` directory, with each frame's embeddings saved in a corresponding `frame.npz` file.

By following these steps, we prepare the data necessary for gesture recognition.
