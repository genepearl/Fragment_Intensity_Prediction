# Fragment Intensity Prediction

### CNN(B) Architecture 


![my_model](https://user-images.githubusercontent.com/124587762/217084872-7f2341ed-9818-4233-afd5-27c24fd39bbf.jpg)

This model takes two inputs: a 7-dimensional vector representing precursor charge and a 30x22 matrix representing peptide sequence, both of which were one-hot encoded24. The peptide sequence input is processed through two 1D convolutional layers in combination with max pooling layers. The output of the convolutional layers is then flattened. The precursor charge input is passed through a max pooling layer and, after that, through a dense layer with ReLU activation and is concatenated with the output produced by peptide sequence input. The concatenated layers are then passed through another dense layer with ReLU activation before finally being passed through a dense layer with 56 units and sigmoid activation to produce the output (Figure 7). The model was compiled using the Model class from the Keras library. It was trained with spectral angle as a loss function, the Adam optimizer from Keras library25, and a batch size of 128 over 10 epochs. 
