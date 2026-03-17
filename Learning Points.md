### Week 2
Convolution
Filter
Flatten
Error signal (gradient of loss) -> weight, bias
Weight update -> gradient descent

### Week 3
Padding - add 0s around images
- Helps maintain image details (eg edges of images)
Strides - how many pixels do you move the filter across an image
- Down sampling images
- 1-2 should be enough for our data
Max pooling - after getting data into linear layer (ReLU)
- Similar to filter (2x2) and takes the highest value of pixel, eliminates rest -> down sampling
Average pooling
- Average value of 2x2 pixels
Backpropagation
- Softmax error signal (probability vector)
- Updating linear layers
	- Weight: take the old weights, subtract the learning rate and then multiply by error signal (also multiplied by the transpose of the weights)
	- Bias: old bias minus the error signal
- Convolutional layer update
	- Update filter using the weight function but instead the transpose of the input activation we would be using the incoming error gradient. 