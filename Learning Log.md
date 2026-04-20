### Week 1
**What have I done?**
For this week my and my friends got together to do our first iteration of teaching each other the basic things needed for us to start building our neural network model. I decided to stick to the algorithmic and logic side of neural networks while my friend focused on the mathematical and data processing aspect. 

**What have I learnt?**
From the mathematical and data aspect I have learnt about the image pre processing where we would flatten the image to process the image into values for the model to interpret; normalisation and convolution; activation gates such as ReLU to help the model pick up on strong signals; and finally sigmoid/softmax output gates to help classify binary/multi-class classification.

From the algorithmic and logic aspect I have taught about the general 3 stage layer architecture of neural network models; how initial models do not have strong training and requires going through epochs to help improve; activation gates; and the process of forward propagation.

**Any changes to goals?**
By the end of week 1 we decided it is best that we both had slightly overlapping areas that we studied so we decided to dive deeper into whichever areas we were studying to ensure that we teach each other the in-depth aspects of all sides instead of surface level knowledge. We also decided that it would be best to start building our neural network model prototypes based on what we have learnt to maintain a consistent development of our model.

**Next steps?**
The next steps would be to research deeper into the areas that I am responsible for to gain deeper understanding of neural network models as I have underestimated previously. I should also start working each week to building a small prototype to merge with my partner's model.

### Week 2
**What have I done?**
Having the responsibility of the algorithmic and logical side of explaining neural networks, I was tasked with talking about logits being converted to probability. Additionally, I talked about the loss function and different procedures we can take to ensure that the model does not crash by dividing by 0 or finding the log of 0. Finally, I talked about how computers prefer a continuous loss gradient curve to learn to optimise its outputs while for humans it would be better to check stepped accuracy to see the performance of the model.

**What have I learnt?**
For this week, my friend decided to dive a little deeper into explaining about the inner workings of a softmax function and its implementation in python logic. He then mentioned cross entropy loss and gradient descent. Gradient of loss can be used to determine the weight and bias of each signal of the neural network to help improve the model. The cross entropy loss is a way for the model to evaluate how well it has performed.

**Any changes to goals?**
After this week's discussion I realized that no matter how well we try to split the work in the category of mathematics and data and algorithmic and logical aspect of the neural network, we will always have overlapping parts. So we decided that it would be best to just list all things that we need to learn to build our own convolutional neural network and divide the work from there instead of dividing it into categories. This way we can ensure that none of our works are overlapping.

**Next steps?**
For the following week, I will be focusing on researching data splitting for optimal performance, mini-batching and epoch loops. My friend will be looking into pooling, padding and strides for our data and he will also be looking into the fundamentals of backpropagations.

### Week 3
**What have I done?**
For this week's discussion, I have decided to tell my friend about data splitting, mini batching and epoch loops. Data splitting is important because we want to be able to train the model and be able to test that model to a set of testing data to test its validity in real world usage. Mini batching is similar to batching except it sequentially processes data to help improve its RAM efficiency when training and to update parameters more frequently compared to batching where it processes the entire dataset in one go. Epoch loops are necessary for the model to learn images by going through entire datasets multiple times instead of just once.

**What have I learnt?**
My friend has focused on teaching me the importance of a convolutional layer for image classification. Within a convolutional layer architecture, there involves a window that slides across an image and it performs particular mathematical calculations to help understand the image more efficiently. Padding adds 0s around images to help maintain image details and strides determines how many pixels to move the window across an image, essentially down sampling the images for efficient training. Pooling involves max pooling and average pooling and it occurs after getting the data into the ReLU. In a 2x2 window, max pooling would only take the highest value of pixel and eliminate the rest of the pixels to down sample an image. On the other hand, average pooling would take the average of all pixels in that window. Back propagation utilises the gradient descent and the softmax error signal of the model to update its network weights and bias. Finally, the convolutional layer updates the window using the weight function but instead the transpose of the input activation, we would be using the incoming error gradient.

**Any changes to goals?**
After today's discussion we found that this method of teaching other different topics instead of the mathematical and algorithmic aspects of a particular topic is much more efficient and less confusing to explain as there were no overlapping areas we talked about. So from now on we'll continue using this method of teaching to help us cover more topics quickly and focus on producing the final product.

**Next steps?**
For next week we'll be focusing on the evaluation of the results the model returns. I'll be focusing on the learning curve of the model, the difference between training and validation accuracy, and finally the GradCAM. My friend will focus on discussing about the confusion matrix and f1-score to investigate dataset imbalances that could be worsening the model performance.
### Week 4
**What have I done?**
For this week I focused on discussing on the model accuracy evaluation and talking about GradCAM. Training accuracy of the model is talking about how accurately it is able to classify images that it has already trained on. Validation accuracy shows how accurately the model has been able to evaluate images that it has not trained on to assess its generalisability. The learning curve is showing these two accuracy outputs across the number of epochs. The GradCAM is then used to ensure that the model is training on the important details of an image to ensure that the model is not learning from the background of its data.

**What have I learnt?**
My friend discussed the confusion matrix and f1-score to evaluate dataset imbalance. The confusion matrix is used to output the correctness of multi-class classification on a table. The f1-score uses the precision and recall metrics that are gathered from a binary classification confusion matrix. The precision focuses on the how many of the model's positive predictions were actually correct and the recall focuses on how many of the positive predictions were correct based on all positive instances. The f1-score tells us how well the model has performed relative its dataset. To use the f1-score for multi-class classification we would have to perform the f1-score metric for each of the 7 classes as it only works on binary classification. 

**Any changes to goals?**
No changes of the goals were discussed this week as we were on a really good pace to meeting our scheduled deadlines.

**Next steps?**
Next week we plan to discuss the final part of our neural network from scratch. We will discuss what methods of optimisation can be implemented into our dataset and the model now that we have insight to the evaluation of the model performance. I will be focusing on the algorithmic optimisation of the model while my friend focuses on the dataset optimisation and dropout layers.

### Week 5
**What have I done?**
This week I focused on teaching about ADAM (Adaptive Moment Estimation) optimisation. ADAM is a combination of two moments, the Momentum and RMSProp (Root Mean Squared Propagation). Momentum is an optimisation algorithm that uses inertia in the search direction to pass local minima. This is done through the usage of its previous gradient to help oscillate and converge. RMSProp normalises the gradient descent of the model through the usage of moving average of squared gradients which works really well for mini-batch learning. It decreases the learning rate for larger gradients to avoid learning too quick and increases learning rate for smaller gradients to avoid vanishing. ADAM takes Momentum and RMSProp as first and second moment. It takes the two formulas and influences their biases through the implementation of decay rate. This enables it to adapt its learning rate across the gradient descent to optimise quicker.

**What have I learnt?**
Continuing on from last week's topic my friend continued to focus on dataset optimisation and regularisation of the model. He discussed about implementing data augmentation to enlarge the dataset to be more balanced and implementing dropout layers to prevent the model from memorising the data from particular neurons. Data augmentation augments the dataset to produce multiple iterations of the images. For example, it can flip, rotate, zoom, shift, mirror, and adjust the brightness of images to produce more data for particular classes if there is a major data imbalance. Dropout prevents the model from overfitting by temporarily removing some of its neurons randomly and increase generalisability.

**Any changes to goals?**
Originally the goal was to develop a neural network from scratch but as our goal was to build a multi-class classification model we have decided to steer our goal towards developing a convolutional neural network from scratch. The architecture of the model would require quite a bit of refactoring as fully connected layers will need to be replaced with convolutional layers and further methods discussed on earlier weeks.

**Next steps?**
As our final discussion meeting has been completed, we are now going to purely focus on building prototypes and iterations of the model based on everything we have learnt. We plan to continue working separately and then merging our models together every week and then have our final discussion after 2 weeks to focus on producing a robust model. We also plan to update each other if we have discovered anything new that we could implement into our model for enhanced performance.

### Week 6
**What have I done?**
This week I focused on making several changes to my model. I replaced the ADAM optimiser with AdamW which applies weight decay directly to updating the parameters unlike ADAM. This leads to more consistent regularisation and improve generalisability. I also significantly increased the dataset from 1000 to around 5000 images per class, bringing the total to around 35000 images in an attempt to break the plateau of validation accuracy. Additionally I also implemented batch normalisation to help reduce the amount of noise generated from mini-batch. As well as, instead of just using dropout layers, I replaced the dropout layers in between convolutional layers with spatial dropout o help reduce overfitting as that may have been plateauing my model. Finally, instead of increasing my epoch, I used an early stopper to make my model stop whenever the accuracy and F1 score shows no improvement in 25 epochs. This way, I can make my model train up to 500 epochs if there are signs of improvements.

**What have I learnt?**
The most significant thing I learnt this week was that data quality and size matter as much as model architecture. No matter how much I refined the architecture, the validation accuracy kept plateauing at around 60% when training on only 1,000 samples per class over 200 epochs. Hence, the increase of my sample to 5000 per class. I also speculate that the model was learning the wrong things in the image due to the image quality as observed from GradCAM. I also discovered that vectorising the mathematical calculations rather than relying on standard arrays significantly boosted the speed of training, which became especially noticeable when working with the larger dataset.

**Any changes to goals?**
After observing the plateau behaviour throughout this process, it would be good to investigate further into what may be causing the model's performance to stop increasing. This would be good to research to further understand model behaviour.

**Next steps?**
I will continue investigating what is causing my model validation accuracy to plateau at 70%. Despite the difficulty in improving the model, I believe that the primary goal of this collaboration project has been achieved, where we wanted to learn how to build a neural network model together. Next week, I will meet with my friend to finalise what extra things we have learnt to help improve the model to produce a final model.


