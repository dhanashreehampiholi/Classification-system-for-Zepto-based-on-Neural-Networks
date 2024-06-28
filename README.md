# Classification-system-for-Zepto-Orders-based-on-Neural-Networks

## Executive Summary
Classification systems based on neural networks are a subset of artificial intelligence (AI) systems that are designed to categorize input data into different classes or categories. These systems are inspired by the structure and function of the human brain, where interconnected neurons work together to process information.
Neural network-based classification systems consist of layers of interconnected nodes (neurons) that process and transform input data to make predictions or classifications.

The most common type of neural network used for classification tasks is the feedforward neural network, which consists of an input layer, one or more hidden layers, and an output layer. The input layer receives the raw data, such as images, text, or numerical values, and passes it through the hidden layers, where the data is processed using weighted connections between neurons. Each neuron in the hidden layers applies a mathematical function to the input data and passes the result to the next layer.
The output layer produces the final classification or prediction based on the processed data. For example, in a binary classification task (e.g., spam detection), the output layer may have two neurons, each representing a class (spam or not spam), and the network predicts the class with the highest output value. Neural network-based classification systems are trained using labeled data, where the input data is paired with the correct classification. During training, the network adjusts the weights of the
connections between neurons to minimize the difference between the predicted and actual classifications. This process is repeated iteratively until the network achieves a desired level of accuracy.

Some common types of neural network-based classification systems include:
Convolutional Neural Networks (CNNs): CNNs are widely used for image classification tasks. Theyconsist of convolutional layers that extract features from input images, followed by pooling layers that reduce the dimensionality of the features, and fully connected layers that perform the final classification.

Recurrent Neural Networks (RNNs): RNNs are suitable for sequential data, such as text or time series data. They process input data one element at a time, maintaining an internal state (memory) that allows them to capture temporal dependencies in the data.

Deep Neural Networks (DNNs): DNNs are neural networks with multiple hidden layers. They are capable of learning complex patterns in the input data and are often used for tasks that require high levels of abstraction, such as natural language processing and speech recognition.

Neural network-based classification systems have been successfully applied to a wide range of tasks, including image recognition, speech recognition, natural language processing, and medical diagnosis. Their ability to learn from data and adapt to different types of inputs makes them a powerful tool for building intelligent systems.


## Introduction to Business

### a) Introduction to Zepto:

Zepto is a dynamic e-commerce platform founded in 2021. Owned and operated by two Stanford University drop-outs Aadit Palicha and Kaivalya Vohra, it has emerged as a leading
player in the online retail space. With a commitment to innovation and customer satisfaction, Zepto offers a diverse range of products and services tailored to meet the evolving needs of its global customer base. The company's ownership structure reflects a blend of strategic investors, venture capitalists, and individual stakeholders, all united by a shared vision of driving growth and excellence in the digital marketplace.

<img width="355" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/ebc024af-eb34-4c9f-abc4-b35b5e96daaf">

### b) Introduction to Products Sold and Pricing Range:

Zepto caters to a wide array of consumer needs, offering an extensive selection of products spanning categories such as electronics, fashion, home goods, beauty, and more. From affordable essentials to premium brands, Zepto ensures there's something for every budget and preference. With competitive pricing and frequent promotions, customers enjoy access to quality products at compelling prices. Whether shopping for everyday essentials or indulging in luxury items, Zepto delivers a seamless and
rewarding shopping experience.

<img width="159" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/09c48002-572d-4b87-8ff4-1b761955e26c">

### c) Introduction to Customer Profile Visiting the Business:

At Zepto, customer satisfaction is paramount, and the diverse clientele reflects this commitment. From tech-savvy millennials seeking the latest gadgets to discerning professionals in search of stylish apparel, Zepto caters to a broad demographic spectrum. Families, students, and working professionals alike find value in Zepto's offerings, drawn by the convenience, affordability, and quality synonymous with the brand. With a focus on delivering personalized service and exceeding customer expectations, Zepto continues to attract and serve a loyal and ever-expanding customer base.

## II. INTRODUCTION TO DATA COLLECTION AND DATA METHODOLOGY

### 1. Introduction to Neural Networks

Neural networks are a fundamental concept in machine learning and artificial intelligence, inspired by the structure and function of the human brain. They are composed of interconnected nodes, called neurons, which work together to process complex data inputs and produce meaningful outputs.

Here's a basic introduction to neural networks:
Neurons: Neurons are the basic building blocks of a neural network. Each neuron receives inputs, processes them using weights (which represent the strength of the connection between neurons), applies an activation function, and produces an output.

Layers: Neurons are organized into layers in a neural network. The three main types of layers are:
Input Layer: The input layer receives the initial data or features that the neural network will process.

Hidden Layers: Hidden layers are layers between the input and output layers. They perform complex transformations and computations on the input data.

Output Layer: The output layer produces the final output of the neural network based on the computations performed in the hidden layers.

Connections: Neurons in one layer are connected to neurons in the next layer through connections. Each connection has a weight associated with it, which determines the strength of the connection. These weights are adjusted during the training process to improve the network's performance.

Activation Functions: Activation functions introduce non-linearity into the neural network, allowing it to learn complex patterns in the data. Common activation functions include sigmoid, tanh, and ReLU (Rectified Linear Unit).

Training: Neural networks are trained using a process called backpropagation, which involves feeding the network with training data, comparing the output with the expected output (ground truth), and adjusting the weights to minimize the error. This process is repeated iteratively until the network's performance is satisfactory.

Applications: Neural networks are used in a wide range of applications, including image and speech recognition, natural language processing, autonomous vehicles, and more. They have shown remarkable success in solving complex problems that were previously considered challenging for traditional algorithms.

### 2. Introduction to relevance of Neural networks in the context of the project

Neural networks can be highly relevant for understanding whether a customer would buy and recommend products from Zepto to others. By analyzing various data points related to
customer interactions, preferences, and behaviors, neural networks can learn complex patterns and relationships that influence customers' likelihood to recommend a product.

For example, neural networks can be trained on data such as customer reviews, ratings, purchase history, demographics, and sentiment analysis from social media. By processing this data, the neural network can identify key factors that drive customer satisfaction and likelihood to recommend, such as product features, pricing, brand perception, and overall customer experience.

Once trained, the neural network can predict the likelihood of a customer buying and recommending Zepto products based on new data inputs. This information can be valuable to
Zepto company tailor their marketing strategies, improve product offerings, and enhance customer satisfaction, ultimately leading to increased brand loyalty and positive word-of-mouth recommendations.

### 3. Introduction to Data collected

The data was collected using a google form from various respondents belonging to Bangalore and Belgavi city. 

### 4. Introduction to Data Methodology

#### a. Data Preprocessing Steps followed
Data preprocessing is a crucial step in data analysis and machine learning. It involves cleaning and transforming raw data into a format that is more suitable for analysis. Here are the general steps followed in data preprocessing:
##### Data Cleaning:
Handling missing data: Remove or impute missing values using techniques like mean, median, or mode imputation.
Handling outliers: Identify and deal with outliers, either by removing them or transforming them.
Data deduplication: Remove duplicate entries if any.
Correcting data format: Ensure all data types are correct and consistent.
Handling irrelevant data: Remove irrelevant features that do not contribute to the analysis.

##### Data Transformation:
Normalization: Scale numerical features to a standard range, such as [0, 1] or [-1, 1].
Standardization: Transform data to have a mean of 0 and a standard deviation of 1.
Encoding categorical variables: Convert categorical variables into numerical representations (e.g., one-hot encoding).
Discretization: Convert continuous data into discrete categories.
Feature engineering: Create new features based on existing ones to improve model performance.

##### Data Reduction:
Feature selection: Select a subset of relevant features to reduce dimensionality.
Dimensionality reduction: Use techniques like Principal Component Analysis (PCA) or
Singular Value Decomposition (SVD) to reduce the number of features while retaining important information.

##### Data Integration:
Combine data from multiple sources into a single dataset, ensuring compatibility and consistency.

##### Data Discretization:
Convert continuous data into discrete form for analysis.

##### Data Normalization:
Scale the data to a specific range to ensure that all features contribute equally to the analysis.

##### Data Sampling:
If the dataset is imbalanced, perform sampling techniques such as oversampling orundersampling to balance the classes.
Data Splitting: Split the dataset into training, validation, and test sets for model training and evaluation.
Data Augmentation: Generate additional training data from existing data to improve model performance,often used in image or text data.

Each of these steps is essential for preparing the data for analysis and ensuring that the
machine learning model can learn effectively from the data.

#### b. Neural Networks
Neural networks are a class of machine learning algorithms inspired by the structure and function of the human brain. They are composed of interconnected nodes, called
neurons, organized into layers. Each neuron receives input, processes it using an activation function, and passes the result to the next layer. Neural networks are used to
model complex patterns in data and are capable of learning and adapting from examples.

There are several types of neural networks, including:
Feedforward Neural Networks (FNN): The simplest form of neural network, where information flows in one direction, from input to output. They are often used for tasks
like classification and regression.
Convolutional Neural Networks (CNN): Designed for processing grid-like data, such as images. CNNs use convolutional layers to extract features and pooling layers to
reduce dimensionality.
Recurrent Neural Networks (RNN): Particularly suited for sequence data, such as time series or natural language. RNNs have loops in their architecture, allowing
information to persist.
Long Short-Term Memory (LSTM): A type of RNN that can learn long-term dependencies. LSTMs have mechanisms to selectively remember or forget information
over time.
Autoencoders: Neural networks used for unsupervised learning tasks, such as dimensionality reduction or data denoising. They consist of an encoder that compresses
the input and a decoder that reconstructs the input from the compressed representation.

Neural networks have been applied to various fields, including computer vision, natural language processing, speech recognition, and robotics. They are known for their ability to learn complex patterns and generalize from examples, making them a powerful tool in machine learning.


### III. INTRODUCTION TO THE RESULTS OBTAINED AND INFERENCES DRAWN

#### 1. Data preprocessing results with codes

##### Incidence matrix output and interpretation

##### WITH ONE HIDDEN LAYER WITH 1 NODE

<img width="252" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/839fb6b3-a476-4863-8f7c-06ea533063f8">

<img width="407" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/d8fb50de-2377-442a-863b-aed82e7e4944">


##### Interpretation:
 Accuracy: Accuracy is the proportion of correct predictions (TP + TN) out of all predictions. In this case, the accuracy is 0.918, indicating that the model correctly
predicted 91.8% of the cases.

 Sensitivity (True Positive Rate): Sensitivity is the proportion of true positives (TP) out of all actual positives (TP + FN). In this case, the sensitivity for predicting "No" is 0.95, indicating that the model correctly predicted 95% of the actual "No" cases.

 Balanced Accuracy: Balanced accuracy is the average of sensitivity and specificity, giving an overall measure of the model's performance across both classes. In this case, the balanced accuracy is 0.9233, indicating an overall balanced performance of the model.


##### WITH ONE HIDDEN LAYER WITH 2 NODES

<img width="266" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/1ba9ae3f-1190-4321-a39c-82ac72df6d18">

<img width="322" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/3e0f139c-c23e-4d92-ba69-8e47131c5301">

##### Interpretation:
 Accuracy: Accuracy is the proportion of correct predictions (TP + TN) out of all predictions. In this case, the accuracy is 0.9388, indicating that the model correctly
predicted 93.88% of the cases.

 Sensitivity (True Positive Rate): Sensitivity is the proportion of true positives (TP) out of all actual positives (TP + FN). In this case, the sensitivity for predicting "No" is 0.9, indicating that the model correctly predicted 90% of the actual "No" cases.

 Balanced Accuracy: Balanced accuracy is the average of sensitivity and specificity, giving an overall measure of the model's performance across both classes. In this case, the balanced accuracy is 0.9328, indicating an overall balanced performance of the model.


##### WITH ONE HIDDEN LAYER AND 3 NODES

<img width="250" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/66629d07-60d3-407f-8b2a-5e5df939a001">

<img width="368" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/5df210e3-4ef0-4229-9bcc-d27c1f6ca2eb">

##### Interpretation:
 Accuracy: The model's overall accuracy is 91.84%, meaning it correctly predicted the outcome for 91.84% of the cases.

 Sensitivity: Also known as the true positive rate or recall, this measures the proportion of actual positives that are correctly identified by the model. In this
case, it's 95%.

 Balanced Accuracy: This is the average of sensitivity and specificity, giving an overall measure of the model's accuracy that takes into account class imbalance. Here, it's 92.33%.

##### TWO HIDDEN LAYERS AND TWO NODES

<img width="248" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/50adfebc-6c7f-47d4-a820-178afbd80fec">

<img width="462" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/6ac44e61-8293-49ef-8f7b-87e2c7eacad4">


##### Interpretation:
 Accuracy: The overall accuracy of the model is 95.92%, meaning it correctly predicted 95.92% of the cases.

 Sensitivity (True Positive Rate): Sensitivity measures the proportion of actual positives that are correctly identified by the model. Here, it's 95%.

 Balanced Accuracy: This is the average of sensitivity and specificity, providing an overall measure of the model's accuracy that accounts for class imbalance.
Here, it's 95.78%.

##### WITH THREE HIDDEN LAYERS AND 5 NODES

<img width="246" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/7f0fcbcf-6ef7-4841-8aed-37087b6202d0">

<img width="467" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/e6e593cd-1caa-4147-9693-856e9d63b0d0">


##### Interpretation:
 Accuracy: The model's accuracy is 91.84%, meaning it correctly predicted all cases.

 Sensitivity (True Positive Rate): Sensitivity measures the proportion of actual positives that are correctly identified by the model. Here, it's 95%

 Balanced Accuracy: This is the average of sensitivity and specificity, providing an overall measure of the model's accuracy that accounts for class imbalance.
Here, it's 92.33%.


##### WITH THREE HIDDEN LAYER AND TEN NODES

<img width="282" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/c4470aa0-396c-4de3-99ff-0533e9ff0fe1">

<img width="491" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/67c65556-e330-4c43-965e-3d59831cdbec">


##### Interpretation:
 Accuracy: The model's accuracy is 97.6%, meaning it correctly predicted all cases.

 Sensitivity (True Positive Rate): Sensitivity measures the proportion of actual positives that are correctly identified by the model. Here, it's 95%.

 Balanced Accuracy: This is the average of sensitivity and specificity, providing an overall measure of the model's accuracy that accounts for class imbalance. Here, it's 97.5%

##### WITH FIVE HIDDEN LAYER AND TEN NODES

<img width="229" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/2d0a24c1-a16e-4f8d-b52b-4d8832fe63c5">

<img width="414" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/dd9bf77f-798a-4b17-9db5-e7268ed66adf">

##### Interpretation:
 Accuracy: The model's accuracy is 100%, meaning it correctly predicted all cases.

 Sensitivity (True Positive Rate): Sensitivity measures the proportion of actual positives that are correctly identified by the model. Here, it's 100%.

 Balanced Accuracy: This is the average of sensitivity and specificity, providing an overall measure of the model's accuracy that accounts for class imbalance.
Here, it's 100%


### IV. CONCLUSION

Interpretation of the results obtained for marketing purposes

<img width="463" alt="image" src="https://github.com/dhanashreehampiholi/Classification-system-for-Zepto-based-on-Neural-Networks/assets/57892263/b513ffdb-6cad-4103-ba27-697774569e1d">

### V. CODE USED

#Problem-1 (NEURAL NETWORK WITH ONE HIDDEN LAYER AND ONE NODE)
install.packages('neuralnet')
install.packages('caret')
library(neuralnet)
library(caret)

#Create the dataframe with the provided data

df <- read.csv(file.choose(),header=TRUE)
View(df)

names(df)
names(df)<-
c('age','spend','Income','mode','recommendation','age_norm','spend_norm','Income_norm')

#Set seed for reproducibility
set.seed(1)

#Train the neural network model
nn <- neuralnet(recommendation ~ age_norm + spend_norm+Income_norm+mode, data = df,
linear.output = FALSE, hidden = c(1)) #ONE HIDDEN LAYER AND ONE NODE

#display weights
nn$weights

#display predictions
prediction(nn)

#plot network
plot(nn,rep='best')

#Make predictions
pred <- compute(nn, df[, c('age_norm','spend_norm','Income_norm','mode')])
pred

#Extract predicted class
pred_class <- ifelse(pred$net.result >= 0.5, 1, 0)

#Convert predicted class to factor with appropriate levels
pred_class <- factor(pred_class, levels = c(0, 1), labels = c('No', 'Yes'))

#Ensure that df$Acceptance is a factor
df$recommendation <- factor(df$recommendation , levels = c(0, 1), labels = c('No', 'Yes'))

#Create confusion matrix
confusionMatrix(pred_class, df$recommendation )

#####################################################################################
#with one hidden layer with two nodes
df <- read.csv(file.choose(),header=TRUE)
View(df)
names(df)
names(df)<-
c('age','spend','Income','mode','recommendation','age_norm','spend_norm','Income_norm')

#Set seed for reproducibility
set.seed(1)

#Train the neural network model
nn <- neuralnet(recommendation ~ age_norm + spend_norm+Income_norm+mode, data = df,

linear.output = FALSE, hidden = c(2)) #ONE HIDDEN LAYER AND ONE NODE
#ONE HIDDEN LAYER AND TWO NODE
#display weights
nn$weights
#display predictions
prediction(nn)
#plot network
plot(nn,rep='best')

#Make predictions
pred <- compute(nn, df[, c('age_norm','spend_norm','Income_norm','mode')])
pred

#Extract predicted class
pred_class <- ifelse(pred$net.result >= 0.5, 1, 0)
pred_class

#Convert predicted class to factor with appropriate levels
pred_class <- factor(pred_class, levels = c(0, 1), labels = c('No', 'Yes'))

#Ensure that df$Acceptance is a factor
df$recommendation
df$recommendation<- factor(df$recommendation, levels = c(0, 1), labels = c('No', 'Yes'))

#Create confusion matrix
confusionMatrix(pred_class, df$recommendation)

################################################################################
#problem-3 NN WITH ONE HIDDEN LAYER AND THREE NODES
df <- read.csv(file.choose(),header=TRUE)
View(df)
names(df)
names(df)<-
c('age','spend','Income','mode','recommendation','age_norm','spend_norm','Income_norm')

#Set seed for reproducibility
set.seed(1)

#Train the neural network model
nn <- neuralnet(recommendation ~ age_norm + spend_norm+Income_norm+mode, data = df,
linear.output = FALSE, hidden = c(3))#ONE HIDDEN LAYER AND THREE NODE
#display weights
nn$weights

NEURAL NETWORKS

21
8

#display predictions
prediction(nn)
#plot network
plot(nn,rep='best')

#Make predictions
pred <- compute(nn, df[, c('age_norm','spend_norm','Income_norm','mode')])
pred

#Extract predicted class
pred_class <- ifelse(pred$net.result >= 0.5, 1, 0)
pred_class

#Convert predicted class to factor with appropriate levels
pred_class <- factor(pred_class, levels = c(0, 1), labels = c('No', 'Yes'))
df$recommendation
df$recommendation<- factor(df$recommendation, levels = c(0, 1), labels = c('No', 'Yes'))

#Create confusion matrix
confusionMatrix(pred_class, df$recommendation)

################################################################
#problem-3 NN WITH TWO HIDDEN LAYER AND TWO NODES
df <- read.csv(file.choose(),header=TRUE)
View(df)
names(df)
names(df)<-
c('age','spend','Income','mode','recommendation','age_norm','spend_norm','Income_norm')

#Set seed for reproducibility
set.seed(1)

#Train the neural network model
nn <- neuralnet(recommendation ~ age_norm + spend_norm+Income_norm+mode, data = df,
linear.output = FALSE, hidden = c(2,2))#TWO HIDDEN LAYER AND TWO NODE

#display weights
nn$weights
#display predictions
prediction(nn)
#plot network
plot(nn,rep='best')

#Make predictions
NEURAL NETWORKS

21
9

pred <- compute(nn, df[, c('age_norm','spend_norm','Income_norm','mode')])
pred

#Extract predicted class
pred_class <- ifelse(pred$net.result >= 0.5, 1, 0)
pred_class

#Convert predicted class to factor with appropriate levels
pred_class <- factor(pred_class, levels = c(0, 1), labels = c('No', 'Yes'))

#Ensure that df$Acceptance is a factor
df$recommendation
df$recommendation<- factor(df$recommendation, levels = c(0, 1), labels = c('No', 'Yes'))

#Create confusion matrix
confusionMatrix(pred_class, df$recommendation)

################################################################
#problem-3 NN WITH THREE HIDDEN LAYER AND FOUR NODES
df <- read.csv(file.choose(),header=TRUE)
View(df)
names(df)
names(df)<-
c('age','spend','Income','mode','recommendation','age_norm','spend_norm','Income_norm')

#Set seed for reproducibility
set.seed(1)

#Train the neural network model
nn <- neuralnet(recommendation ~ age_norm + spend_norm+Income_norm+mode, data = df,
linear.output = FALSE, hidden = c(4,4,4)) #THREE HIDDEN LAYER AND TEN NODE EACH
#display weights
nn$weights
#display predictions
prediction(nn)
#plot network
plot(nn,rep='best')

#Make predictions
pred <- compute(nn, df[, c('age_norm','spend_norm','Income_norm','mode')])
pred

#Extract predicted class
pred_class <- ifelse(pred$net.result >= 0.5, 1, 0)

NEURAL NETWORKS

22
0

pred_class
#Convert predicted class to factor with appropriate levels
pred_class <- factor(pred_class, levels = c(0, 1), labels = c('No', 'Yes'))

#Ensure that df$Acceptance is a factor
df$recommendation
df$recommendation<- factor(df$recommendation, levels = c(0, 1), labels = c('No', 'Yes'))

#Create confusion matrix
confusionMatrix(pred_class, df$recommendation)

#problem-3 NN WITH THREE HIDDEN LAYER AND FOUR NODES
df <- read.csv(file.choose(),header=TRUE)
View(df)
names(df)
names(df)<-
c('age','spend','Income','mode','recommendation','age_norm','spend_norm','Income_norm')

#Set seed for reproducibility
set.seed(1)

#Train the neural network model
nn <- neuralnet(recommendation ~ age_norm + spend_norm+Income_norm+mode, data = df,
linear.output = FALSE, hidden = c(10,10,10)) #THREE HIDDEN LAYER AND TEN NODE
EACH
#display weights
nn$weights
#display predictions
prediction(nn)
#plot network
plot(nn,rep='best')

#Make predictions
pred <- compute(nn, df[, c('age_norm','spend_norm','Income_norm','mode')])
pred

#Extract predicted class
pred_class <- ifelse(pred$net.result >= 0.5, 1, 0)
pred_class

#Convert predicted class to factor with appropriate levels
pred_class <- factor(pred_class, levels = c(0, 1), labels = c('No', 'Yes'))

NEURAL NETWORKS

22
1

#Ensure that df$Acceptance is a factor
df$recommendation
df$recommendation<- factor(df$recommendation, levels = c(0, 1), labels = c('No', 'Yes'))

#Create confusion matrix
confusionMatrix(pred_class, df$recommendation)
################################################################
#problem-3 NN WITH FIVE HIDDEN LAYER AND TEN NODES
df <- read.csv(file.choose(),header=TRUE)
View(df)
names(df)
names(df)<-
c('age','spend','Income','mode','recommendation','age_norm','spend_norm','Income_norm')

#Set seed for reproducibility
set.seed(1)

#Train the neural network model
nn <- neuralnet(recommendation ~ age_norm + spend_norm+Income_norm+mode, data = df,
linear.output = FALSE, hidden = c(10,10,10,10,10)) #THREE HIDDEN LAYER AND TEN NODE
EACH
#display weights
nn$weights
#display predictions
prediction(nn)
#plot network
plot(nn,rep='best')

#Make predictions
pred <- compute(nn, df[, c('age_norm','spend_norm','Income_norm','mode')])
pred

#Extract predicted class
pred_class <- ifelse(pred$net.result >= 0.5, 1, 0)
pred_class

#Convert predicted class to factor with appropriate levels
pred_class <- factor(pred_class, levels = c(0, 1), labels = c('No', 'Yes'))

#Ensure that df$Acceptance is a factor
df$recommendation
df$recommendation<- factor(df$recommendation, levels = c(0, 1), labels = c('No', 'Yes'))

NEURAL NETWORKS

22
2

#Create confusion matrix
confusionMatrix(pred_class, df$recommendation)







