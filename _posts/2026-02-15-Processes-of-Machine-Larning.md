---
layout: post
title: Consist of Machine Learning
date: 2026-02-15 19:20:23 +0900
category: sample
---

# Two key calculation of Machine Learning
if the dataset have eight features in each data and consist of five data, the input $x$ is the matrix $M^{5\times 8}$. Its column size is the number of feature and the row size is the number of data.
</br></br>



## Linear transformation
* formulation : $\rm Z^{(k)}=A^{(k-1)\top}Q^{(k)}+b^{(k)}$

&nbsp; This process makes input from previous layer 

**Multiplication of weight matrix** 
</br> &nbsp; This process simply multiplicate each input by its matching weight. It uses matrix $\rm M^{n\times m}$, where $\rm n$ represents previous layer's size  and $\rm m$ represents current layer's size. Each elements $m_{(i,j)}$ is multiplicated to the previous layer's ith neuron and become part of the current layer's jth neuron.

**Addition to bias**
</br> &nbsp; This process add bias to its matching weighted value, enhaencing the model's complexity. It uses vector $\rm b^{m}$, where $\rm m$ represents current layer's size. Each elements are added to corresponding elementes of vector $\rm A^{(k-1)\top}Q^{(k)}$.

**Relationship with hardware**
</br> &nbsp; The apparent simplicity of this expression makes its computational complexity. **The multiplication** requires high-thorughput arithmetic units and **the summation** demands accumulator units. When scaled across millions of neurons and billions of parameters, this memory access patterns necessitate **high-bandwitdh memory system**, becoming the performance bottleneck.
</br></br>



## Non-Linear activation
* formulation : $\rm A^{(k)}=f(Z^{(k)})$

&nbsp; this process makes weighted output $\rm Z$ from linear transformation into non-linear form, using **activation function**.
</br> &nbsp; What activation sunctions we use is determined depending on the model's kind, especially on form of outputs. (For example, we use softmax for multi-class classification)
</br> &nbsp; Since Most of dataset can't be represented in a linear form, we should go through. While linear models express relationship very simple form, non-linear models learn complex patterns and express them like a real-world. 

### four key activation function
**1. sigmoid**

**2. Tanh**

**3. ReLU**

**4. softmax**

### why Non-linear process is needed?
&nbsp; Non-linear activation convert linear output into non-linear output.
</br> &nbsp; Figure shows the reason : Since linear output wihtout an activation could not express nonlinear patterns in real-world, using non-linear activation functions is essential for real complex pattern recognition.
The universal approximation theorem 
</br></br>



# Neuron architecture
**Buttom-up approch**
</br> &nbsp; We build from simple to complex : neurons -> layers -> networks.
</br> &nbsp; As we discussed eariler, 


## Neurons



## Layers
In typical neural networks, we organize layers hierarchically: 
1. **Input layer** : Receives raw data features (or, the input from dataset that machines learn)
2. **Hidden layer** : Process and transfrom the data throudgh multiple stage, from individual data to implicated features.
3. **Output layer** : Produces the final prediction or decision (or, the output machines produce based on their learning)
</br> </br>




# Processes of Machine Larning
&nbsp; Learning has two parts : training and infrerence. 
</br> &nbsp; Trainning operates as a loop. Each iteration involves trainning batch (subests of the data). For each batch, it operates  several key operations.
</br>
1. Forward propagation to generate expectations.
2. Loss function to evaluate prediction accuracy.
3. Backward propagation to compute weight adjustment.
4. Weight update to improve future predictions.

Given the input $x$ and output $y$, we can expresse this processes mathmetically :
$$\hat y=f(x;\theta), \ Loss = L (\hat y,y)$$

where $f$ represents the neural network function, $\hat y$ represents the prediction value, and $\theta$ weights.
</br></br>



## Forward propagation
&ensp; **Forward propagation** is process where input data flows.
The process begins with the input layers, propagate forward through the hidden layers, and end with the output layer(s).
</br>&ensp; At each layer, the process have two key steps : a **linear transformation** (multiplication and addition) and **non-linear activation**.

1. first layer (input layer) : 
$$ Z^{(1)}=W^{(L)}X+b^{(1)} $$ $$ A^{(1)}=f^{(1)}(Z^{(1)})$$

2. hidden layer ($k=2,3,...,L-1$) :
$$Z^{(k)}=W^{(k)}A^{(k-1)}+b^{(k)} $$ $$ A^{(k)}=f^{(k)}(Z^{(k)})$$

3. last alyer (output layer) : 
$$Z^{(L)}=W^{(L)}A^{(L-1)}+b^{(L)}$$ $$ A^{(L)}=f^{(L)}(Z^{(L)})$$
</br>

All the models use activation function, yet they use different by their tasks. **ReLU** is for regression. And **Sigmoid** is for binary classification, while **softmax** is for multi class classification.



## Error function
The error functions we use depend on models. For example, for multi-class classification, we can use cross entrpy (CEE), and mean square error for regression.
 
### of error functions

**1. mean squared error**
</br>**mean squared function** is utilized for regression tasks.  

**2. cross entrophy error**
</br>**cross entrophy function** is utilized for classification tasks.  
Binary classification, asking how much does a data belong to a class, is use sine the probability should be expressed with number between 0 and 1. On the other hand, Multi-class classification utilizes because it makes each results like the probabilties of their belongings. 


## Backward propagation (backpropagation)



## Weight update & Optimization