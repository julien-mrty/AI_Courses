"""
IV) Unsupervised learning
Chapter 14
Self-supervised learning and foundation models
"""


"""
Pre-training and adaptation : 

Pre-training and adaption are the two main phases in the foundation models paradigm.

Pretraining Phase :
- A large model is trained on a massive, often unlabeled dataset (ex : billions of images).
- The goal is for the model to learn useful representations that capture intrinsic semantic structure in the data.
- The model fθ maps input data x to an embedding or feature space. The loss function L_pre(θ) is optimized using methods
  like SGD or Adam.
- Pretraining may also use self-supervised loss functions, where supervision comes from the data itself.

Example of self-supervised loss function : 
The contrastive Loss encourages similar data points (two augmented views of the same image) to have similar 
representations, while pushing apart the representations of dissimilar data points (will discuss this examples in the
next point) .


Adaptation Phase :
- After pretraining, the model is adapted to a specific downstream task (ex : cancer detection). This may involve 
prediction tasks with limited or no labeled data.
- Two key adaptation methods are:
    - Linear probe : A simple linear layer (or head) is trained on top of the pretrained model while keeping the 
      pretrained model fixed.
    - Fine-tuning : Both the linear layer (on top of the model) and the pretrained model are further trained on the 
      downstream data, updating the entire model.
- Tasks with no labeled data use zero-shot learning, while tasks with small labeled data (1–50 examples) use few-shot 
learning.
"""


"""
Pretraining methods in computer vision.

Supervised Pretraining :
- Involves using a large labeled dataset to train a neural network with supervised learning.
- After training, the last layer (usually fully connected) is discarded, and the penultimate layer's activations are 
  used as the model's learned representations.
- The goal here is to use a pretrained model for other tasks for which it was initially intended by removing the last 
  layer (the head) and replacing it with the head that we need.


Contrastive Learning :
- A self-supervised method using only unlabeled data.
- The goal is for similar images (ex : two pictures of a husky) to have similar representations, while dissimilar images
  (ex : a husky and an elephant) have distinct representations. 
- Positive pairs (augmentations of the same image) are pushed closer in the representation space, while negative pairs 
  (or random pairs : different images) are pushed apart.
- Data augmentation (ex : cropping, flipping) is used to create positive pairs from the same image, and random images 
  form negative pairs. To get similar images (positive paris) we use two data augmentations of the same image. Then, we 
  compare one of the augmented image with an augmented image of another class to get negative pairs.
- SIMCLR is an example of a contrastive learning algorithm, where a batch of images is augmented, and the loss 
  encourages representations of positive pairs to be close (by minimizing loss) while pushing apart representations of 
  negative pairs (by increasing loss).
"""



"""
Pretraining large language models (LLM)

Pretraining large language models (LLMs) in natural language processing (NLP).

Pretrained Language Models :
- A language model predicts the probability of a document based on a sequence of words. The probability is broken down 
  using the chain rule of conditional probability, predicting each word given the previous words.
- Words are transformed into embeddings, which are passed through a transformer model. The transformer outputs 
  contextual embeddings for each word based on the predictions of the previous embeddings. This allow the prediction of 
  the next word using a softmax function.
- There is two parameters to learn. One in the transformer. The second is a weight matrix that maps the contextualized 
  embedding at the output of the transformer (before the softmax is applied).
- The model is trained using cross-entropy loss, minimizing the difference between predicted and actual word 
  probabilities (no more precisions on the loss function in the lecture).


Zero-shot Learning :
- In zero-shot learning, the model is adapted to a new task without additional labeled examples. Tasks are framed as 
  natural language questions, and the model predicts the next word to solve the task.
- For example, given the question "Is the speed of light a universal constant?" the model predicts "No" based on its 
  pretraining.
  
  
In-context Learning :
- In few-shot settings, the model is given a few labeled examples and then prompted with a new task to generate 
  predictions based on prior examples.
- For instance, given math questions with solutions, the model can generalize the pattern and solve a new problem.
"""