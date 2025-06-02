import jax
import jax.numpy as jnp
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Machine:
    def __init__(self, data, hd1, hd2, lr, bs, e):
        #Setting Fields
        self.data = data
        self.hiddenDim1 = hd1
        self.hiddenDim2 = hd2
        self.learningRate = lr
        self.batchSize = bs
        self.epochs = e

        # Train-test split and scaling
        self.input = data[:, :-1]
        self.inputDim = self.input.shape[1]
        self.output = data[:, -1].reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        self.output = encoder.fit_transform(self.output)
        self.outputDim = self.output.shape[1]

        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            self.input,
            self.output,
            test_size=0.15,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.xTrain = self.scaler.fit_transform(self.xTrain)
        self.xTest = self.scaler.transform(self.xTest) 

        #randomKey = jax.random.key(int(time.time())) #different key based on time
        randomKey = jax.random.PRNGKey(int(time.time())) #alternative method
        self.params = self.init_params(self.inputDim, self.hiddenDim1, self.hiddenDim2, self.outputDim, randomKey)

        self.run()

    def init_params(self, inputDim, hiddenDim1, hiddenDim2, outputDim, randomKey):
        randomKey = jax.random.split(randomKey, 3) #spliting into three keys

        #first weight matrix
        W1 = jax.random.normal(randomKey[0], (inputDim, hiddenDim1))
        B1 = jnp.zeros((hiddenDim1, ))

        W2 = jax.random.normal(randomKey[1], (hiddenDim1, hiddenDim2))
        B2 = jnp.zeros((hiddenDim2, ))

        W3 = jax.random.normal(randomKey[2], (hiddenDim2, outputDim))
        B3 = jnp.zeros((outputDim, ))

        return W1, B1, W2, B2, W3, B3
        
    def forward(self, params, X):
        W1, B1, W2, B2, W3, B3 = params
        h1 = jax.nn.relu(jnp.dot(X, W1) + B1)
        h2 = jax.nn.relu(jnp.dot(h1, W2) + B2)
        logits = jnp.dot(h2, W3) + B3
        return logits
    
    def lossFN(self, params, x, y, l2_reg=0.0001):
        #somthing about a derivative, gradient, and optimization
        logits = self.forward(params, x)
        probs = jax.nn.softmax(logits, axis=1)
        l2_loss = l2_reg * sum([jnp.sum(w ** 2) for w in params[::2]]) #soemthing that leads to smaller weights
        return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-8), axis=1)) + l2_loss
    
    def train(self, params, x, y, lr):
        grads = jax.grad(self.lossFN)(params, x, y)
        return [(param - lr * grad) for param, grad in zip(params, grads)]
    
    def accuracy(self, params, x, y):
        preds = jnp.argmax(self.forward(params, x), axis=1)
        targets = jnp.argmax(y, axis=1)
        return jnp.mean(preds == targets)
    
    def dataLoader(self, X, y, batchSize): #for saving data to JSON
        for i in range(0, len(X), batchSize):
            yield X[i:i+batchSize], y[i:i+batchSize]

    def predict(self, data):
        """
        Takes in a single data sample (or batch) and returns the predicted class label(s).
        """
        # Ensure the input data has the correct shape
        if len(data.shape) == 1:
            data = data.reshape(1, -1)  # Reshape single sample to batch format

        # Scale the input data using the trained scaler
        data = self.scaler.transform(data)

        # Perform forward pass and get predicted class
        logits = self.forward(self.params, data)
        predicted_class = jnp.argmax(logits, axis=1)

        return predicted_class

    def run(self):
        for epoch in range(self.epochs):
            for i in range(0, len(self.xTrain), self.batchSize):
                xBatch, yBatch = self.xTrain[i:i+self.batchSize], self.yTrain[i:i+self.batchSize]
                self.params = self.train(self.params, xBatch, yBatch, self.learningRate)

            if epoch % 10 == 0:
                train_acc = self.accuracy(self.params, self.xTrain, self.yTrain)
                test_acc = self.accuracy(self.params, self.xTest, self.yTest)
                print(f"Epoch {epoch}: Train Acc ({train_acc:.4f}), Test Acc ({test_acc:.4f})")

        print(f"Final Test Accuracy: {self.accuracy(self.params, self.xTest, self.yTest):.4f}")