# Author: Jordan Bannister 11.03.18 

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def main():
    """ Trains a logistic regression model on the MNIST handwritten digit dataset """

    # Tensorflow provides the mnist data in it's api
    mnist = input_data.read_data_sets("/project/data", one_hot=True)
    _check_data(mnist)

    # tf is kind of like a functional language with lazy evaluation.
    # You define a graph of operations, but nothing is computed until
    # you explicitly run an operation in a tf.Session()

    training_iterations = 20000
    training_batch_size = 100
    learning_rate = 1e-5
    regularization = 1e-4

    graph_parameters = _create_graph_parameters()
    graph = _create_regression_model(learning_rate, graph_parameters, regularization)

    training_accuracies = np.zeros(training_iterations)
    training_losses = np.zeros(training_iterations)

    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())

        # Train the model
        loop = tqdm(range(training_iterations))
        for i in loop:
            train_batch = mnist.train.next_batch(training_batch_size)
            _train_step(sesh, graph, train_batch)

            # monitor progress
            training_accuracies[i] = _accuracy(sesh, graph, train_batch)
            training_losses[i] = _loss(sesh, graph, train_batch)

        print("Training is finished!\n\n")

        # Check the accuracy on all of the training data
        train_batch = mnist.train.next_batch(55000)
        train_accuracy = _accuracy(sesh, graph, train_batch)

        # Check the accuracy on all of the testing data
        test_batch = mnist.test.next_batch(10000)
        test_accuracy = _accuracy(sesh, graph, test_batch)

        print("Final training accuracy: ", train_accuracy)
        print("Final testing accuracy: ", test_accuracy)

        # Visualizations
        _plot_learning_curve(training_accuracies, training_losses)

        weights = sesh.run(graph_parameters['weights'])
        _show_weights(weights)


###############################################################################


def _plot_learning_curve(training_accuracies, training_losses):

    plt.plot(training_losses, '.')
    plt.ylabel("Training loss")
    plt.xlabel("iterations")
    plt.show()

    plt.plot(training_accuracies, '.')
    plt.ylabel("Training accuracy")
    plt.xlabel("iterations")
    plt.show()


def _accuracy(sesh, graph, batch):
    return graph['accuracy'].eval(
        feed_dict=
        {graph['x']: batch[0], graph['y']: batch[1]})


def _loss(sesh, graph, batch):
    return graph['average_loss'].eval(
        feed_dict=
        {graph['x']: batch[0], graph['y']: batch[1]})
    

def _train_step(sesh, graph, batch):
    graph['optimizer'].run(
        feed_dict=
        {graph['x']: batch[0], graph['y']: batch[1]})


def _check_data(mnist):
    """ Render a few images from mnist dataset """

    # There are 55000 training images and 10000 test images
    # each image is 28 pixels by 28 pixels (d=784)
    print("\nTraining data images shape: ", mnist.train.images.shape)
    print("Testing data images shape: ", mnist.test.images.shape)

    # The labels are encoded in 1-hot vectors of length 10
    print("\nTraining data labels shape: ", mnist.train.labels.shape)
    print("Testing data labels shape: ", mnist.test.labels.shape)

    # Check a few images
    for index in range(0, 5):
        _show_image(mnist, index)


def _show_image(mnist, training_image_index):
    """ Displays the specified image and prints the digit label """

    test_label = np.where(mnist.train.labels[training_image_index] == 1)[0]

    test_image = np.reshape(
        mnist.train.images[training_image_index], [28, 28])

    plt.imshow(test_image, cmap='Greys')
    plt.title("Label: " + str(test_label))
    plt.colorbar()
    plt.show()


def _show_weights(weights):
    """ Visualizes the weights of the glm as images """

    v_max = max(np.max(weights), abs(np.min(weights)))
    v_min = -v_max

    for index in range(10):
        weight_image = np.reshape(weights[:, index], [28, 28])
        plt.imshow(weight_image, vmin=v_min, vmax=v_max, cmap='bwr')
        plt.title("Digit " + str(index) + " weights")
        plt.colorbar()
        plt.show()

def _create_weight_variable(shape):
    """ Helper function to create and initialize weight variables """

    init = tf.zeros(shape)
    return tf.Variable(init)


def _create_bias_variable(shape):
    """ Helper function to create and initialize bias variables """

    init = tf.zeros(shape)
    return tf.Variable(init)


def _create_graph_parameters():
    """ Create the trainable parameters for the model """

    graph_parameters = {}

    graph_parameters['weights'] = _create_weight_variable([784, 10])
    graph_parameters['biases'] = _create_bias_variable([10])

    return graph_parameters


def _create_regression_model(learning_rate, graph_parameters, regularization):
    """ Create a logistic regression model for the MNIST data """

    graph = {}

    # Placeholders are the operations that accept input data when you run your graph
    graph['x'] = tf.placeholder(tf.float32, [None, 784])
    graph['y'] = tf.placeholder(tf.float32, [None, 10])

    # The linear estimator is just a general linear model: y = W*x +B
    linear_estimator = tf.add(
            tf.matmul(graph['x'], graph_parameters['weights']),  
            graph_parameters['biases'])

    # predictions and accuracy
    predicted_class = tf.argmax(linear_estimator, 1)
    correct_prediction = tf.equal(predicted_class, tf.argmax(graph['y'], 1))

    graph['accuracy'] = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32))

    # loss function
    loss = tf.add(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=graph['y'], 
                logits=linear_estimator),
            regularization*tf.norm(graph_parameters['weights'], ord=1))

    graph['average_loss'] = tf.reduce_mean(loss)


    # Adam is a gradient descent optimization algorithm 
    graph['optimizer'] = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return graph

###############################################################################

if __name__ == '__main__':
    main()
