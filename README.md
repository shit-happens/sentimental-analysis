# sentimental-analysis
A retail firm names 'Poseidon' has employed a chatbot on its progressive web app to interact with its customers. 
My job was to determine the core intent of the customer's interaction (query) and classify it into one of the following 10
different tags :
0 - order.cancellation
1 - order.concern.delay
2 - order.modification
3 - order.status
4 - product.browse
5 - product.reviews
6 - shipping.address.modification
7 - shipping.plans.browse
8 - store.browse
9 - store.timing

Queries are customer interactions with a chatbot, the tags classify them on a broader basis. To train a learning algorithm 
on this text data, I used Long short-term memory (LSTM) which is an artificial recurrent neural network (RNN) architecture
used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not
only process single data points (such as images), but also entire sequences of data (such as speech or video).

As per the plots of Training and Validation loss V/S epochs, it can be seen that the my model was able to achieve a pretty
decent accuracy of 83.98% accuracy on validation set.
