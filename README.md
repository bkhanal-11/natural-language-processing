# Natural Language Processing

Study and applications of NLP algorithm prior transformers like RNNs, GRUs and LSTMs.

## Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a type of neural network commonly used for processing sequential data such as time series data or natural language. Unlike traditional feedforward neural networks, RNNs have a feedback loop that allows information to persist over time.

![RNN Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/1280px-Recurrent_neural_network_unfold.svg.png)

As shown in the diagram above, the input at each time step is fed into the RNN along with the hidden state from the previous time step. The hidden state is updated at each time step using the input and the previous hidden state. This allows the RNN to take into account previous inputs when making predictions.

## Long Short-Term Memory (LSTM)

One of the most popular types of RNNs is the Long Short-Term Memory (LSTM) network. LSTMs are designed to address the vanishing gradient problem that can occur in traditional RNNs, where the gradients become very small and cause the network to stop learning.

![LSTM Diagram](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

LSTMs use a series of gates to control the flow of information through the network. These gates allow the network to selectively forget or remember information from previous time steps, which helps prevent the gradients from becoming too small.
Applications of RNNs

RNNs and LSTMs have been successfully applied to a wide range of applications, including:

- Language modeling
- Speech recognition
- Machine translation
- Image captioning
- Time series prediction
