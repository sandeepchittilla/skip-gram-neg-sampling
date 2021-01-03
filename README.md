# skip-gram-neg-sampling
Compute similarity between words using a Skip-gram with Negative Sampling approach. Additionally, we do a manual implementation of the feed-forward and back-propagation steps as well as the negative sampling and embedding steps.

## Data
We use the [SimLex-999 dataset](https://fh295.github.io/SimLex-999.zip) as a ground truth table for text association scores between two words. More information about it can be found [here](https://fh295.github.io/simlex.html)

For training we use the [1 Billion Words Corpus](https://opensource.google/projects/lm-benchmark)

## Solution

The entire code can be found in skipGram.py

### PreProcessing


We use spaCy en.core.web.sm for tokenization. We then : 
- convert to lower case
- lemmatization
- remove punctuations
- filter stop words and tokens of size <3

### SkipGram Training

We model a neural network with 1 hidden layer and the following loss function :

<img src="https://render.githubusercontent.com/render/math?math=L(t,c) = -log[\sigma(c.t)] %2B \sum_{n\in Neg}log[\sigma(-n.t)]">

where t is the targeted word, c the context word, and n ∈ N the sample of negative words.

The input to the network is the target word and the output is probabilities for all words to be in the target word's context words.

The forward propogation is relatively simple operation and we perform this along with computing partial differentials for the back propagation in ```trainWord``` function of the SkipGram class.

### Similarity Scores

The idea is that, after training the model and saving the weights matrix, the words with high similarity _(note that we define similarity based on usage and this in turn is determined by the context words surrounding the target word)_ will have similar weights. 

In order to predict we pass the ```--test``` argument to the script which :
- loads the word pairs for which we want to predict the similarity
- loads pre-trained weights matrix
- calls the ```similarity``` function which computes the similarity i.e. _dot product_ of the weight vectors (we retrieve by id from the weights matrix) for each word pair
