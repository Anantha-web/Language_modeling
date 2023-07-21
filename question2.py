#libraries to import
# %%
import string
import random
import os
import numpy as np
import torch
import nltk
#nltk.download('punkt')

# %%
def perplexity(prediction, target):
    perplexity = 0
    prediction = prediction.detach().numpy()
    target = target.detach().numpy()
    for i in range(len(prediction)):
        val = np.dot(prediction[i], target[i])
        if val != 0:
            perplexity += np.log2(val)
    perplexity = 2 ** (-perplexity / len(prediction))
    return perplexity

# %%
print("opening the files")
file1 = open('2020101103-LM2-train-perplexity.txt', 'w')
file2 = open('2020101103-LM2-validation-perplexity.txt', 'w')
file3 = open('2020101103-LM2-test-perplexity.txt', 'w')

# %%
with open('data.txt', 'r') as f:
    data = f.read()

# %%
data = data.lower()

# %%
print("making clear data")
punctuation = string.punctuation
punctuation = [sym for sym in punctuation if sym not in [
    ',', '.', ';', ':', '!', '?']]
punctuation.append('\n')
endMarker = [',', '.', ';', ':', '!', '?']
data = data.replace("'s ", "")
data = "".join([c if c not in punctuation else ' ' for c in data])
data = "".join([c if c not in endMarker else '.' for c in data])
#Split the data into sentences
sentences = data.split('.')
#Tokenize the sentences
def tokenize(sentences):
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    return sentences

# %%
sentences = sentences[:30000]

# %%
# create a vocabulary of all the words in the data
vocabulary = []
for sentence in sentences:
    vocabulary.extend(tokenize([sentence])[0])
vocabulary = set(vocabulary)
print("Loading Vocabulary")

# %%
def load_glove_vectors(File):
    print("Glove vectors Loading")
    glove_vectors = {}
    file_size = os.path.getsize(File)
    bytes_read = 0
    with open(File, 'r') as f:
        for line in f:
            bytes_read += len(line)
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            if word in vocabulary:
                glove_vectors[word] = embedding
            print(f"glove progress:{bytes_read*100/ file_size }%")
    print(f"{len(glove_vectors)} words loaded!")
    return glove_vectors

# %%
word_vector_dimension = 300
glove_vectors = load_glove_vectors(f'glove.6B.{word_vector_dimension}d.txt')

# %%
vocabulary = list(vocabulary)

# %%
def gram(sentences):
    grams = []
    for idx, sentence in enumerate(sentences):
        length = len(sentence)
        for i in range(length - 1):
            try:
                x = glove_vectors[sentence[i]]
                y = [0 for i in range(len(vocabulary))]
                idx = vocabulary.index(sentence[i+1])
                y[idx] = 1
                grams.append([x, y])
            except KeyError:
                continue
    return grams

# %%
total_samples = len(sentences)

# %%
# split the samples into training, validation and testing
train_samples = sentences[:int(total_samples * 0.7)]
random.shuffle(train_samples)
validation_samples = sentences[int(
    total_samples * 0.7):int(total_samples * 0.8)]
random.shuffle(validation_samples)
test_samples = sentences[int(total_samples * 0.8):]

# %%
class MyRNN(torch.nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.ht_1 = torch.zeros(batch_size, 300)
        self.h = torch.nn.Sequential(
            torch.nn.Linear(300 + word_vector_dimension, 300),
            torch.nn.Tanh(),
        )
        self.y = torch.nn.Sequential(
            torch.nn.Linear(300, len(vocabulary)),
            torch.nn.Softmax(),
        )
    def setBatch_size(self, batch_size):
        self.batch_size = batch_size
    def forward(self, xt):
        input = [[] for i in range(self.batch_size)]
        for i in range(self.batch_size):
            input[i].extend(xt[i])
            input[i].extend(self.ht_1[i])
        input = torch.tensor(input)
        self.ht = self.h(input)
        yt = self.y(self.ht)
        return yt

# %%
epochs = 1
batch_size = 5
language_model = MyRNN(batch_size)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(language_model.parameters(), lr=0.001)

# %%
score = 1
for epoch in range(epochs):
    #Training
    avg = 0
    num_samples = 0
    print("Training")
    for idx, sentence in enumerate(train_samples):
        sentence.strip()
        train_sample = tokenize([sentence])
        train_sample = gram(train_sample)
        if len(train_sample) == 0:
            continue
        num_samples += 1
        for i in range(0, len(train_sample), batch_size):
            if (i + batch_size > len(train_sample)):
                break
            score = 1
            batch = train_sample[i:i+batch_size]
            x = torch.tensor(np.array([x for x, y in batch]))
            y = torch.tensor(np.array([y for x, y in batch]))
            yt = language_model.forward(x.float())
            score *= perplexity(yt, y.float())
            optimizer.zero_grad()
            loss = loss_function(yt, y.float())
            loss.backward()
            optimizer.step()
        avg = (num_samples*avg + score) / (num_samples + 1)
        file2.write(f"{sentence}    {score}\n")
        print(f"Training progress: {(idx + 1)*100/len(train_samples)}%")
    file2.write(f"{avg}\n")
    print(avg)
    #Validation
    avg = 0
    num_samples = 0
    print("Validation")
    for idx, sentence in enumerate(validation_samples):
        sentence.strip()
        validation_sample = tokenize([sentence])
        validation_sample = gram(validation_sample)
        if len(validation_sample) == 0:
            continue
        num_samples += 1
        for i in range(0, len(validation_sample), batch_size):
            if (i + batch_size > len(validation_sample)):
                break
            score = 1
            batch = validation_sample[i:i+batch_size]
            x = torch.tensor(np.array([x for x, y in batch]))
            y = torch.tensor(np.array([y for x, y in batch]))
            yt = language_model.forward(x.float())
            score *= perplexity(yt, y.float())
        avg = (num_samples*avg + score) / (num_samples + 1)
        file3.write(f"{sentence}    {score}\n")
        print(f"Validation progress: {(idx + 1)*100/len(validation_samples)}%")
    file2.write(f"{avg}\n")
    print(avg)
    print(f"Epoch {epoch + 1} loss of validation: {loss.item()}")
# save the model
torch.save(language_model, '2020101103-LM2.pt')
#print("saved model")

# %%
#Testing
avg = 0
num_samples = 0
print("Testing")
for idx, sentence in enumerate(test_samples):
    sentence.strip()
    test_sample = tokenize([sentence])
    test_sample = gram(test_sample)
    if (len(test_sample) == 0):
        continue
    num_samples += 1
    for i in range(0, len(test_sample), batch_size):
        if (i + batch_size > len(test_sample)):
            break
        score = 1
        batch = test_sample[i:i+batch_size]
        x = torch.tensor(np.array([x for x, y in batch]))
        y = torch.tensor(np.array([y for x, y in batch]))
        yt = language_model.forward(x.float())
        score *= perplexity(yt, y.float())
    avg = (num_samples*avg + score) / (num_samples + 1)
    file3.write(f"{sentence}    {score}\n")
    print(f"Test progress: {(idx + 1)*100/len(test_samples)}%")
file3.write(f"{avg}\n")
print(avg)

# %%
file1.close()
file2.close()
file3.close()