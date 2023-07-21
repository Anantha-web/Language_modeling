#import the below libraies, the libraries we using 
import string
import random
import os
import numpy as np
import torch
import nltk
#nltk.download('punkt')

#Perplexity returning
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

print("opening the files")
file1 = open('2020101103-LM1-train-perplexity.txt', 'w')
file2 = open('2020101103-LM1-validation-perplexity.txt', 'w')
file3 = open('2020101103-LM1-test-perplexity.txt', 'w')

with open('data.txt', 'r') as f:
    data = f.read()
data = data.lower()

print("making clear data")
punctuation = string.punctuation
punctuation = [sym for sym in punctuation if sym not in [
    ',', '.', ';', ':', '!', '?']]
punctuation.append('\n')
endMarker = [',', '.', ';', ':', '!', '?']

data = data.replace("'s ", "")
data = "".join([c if c not in punctuation else ' ' for c in data])
data = "".join([c if c not in endMarker else '.' for c in data])
sentences = data.split('.')

#Tokenize the sentences
def tokenize(sentences):
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    return sentences
sentences = sentences[:30000]

#Vocabulary words in the Data
vocabulary = []
for sentence in sentences:
    vocabulary.extend(tokenize([sentence])[0])
vocabulary = set(vocabulary)
print("Loading Vocabulary")

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

word_vector_dimension = 300
glove_vectors = load_glove_vectors(f'glove.6B.{word_vector_dimension}d.txt')
vocabulary = list(vocabulary)

def quadgram(sentences):
    quadgrams = []
    for idx, sentence in enumerate(sentences):
        length = len(sentence)
        for i in range(length - 4):
            try:
                x = (np.array(glove_vectors[sentence[i]]) + np.array(glove_vectors[sentence[i+1]]) + np.array(
                    glove_vectors[sentence[i+2]]) + np.array(glove_vectors[sentence[i + 3]]))/4
                y = [0 for i in range(len(vocabulary))]
                idx = vocabulary.index(sentence[i+4])
                y[idx] = 1
                quadgrams.append([x, y])
            except KeyError:
                continue
    return quadgrams
total_samples = len(sentences)

#Split the samples into Training, Validation and Testing
train_samples = sentences[:int(total_samples * 0.7)]
random.shuffle(train_samples)
validation_samples = sentences[int(
    total_samples * 0.7):int(total_samples * 0.8)]
random.shuffle(validation_samples)
test_samples = sentences[int(total_samples * 0.8):]

language_model = torch.nn.Sequential(
    torch.nn.Linear(word_vector_dimension, 300),
    torch.nn.ReLU(),
    torch.nn.Linear(300, 300),
    torch.nn.ReLU(),
    torch.nn.Linear(300, len(vocabulary)),
    torch.nn.Softmax(dim=1)
)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(language_model.parameters(), lr=0.001)

epochs = 1
batch_size = 5
for epoch in range(epochs):
    #Training
    avg = 0
    num_samples = 0
    for idx, sentence in enumerate(train_samples):
        sentence.strip()
        train_sample = tokenize([sentence])
        train_sample = quadgram(train_sample)
        if len(train_sample) == 0:
            continue
        num_samples += 1
        batch = train_sample
        input = torch.stack([torch.tensor(x[0]) for x in batch])
        target = torch.stack([torch.tensor(x[1]) for x in batch])
        prediction = language_model(input.float())
        optimizer.zero_grad()
        loss = loss_function(prediction, target.float())
        loss.backward()
        optimizer.step()
        score = perplexity(prediction, target)
        avg = (num_samples*avg + score) / (num_samples + 1)
        file1.write(f"{sentence}    {score}\n")
        print(f"training progress: {(idx + 1)*100/len(train_samples)}%")
    file1.write(f"{avg}\n")
    print(avg)
    print(f"Epoch {epoch + 1} loss: {loss.item()}")

    #Validation
    avg = 0
    num_samples = 0
    for idx, sentence in enumerate(validation_samples):
        sentence.strip()
        validation_sample = tokenize([sentence])
        validation_sample = quadgram(validation_sample)
        if len(validation_sample) == 0:
            continue
        num_samples += 1
        batch = validation_sample
        input = torch.stack([torch.tensor(x[0]) for x in batch])
        target = torch.stack([torch.tensor(x[1]) for x in batch])
        prediction = language_model(input.float())
        optimizer.zero_grad()
        loss = loss_function(prediction, target.float())
        loss.backward()
        optimizer.step()
        score = perplexity(prediction, target)
        avg = (num_samples*avg + score) / (num_samples + 1)
        file2.write(f"{sentence}    {score}\n")
        print(f"progress of validation: {(idx + 1)*100/len(validation_samples)}%")
    file2.write(f"{avg}\n")
    print(avg)
    print(f"Epoch {epoch + 1} loss of validation: {loss.item()}")
# save the model
torch.save(language_model, '2020101103-LM1.pt')
#print('saved model')

#Testing
avg = 0
num_samples = 0
print("Testing")
for idx, sentence in enumerate(test_samples):
    sentence.strip()
    test_sample = tokenize([sentence])
    test_sample = quadgram(test_sample)
    if (len(test_sample) == 0):
        continue
    num_samples += 1
    batch = test_sample
    input = torch.stack([torch.tensor(x[0]) for x in batch])
    target = torch.stack([torch.tensor(x[1]) for x in batch])
    prediction = language_model(input.float())
    score = perplexity(prediction, target)
    avg = (num_samples*avg + score) / (num_samples + 1)
    file3.write(f"{sentence}    {score}\n")
    print(f"Test progress: {(idx + 1)*100/len(test_samples)}%")
file3.write(f"{avg}\n")
print(avg)

#Closing write('w') files
file1.close()
file2.close()
file3.close()