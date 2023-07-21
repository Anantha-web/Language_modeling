# Language_modeling

## Advanced NLP
#### Assignment-1
### Name: Anantha Lakshmi
### Roll. No: 2020101103

- Source codes uploaded in zip file as extension .py
  
OneDrive links for the saved models:
- LM1 - https://iiitaphyd-my.sharepoint.com/:u:/g/personal/yadavalli_lakshmi_students_iiit
_ac_in/EaXBCeqC3ZFFjulvcAAaEbUBfwXPIBeOJIUuWn3Cw0ZvbA?e=pmtcqV
- LM2 - https://iiitaphyd-my.sharepoint.com/:u:/g/personal/yadavalli_lakshmi_students_iiit
_ac_in/EbXsy-MZcIdAh9UPy9G0IOgB0fc0r2zx1nwpBeEyQNP7KQ?e=z7mDjY
- LM1 and LM2 output files submitted with perplexities. (I created folders)

Used 30k sentences for the language model implementation.

#### Code:

Code Run instructions:
To run the code:
- Run ‘python3 question1.py’ for the first question
- Run ‘python3 question2.py’ for the second question

Files used:
- data.txt (which is provided in the assignment pdf as ‘brown.txt’)
- glove.6B.300d.txt(Which is pretrained 300 dimensional glove)

## 3. Analysis:
#### LM1:
LM1 average perplexity score on train data - 2.274075899756345e+41
LM1 average perplexity score on test data - 2.7798431350082497e+41LM1 average perplexity score on validation data - 2.1083724694231478e+40
#### LM2:
1. LM1 average perplexity score on train data - 8.289116050499923E+24
2. LM1 average perplexity score on test data - 6.418237282758303E+22
3. LM1 average perplexity score on validation data - 9.5344724694231478E+40

LM2 gave us better perplexity on three of the sets: Train, Test, and Validation.
RNN model performed ‘1e19’ times better than the 2-layer ne
