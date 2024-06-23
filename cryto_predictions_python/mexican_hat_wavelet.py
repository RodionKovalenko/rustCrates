
import pywt
import numpy as np

wavelist = pywt.wavelist(kind='continuous')
print('wavelist', wavelist)

# wavelist = pywt.wavelist(kind='discrete')
# print('wavelist', wavelist)

wavelet = pywt.Wavelet('db2')

[dec_lo, dec_hi, rec_lo, rec_hi] = wavelet.filter_bank

print('dec_lo', dec_lo)
print('dec_hi', dec_hi)
print('rec_lo', dec_lo)
print('rec_hi', dec_hi)


scales = np.arange(1, 3)

# sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7, 8, 9])
# print('data', sig)
# ca, cd = pywt.dwt(sig, 'db3', mode)
# print('ca', ca)
# print('cd', cd)


# sig = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8, 9]])
# print('data', sig)
# ca, cd = pywt.dwt2(sig, 'db1', mode)
# print('ca', ca)
# print('cd', cd)

w_type = 'db2'
mode = 'periodization'
sig = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])

wavelet = pywt.dwt2(sig, w_type, mode)
print('wavelet', w_type, wavelet)


sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
padded = pywt.pad(sig, (10, 10), 'periodization')
print('padded', padded)

import tiktoken

# Initialize the tokenizer with a specific encoding (e.g., 'gpt2')
tokenizer = tiktoken.get_encoding("gpt2")

# Example text
text = "Hello, world!!!"

# Tokenize the text
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

# Decode the token IDs back into text
decoded_text = tokenizer.decode(token_ids)
print("Decoded Text:", decoded_text)

# Decode each token ID to see individual tokens
tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
print("Individual Tokens:", tokens)


