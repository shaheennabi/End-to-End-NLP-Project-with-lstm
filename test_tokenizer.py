import pickle
import os

# Specify the full path to your tokenizer.pickle file
tokenizer_path = os.path.join('tokenizer', 'tokenizer.pickle')

try:
    # Open and load the tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Perform a simple test with the tokenizer
    test_sentences = ["This is a test sentence.", "Another sentence for verification."]
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    
    print("Tokenizer successfully loaded.")
    print("Test sequences:", test_sequences)

except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
