# BART Conversation Clone

This projects files aim to fine-tunes Facebook's `bart-base` model to generate personalized responses based on your past message-reply pairs.

## Features
- cleans meta data scraped from WhatsApp web
- Tokenizes and preprocesses message-reply pairs from CSV
- Fine-tunes BART on your chat history using PyTorch
- Tracks and saves the best model based on average loss
- Plots and exports a loss curve (`loss_graph.pdf`)

## Files
- `main.py` — Training script
- `encoded_data.py` — Custom PyTorch Dataset
- `model_selector.py` — Logic to track best models during training
- `message_reply_pairs.csv` — Your message–reply data | not visible on git
- `requirements.txt` — Python dependencies
- `clean_meta_data.ipynb` - python notebook that cleans the meta data

