import torch
from model import CustomTransformer
from tokenizer import CustomTokenizer
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

class AIChat:
    def __init__(self, model_path='checkpoints/model_epoch_1.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Keep existing initialization code
        self.tokenizer = CustomTokenizer()
        vocab_path = Path("tokenizer/vocab.json")
        merges_path = Path("tokenizer/merges.txt")
        
        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(
                "Tokenizer files not found. Please run train_tokenizer.py first to create them."
            )
        
        try:
            self.tokenizer.initialize_tokenizer(str(vocab_path), str(merges_path))
            # Test tokenizeru
            test_text = "Test tokenizeru"
            encoded = self.tokenizer.encode(test_text)
            decoded = self.tokenizer.decode(encoded.ids)
            if not decoded.strip():
                raise ValueError("Tokenizer test failed - empty output")
        except Exception as e:
            logging.error(f"Error initializing tokenizer: {str(e)}")
            raise ValueError("Failed to initialize tokenizer")
        
        # Load vocabulary for special tokens
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            
        # Special tokens
        self.sos_token = vocab.get("<sos>", 2)
        self.eos_token = vocab.get("<eos>", 3)
        self.pad_token = vocab.get("<pad>", 0)
        
        logging.info(f"Special tokens: SOS={self.sos_token}, EOS={self.eos_token}, PAD={self.pad_token}")
        
        # Inicializace modelu
        self.device = device
        self.model = CustomTransformer(vocab_size=30000).to(device)
        
        # Načtení vah modelu
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.debug = False
        
        logging.info("Model and tokenizer initialized successfully")
        
    def preprocess_text(self, text):
        """Předzpracování vstupního textu"""
        text = text.strip()
        return text
        
    def generate_response(self, prompt, max_length=100, temperature=0.7):
        try:
            # Preprocessing and tokenization
            processed_prompt = self.preprocess_text(prompt)
            logging.info(f"Preprocessed prompt: {processed_prompt}")
            
            try:
                encoded = self.tokenizer.encode(processed_prompt)
                logging.info(f"Encoded tokens: {encoded}")
            except Exception as e:
                logging.error(f"Tokenization error: {str(e)}")
                return "Omlouvám se, ale nemohu zpracovat tento vstup."
            
            if not encoded:
                return "Omlouvám se, ale nemohu zpracovat tento vstup."
            
            # Add SOS and EOS tokens - použijeme .ids pro získání seznamu ID tokenů
            input_ids = [self.sos_token] + encoded.ids + [self.eos_token]
            logging.info(f"Final input tokens: {input_ids}")
            
            # Create input tensor with correct shape [batch_size=1, sequence_length]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            if self.debug:
                logging.info(f"Initial input shape: {input_tensor.shape}")
            
            generated_tokens = []
            current_input = input_tensor
            
            with torch.no_grad():
                for i in range(max_length):
                    outputs = self.model(current_input)
                    if self.debug:
                        logging.info(f"Model output shape: {outputs.shape}")
                    
                    # Get logits for the last token
                    next_token_logits = outputs[:, -1, :]
                    
                    # Upravené parametry pro malý model
                    next_token_logits = torch.clamp(next_token_logits, -10, 10)  # menší rozsah pro stabilitu
                    next_token_logits = next_token_logits / temperature
                    
                    # Aplikujeme softmax
                    next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Nahradíme NaN hodnoty nulami
                    next_token_probs = torch.nan_to_num(next_token_probs, 0.0)
                    
                    # Pokud jsou všechny pravděpodobnosti nulové, použijeme uniformní rozdělení
                    if next_token_probs.sum() == 0:
                        next_token_probs = torch.ones_like(next_token_probs) / next_token_probs.size(-1)
                    
                    # Snížený top-k pro přesnější výběr
                    top_k = 20  # Increased from 10 to 20
                    top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k)
                    
                    # Normalizujeme pravděpodobnosti top-k tokenů
                    top_k_probs = top_k_probs / top_k_probs.sum()
                    
                    try:
                        # Sample next token
                        next_token_idx = torch.multinomial(top_k_probs[0], 1)
                        next_token_id = top_k_indices[0, next_token_idx]
                    except RuntimeError as e:
                        logging.error(f"Sampling error: {str(e)}")
                        # Fallback: vybereme token s nejvyšší pravděpodobností
                        next_token_id = top_k_indices[0, 0].unsqueeze(0)
                    
                    token_item = next_token_id.item()
                    generated_tokens.append(token_item)
                    #logging.info(f"Generated token {i}: {token_item}")
                    
                    # Přísnější podmínka pro ukončení generování
                    if token_item == self.eos_token or i >= max_length - 1:
                        break
                    
                    # Add new token to input sequence
                    next_token_tensor = next_token_id.view(1, 1)
                    if self.debug:
                        logging.info(f"Next token tensor shape: {next_token_tensor.shape}")
                        logging.info(f"Current input shape before cat: {current_input.shape}")
                    
                    try:
                        current_input = torch.cat([current_input, next_token_tensor], dim=1)
                        if self.debug:
                            logging.info(f"New input shape after cat: {current_input.shape}")
                    except Exception as e:
                        logging.error(f"Concatenation error: {str(e)}")
                        logging.error(f"Current input: {current_input.shape}, Next token: {next_token_tensor.shape}")
                        raise e
            
            # Decode the generated tokens
            try:
                response = self.tokenizer.decode(generated_tokens)
                logging.info(f"Generated response: {response}")
                return response.strip() if response.strip() else "Omlouvám se, ale nemohu vygenerovat smysluplnou odpověď."
            except Exception as e:
                logging.error(f"Decoding error: {str(e)}")
                return "Omlouvám se, ale nastala chyba při generování odpovědi."
                
        except Exception as e:
            logging.error(f"Generation error: {str(e)}", exc_info=True)
            return "Omlouvám se, ale nastala chyba."

def chat():
    try:
        ai = AIChat()
        print("AI je připraven k chatu! (Pro ukončení napište 'konec')")
        
        while True:
            user_input = input("\nVy: ")
            if user_input.lower() == 'konec':
                break
                
            response = ai.generate_response(user_input)
            print(f"\nAI: {response}")
    except Exception as e:
        print(f"Kritická chyba: {str(e)}")
        logging.error("Kritická chyba", exc_info=True)

if __name__ == "__main__":
    chat()
