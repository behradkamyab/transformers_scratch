import os
from datasets import load_dataset

def download_wikitext_from_hf(data_dir="../data/raw"):
    """
    Downloads WikiText-2 raw using the Hugging Face datasets library
    and consolidates it into data/raw/input.txt.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    final_output = os.path.join(data_dir, "input.txt")
    
    if os.path.exists(final_output):
        print(f"Dataset already exists at {final_output}")
        return

    print("Loading wikitext-2-raw-v1 from Hugging Face...")
    # Load the raw wikitext-2 dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    
    print("Consolidating splits into input.txt...")
    with open(final_output, 'w', encoding='utf-8') as outfile:
        for split in ['train', 'validation', 'test']:
            text_data = dataset[split]['text']
            # Join lines back together with newlines
            outfile.write("\n".join(text_data))
            
    print(f"Dataset ready at {final_output}. File size: {os.path.getsize(final_output) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    download_wikitext_from_hf()