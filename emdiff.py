import argparse
import os
import docx
import numpy as np
import logging
import torch
import multiprocessing
import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# Function to read .txt and .docx files and extract paragraphs with source file information
def read_files(folder_path):
    paragraphs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.txt'):
            LOGGER.info(f'Reading {file_path}')
            with open(file_path, 'r', encoding='utf-8') as file:
                paragraphs.extend({'text': p.strip(), 'source': filename} for p in file.read().split('\n') if p.strip() and len(p.strip().split()) >= 6)
        elif filename.endswith('.docx'):
            LOGGER.info(f'Reading {file_path}')
            doc = docx.Document(file_path)
            paragraphs.extend({'text': p.text.strip(), 'source': filename} for p in doc.paragraphs if p.text.strip() and len(p.text.strip().split()) >= 6)
    LOGGER.info(f'Extracted {len(paragraphs)} paragraphs from {folder_path}')
    return paragraphs

# Function to compare a single paragraph embedding with all others
def compare_embeddings(idx1, emb1, embeddings_2_np, paragraphs_1, paragraphs_2, threshold):
    similarities = cosine_similarity([emb1], embeddings_2_np)[0]
    matches = []
    for idx2, similarity_score in enumerate(similarities):
        if similarity_score > threshold:
            matches.append({
                'folder1_paragraph': paragraphs_1[idx1]['text'],
                'folder1_source': paragraphs_1[idx1]['source'],
                'folder2_paragraph': paragraphs_2[idx2]['text'],
                'folder2_source': paragraphs_2[idx2]['source'],
                'similarity': similarity_score
            })
    return matches

# Global model loading to avoid reloading in child processes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def emdiff(folder_1, folder_2, model_name, similarity_threshold, output_csv_path, output_summary_csv_path):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Determine if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOGGER.info(f'Using device: {device}')

    # Load pre-trained model for generating embeddings
    LOGGER.info('Loading pre-trained model for generating embeddings...')
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    LOGGER.info('Model loaded successfully.')
    model.max_seq_length = 32768
    model.tokenizer.padding_side="right"
    
    def add_eos(input_examples):
        input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
        return input_examples

    # Read paragraphs from both folders
    LOGGER.info('Reading paragraphs from folder 1...')
    paragraphs_1 = read_files(folder_1)
    LOGGER.info('Reading paragraphs from folder 2...')
    paragraphs_2 = read_files(folder_2)

    # Generate embeddings for each paragraph
    LOGGER.info('Generating embeddings for paragraphs from folder 1...')
    batch_size = 2
    embeddings_1 = model.encode(add_eos([p['text'] for p in paragraphs_1]), batch_size=batch_size, convert_to_tensor=True, device=device, normalize_embeddings=False)
    LOGGER.info('Generating embeddings for paragraphs from folder 2...')
    embeddings_2 = model.encode(add_eos([p['text'] for p in paragraphs_2]), batch_size=batch_size, convert_to_tensor=True, device=device, normalize_embeddings=False)

    # Convert embeddings to numpy arrays for similarity calculation
    LOGGER.info('Converting embeddings to numpy arrays for similarity calculation...')
    embeddings_1_np = embeddings_1.cpu().numpy()
    embeddings_2_np = embeddings_2.cpu().numpy()

    # Compare embeddings and identify similar paragraphs using multiprocessing
    LOGGER.info('Comparing embeddings and identifying similar paragraphs using multiprocessing...')
    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            compare_embeddings,
            [(idx1, emb1, embeddings_2_np, paragraphs_1, paragraphs_2, similarity_threshold) for idx1, emb1 in enumerate(embeddings_1_np)]
        )

    # Flatten the list of results
    similar_paragraphs = [match for sublist in results for match in sublist]

    # Sort similar paragraphs by similarity score in descending order
    similar_paragraphs = sorted(similar_paragraphs, key=lambda x: x['similarity'], reverse=True)

    LOGGER.info(f'Found {len(similar_paragraphs)} similar paragraphs exceeding the similarity threshold.')

    # Save results to a CSV file
    LOGGER.info(f'Saving similar paragraphs to {output_csv_path}...')
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['FILENAME 1', 'FILENAME 2', 'SCORE', 'TEXT 1', 'TEXT 2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for match in similar_paragraphs:
            writer.writerow({
                'FILENAME 1': match['folder1_source'],
                'FILENAME 2': match['folder2_source'],
                'SCORE': f"{match['similarity']:.2f}",
                'TEXT 1': match['folder1_paragraph'],
                'TEXT 2': match['folder2_paragraph']
            })
    LOGGER.info('Similar paragraphs saved successfully in CSV format.')

    # Group results by source files and compute statistics
    LOGGER.info('Calculating file-level similarity statistics...')
    df = pd.DataFrame(similar_paragraphs)
    if not df.empty:
        df['pair'] = df.apply(lambda row: (row['folder1_source'], row['folder2_source']), axis=1)
        grouped = df.groupby('pair')['similarity']
        summary = grouped.agg(['mean', 'max', lambda x: np.percentile(x, 90)])
        summary.columns = ['average_score', 'max_score', '90_percentile']
        summary = summary.reset_index()
        # Filter pairs with average similarity score greater than similarity threshold
        filtered_summary = summary[summary['average_score'] > similarity_threshold]

        # Save summary to CSV file
        LOGGER.info(f'Saving file similarity summary to {output_summary_csv_path}...')
        filtered_summary.to_csv(output_summary_csv_path, index=False)
        LOGGER.info('File similarity summary saved successfully.')
    else:
        LOGGER.info('No similar paragraphs found to generate file-level summary.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare document similarity between two folders.')
    parser.add_argument('--folder1', type=str, required=True, help='Path to the first folder containing documents')
    parser.add_argument('--folder2', type=str, required=True, help='Path to the second folder containing documents')
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2', help='Pre-trained model for generating embeddings')
    parser.add_argument('--threshold', type=float, default=0.8, help='Similarity threshold for considering two paragraphs as similar')
    parser.add_argument('--output_csv', type=str, default='similar_paragraphs.csv', help='Path to output CSV file for similar paragraphs')
    parser.add_argument('--output_summary_csv', type=str, default='file_similarity_summary.csv', help='Path to output CSV file for file similarity summary')
    args = parser.parse_args()

    emdiff(args.folder1, args.folder2, args.model, args.threshold, args.output_csv, args.output_summary_csv)
