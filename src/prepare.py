import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def save_data(df, path):
    df.to_csv(path, index=False)

def main(args):
    df = load_data(args.input_path)
    
    # Stratified split to maintain class distribution for 'stroke'
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df['stroke']
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, 'train.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')
    
    save_data(train_df, train_path)
    save_data(test_df, test_path)
    
    print(f"Data split completed. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("input_path", type=str, help="Path to raw dataset")
    parser.add_argument("output_dir", type=str, help="Directory to save prepared data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of dataset to include in the test split")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    
    args = parser.parse_args()
    main(args)
