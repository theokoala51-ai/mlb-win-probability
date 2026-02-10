import pybaseball
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Configuration
START_DATE = '2023-03-30'
END_DATE = '2025-11-05'
CACHE_FILE = 'mlb_data_2023_2025.csv'
OUTPUT_CSV = 'win_probability_table.csv'
OUTPUT_HTML = 'win_probability_table.html'

def fetch_data():
    """Fetches Statcast data or loads from cache."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading data from {CACHE_FILE}...")
        return pd.read_csv(CACHE_FILE)
    
    print("Fetching data from Statcast (this may take a while)...")
    # Fetching in chunks is safer but pybaseball handles it well usually.
    # We fetch by year to be safe and merge.
    years = [2023, 2024, 2025]
    dfs = []
    for year in years:
        print(f"Fetching {year}...")
        try:
            # statcast(start_dt, end_dt)
            df = pybaseball.statcast(start_dt=f'{year}-03-01', end_dt=f'{year}-11-30')
            dfs.append(df)
        except Exception as e:
            print(f"Error fetching {year}: {e}")
            
    if not dfs:
        raise ValueError("No data fetched.")
        
    full_df = pd.concat(dfs)
    full_df.to_csv(CACHE_FILE, index=False)
    return full_df

def process_data(df):
    """Processes raw Statcast data into model-ready features."""
    print("Processing data...")
    
    # Sort by game and time
    df = df.sort_values(['game_pk', 'at_bat_number'])
    
    # Define Target: Did Home Team Win?
    # We need the final score of each game.
    game_results = df.groupby('game_pk').agg({
        'post_home_score': 'last',
        'post_away_score': 'last'
    }).reset_index()
    
    game_results['home_won'] = (game_results['post_home_score'] > game_results['post_away_score']).astype(int)
    
    # Merge result back to every pitch/play
    df = df.merge(game_results[['game_pk', 'home_won']], on='game_pk', how='left')
    
    # Feature Engineering
    # State: Inning, Top/Bot, Outs, BaseState, RunDiff
    
    # Run Differential (Home - Away)
    # Use 'home_score' and 'away_score' which are pre-play scores
    df['run_diff'] = df['home_score'] - df['away_score']
    
    # Base State
    # on_1b, on_2b, on_3b are player IDs or NaNs. Convert to boolean.
    df['on_1b_bool'] = df['on_1b'].notna().astype(int)
    df['on_2b_bool'] = df['on_2b'].notna().astype(int)
    df['on_3b_bool'] = df['on_3b'].notna().astype(int)
    
    # Create a readable base state string
    # 000, 100, 020, 003, 120, 103, 023, 123
    df['base_state'] = (
        df['on_1b_bool'].astype(str) + 
        df['on_2b_bool'].astype(str) + 
        df['on_3b_bool'].astype(str)
    )
    
    # Top/Bot
    # inning_topbot is 'Top' or 'Bot'
    df['is_top'] = (df['inning_topbot'] == 'Top').astype(int)
    
    # Inning
    # Clip innings at 10 for the model to group extras (or keep raw if useful)
    # Let's clip at 10 to ensure we have enough data for "late game" logic
    df['inning_clipped'] = df['inning'].clip(upper=10)
    
    # Outs
    df['outs'] = df['outs_when_up']
    
    # Run Differential
    # Clip run diff to [-6, 6] to keep categories manageable and avoid sparse tails
    df['run_diff_cat'] = df['run_diff'].clip(lower=-6, upper=6).astype(int).astype(str)
    
    # Select Columns for Model
    features = ['inning_clipped', 'is_top', 'outs', 'base_state', 'run_diff_cat']
    target = 'home_won'
    
    # Drop rows with missing values in key columns
    model_df = df[features + [target]].dropna()
    
    return model_df

def train_model(df):
    """Trains a Logistic Regression model for smoothing."""
    print("Training model...")
    
    models = {}
    # Train one model per inning (1-9, 10+)
    unique_innings = sorted(df['inning_clipped'].unique())
    
    for inn in unique_innings:
        print(f"  Training for Inning {inn}...")
        inning_data = df[df['inning_clipped'] == inn]
        
        # Sub-features: Top/Bot, Outs, BaseState, RunDiff (Categorical)
        X_sub = inning_data[['is_top', 'outs', 'base_state', 'run_diff_cat']]
        y_sub = inning_data['home_won']
        
        if len(y_sub) < 100:
            continue # Skip sparse innings if any
            
        # All features are now categorical!
        # This allows the model to learn the exact value of a Tie vs +1 vs -1 independently.
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', drop=None), ['is_top', 'outs', 'base_state', 'run_diff_cat'])
            ]
        )
        
        # Using a slightly stronger regularization (C=1.0 is default) to prevent overfitting on rare combos
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs', C=0.5))
        ])
        
        pipeline.fit(X_sub, y_sub)
        models[inn] = pipeline
        
    return models

def generate_probability_table(models):
    """Generates the full grid of probabilities."""
    print("Generating probability table...")
    
    # Define Grid
    innings = range(1, 10) # 1 to 9
    top_bot = [1, 0] # 1=Top, 0=Bot
    outs = [0, 1, 2]
    base_states = ['000', '100', '020', '003', '120', '103', '023', '123']
    run_diffs = range(-6, 7) # -6 to +6
    
    rows = []
    
    for inn in innings:
        model = models.get(inn)
        if not model:
            continue
            
        for tb in top_bot:
            for o in outs:
                for b in base_states:
                    row_data = {
                        'Inning': inn,
                        'Top/Bot': 'Top' if tb == 1 else 'Bot',
                        'Outs': o,
                        'BaseState': b,
                        'key': f"{inn}_{'Top' if tb==1 else 'Bot'}_{o}_{b}"
                    }
                    
                    # Predict for each RunDiff
                    batch_data = []
                    for rd in run_diffs:
                        batch_data.append({
                            'is_top': tb,
                            'outs': o,
                            'base_state': b,
                            'run_diff_cat': str(rd)
                        })
                    
                    batch_df = pd.DataFrame(batch_data)
                    probs = model.predict_proba(batch_df)[:, 1] # Probability of Home Win
                    
                    # Add probabilities to row
                    for rd, prob in zip(run_diffs, probs):
                        row_data[rd] = prob
                        
                    rows.append(row_data)
    
    results_df = pd.DataFrame(rows)
    
    # Formatting for CSV/Display
    # We want: Rows (Inning, Top/Bot, Outs, BaseState) vs Cols (RunDiff)
    # The dataframe is already structured with RunDiffs as columns.
    
    # Rename BaseStates to readable
    base_map = {
        '000': 'Empty',
        '100': '1B',
        '020': '2B',
        '003': '3B',
        '120': '1B-2B',
        '103': '1B-3B',
        '023': '2B-3B',
        '123': 'Loaded'
    }
    results_df['BaseState'] = results_df['BaseState'].map(base_map)
    
    # Reorder columns
    meta_cols = ['Inning', 'Top/Bot', 'Outs', 'BaseState']
    diff_cols = list(run_diffs)
    results_df = results_df[meta_cols + diff_cols]
    
    return results_df

def style_table(df):
    """Applies conditional formatting (Blue=Low, Red=High)."""
    # Select only numeric columns for styling
    numeric_cols = df.columns[4:]
    
    def color_gradient(val):
        # 0.0 -> Blue (0, 0, 255)
        # 0.5 -> White (255, 255, 255)
        # 1.0 -> Red (255, 0, 0)
        
        # Simple interpolation
        if val < 0.5:
            # Blue to White
            # val 0: Blue, val 0.5: White
            ratio = val * 2
            r = int(255 * ratio)
            g = int(255 * ratio)
            b = 255
        else:
            # White to Red
            # val 0.5: White, val 1: Red
            ratio = (val - 0.5) * 2
            r = 255
            g = int(255 * (1 - ratio))
            b = int(255 * (1 - ratio))
            
        return f'background-color: rgb({r}, {g}, {b}); color: black'

    return df.style.map(color_gradient, subset=numeric_cols).format("{:.1%}", subset=numeric_cols)

def main():
    # 1. Fetch
    raw_df = fetch_data()
    
    # 2. Process
    model_df = process_data(raw_df)
    
    # 3. Train
    models = train_model(model_df)
    
    # 4. Generate
    prob_table = generate_probability_table(models)
    
    # 5. Save CSV
    prob_table.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved table to {OUTPUT_CSV}")
    
    # 6. Save HTML with styling
    styled = style_table(prob_table)
    with open(OUTPUT_HTML, 'w') as f:
        f.write(styled.to_html())
    print(f"Saved heatmap to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
