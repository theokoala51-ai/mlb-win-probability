# MLB Win Probability Table (2023-2025)

This project generates a smoothed Win Probability table for MLB games based on data from the 2023-2025 seasons.

It uses **Statcast data** (via `pybaseball`) and trains a **Logistic Regression** model to estimate the probability of the home team winning from any given game state.

## Features

- **Data Source**: Fetches pitch-by-pitch data for 2023, 2024, and 2025 using `pybaseball`.
- **Game State Modeling**:
  - **Inning**: 1-9+
  - **Top/Bot**: Top or Bottom of the inning
  - **Outs**: 0, 1, 2
  - **Base State**: Runners on 1st, 2nd, 3rd (Empty to Loaded)
  - **Run Differential**: Home Score - Away Score
- **Smoothing**: Uses Logistic Regression to provide probabilities for all states, even those with sparse data in the 3-year sample (e.g., highly specific extra-inning scenarios).
- **Output**:
  - `win_probability_table.csv`: Raw probability matrix.
  - `win_probability_table.html`: Heatmap with conditional formatting (Blue = Low Win Prob, Red = High Win Prob).

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Script**:
    ```bash
    python main.py
    ```

    *Note: The first run will take some time to download the Statcast data. Subsequent runs will use the cached CSV.*

## Methodology

We model `P(HomeWin)` as a function of the game state.

To ensure "smoothness" and handle the non-linear value of runs across innings (e.g., a 1-run lead is worth more in the 9th than the 1st), we train a **separate Logistic Regression model for each Inning**.

This captures the specific run environment and leverage of that inning while maintaining the smooth probability curve of the logistic function with respect to Run Differential.

## Output Format

The output table rows define the state:
- **Inning**
- **Top/Bot**
- **Outs**
- **BaseState**

The columns represent the **Run Differential** (Home - Away), ranging from -6 to +6.
The cells contain the probability (0% to 100%) that the Home Team wins.
