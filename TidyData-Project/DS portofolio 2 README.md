# Tidy Data Project: 2008 Olympics Medalists

## Project Overview
The goal of this project is to apply **Tidy Data Principles** to clean and transform a messy dataset, followed by exploratory data analysis. In the original dataset, variables like "Gender" and "Sport" were trapped in column headers (e.g., `male_archery`). By reshaping the data into a long format, we ensured that:
1. Each variable is in its own column (Medalist, Gender, Sport, Medal).
2. Each observation forms its own row.
3. Each type of observational unit forms its own table.

## Instructions
To run this project locally:

1. Ensure you have Python installed.
2. Set up a virtual environment and install the required dependencies:
   ```bash
   pip install pandas matplotlib seaborn jupyter
   ```
3. Open the Olympics_Data_Cleaning.ipynb file using Jupyter Notebook or VS Code.

4.Run the cells sequentially from top to bottom.

## Dataset Description
The dataset olympics_08_medalists.csv contains information on athletes who won medals during the 2008 Olympic Games.

Pre-processing steps: The dataset was originally in a wide format. I used pd.melt() to collapse the 70+ sport columns into a single column. I then used str.split() to separate gender and sport, and str.replace() to clean up the sport names (removing underscores and standardizing capitalization). Finally, rows containing NaN for medals were dropped, as they did not represent actual observations of a medal won.

## References
Pandas Cheat Sheet

Tidy Data Principles Paper by Hadley Wickham

## Visual Examples
Once you run the notebook, you will generate visual outputs such as:

A sorted Pivot Table showing the top sports by medal counts.

A Bar Chart visualizing the Top 10 sports by total medal volume.

A Count Plot illustrating the distribution of medals between male and female athletes.
