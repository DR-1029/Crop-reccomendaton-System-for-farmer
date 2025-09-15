

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_PATH   = "Crop_recommendation.csv"   # update if your file lives elsewhere
MODEL_PATH  = "crop_model.pkl"
TEST_SIZE   = 0.20
RANDOM_SEED = 42
N_ESTIMATORS = 200

# ──────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("\n🛈 First five rows ↓")
    print(df.head())
    print("\n🛈 Column names & dtypes ↓")
    print(df.dtypes)
    # Remove rows without a label (if any)
    df = df.dropna(subset=['label'])
    return df

# ──────────────────────────────────────────────────────────────────────────
def show_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.drop(columns=['label'])
    corr = numeric_df.corr(numeric_only=True)
    heatmap_path = os.path.join("static", "heatmap.png")
    sns.heatmap(corr, annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap of Soil Features")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=120)
    plt.close()
    print("📊 Heat‑map saved to", heatmap_path)




# ──────────────────────────────────────────────────────────────────────────
def train_random_forest(df: pd.DataFrame) -> float:
    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy on held‑out test set: {acc:.3%}")

    # Uncomment for a full classification report
    # print("\nDetailed classification report:\n", classification_report(y_test, y_pred))

    # Save the trained model for later use (Flask/Streamlit)
    joblib.dump(clf, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
    return acc

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    show_heatmap(df)          # comment out if you don't need to plot every run
    train_random_forest(df)

import os
import seaborn as sns
import matplotlib.pyplot as plt

def generate_crop_plots(df):
    crops = df['label'].unique()
    os.makedirs("static/plots", exist_ok=True)

    for crop in crops:
        crop_df = df[df['label'] == crop]

        # 1️⃣ SCATTER PLOT: N vs K
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=crop_df, x='N', y='K')
        plt.title(f"{crop.title()} - N vs K")
        plt.tight_layout()
        plt.savefig(f'static/plots/active_scatter.png')
        plt.close()

        # 2️⃣ BAR CHART: average values
        plt.figure(figsize=(6, 4))
        crop_df.mean(numeric_only=True).plot(kind='bar', color='skyblue')
        plt.title(f"{crop.title()} - Average Feature Values")
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(f'static/plots/active_bar.png')
        plt.close()

        # 3️⃣ BOX PLOT: pH distribution
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=crop_df['ph'])
        plt.title(f"{crop.title()} - pH Distribution")
        plt.tight_layout()
        plt.savefig(f'static/plots/active_box.png')
        plt.close()

    print("✅ All crop plots saved to static/plots/")



