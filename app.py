import streamlit as st
from PIL import Image
import pandas as pd
import os
import random
import torch
from torchvision import transforms as T

torch.manual_seed(42)

from utils import (
    get_model_paths_by_input_size,
    load_model,
    preprocess_image,
    predict,
    load_metrics,
    get_val_accuracy,
    count_trainable_params,
)

# Optional: list of class names for readable predictions
CLASS_NAMES = [
    "Football",
    "Basketball",
    "Tennis",
    "Running",
    "Swimming",
    "Cycling",
    "Volleyball",
]

# --- App Setup ---
st.set_page_config(page_title="Sports Classifier", layout="wide")
st.title("🏅 Sports Image Classification App")

# --- Page Switcher ---
view_choice = st.sidebar.radio(
    "🧭 Select View",
    [
        "Image Analysis",
        "Input Transform Showcase",
        "Optuna Analysis",
        "Classification Dashboard",
    ],
)

# --- Page 1: Classification Dashboard ---
if view_choice == "Classification Dashboard":
    # --- Sidebar: Input Size Selection ---
    st.sidebar.header("Configuration")
    input_size_choice = st.sidebar.selectbox(
        "Input Size & Source",
        [
            "128_scratch_random_cropping",
            "128_pretrained_random_cropping",
            "128_pretrained_center_cropping",
            "224_scratch_random_cropping",
            "224_pretrained_random_cropping",
            "224_pretrained_center_cropping",
            "224_pretrained_random_decay",
            "224_pretrained_center_decay",
        ],
    )
    input_dim = 128 if "128" in input_size_choice else 224

    # --- Sidebar: Model Selection ---
    model_paths = get_model_paths_by_input_size(input_size_choice)
    model_names = list(model_paths.keys())
    model_choice = st.sidebar.selectbox("Select a Model", model_names)
    model_dir = model_paths[model_choice]

    # --- Load Model + Metrics ---
    model = load_model(model_dir, input_size=input_dim)
    metrics = load_metrics(model_dir)

    # --- Display Trainable Parameters ---
    trainable_params = count_trainable_params(model)
    st.markdown(f"🧮 **Trainable Parameters**: `{trainable_params:,}`")

    # --- Toggle Performance Plots ---
    st.subheader(f"📈 Model Performance: {model_choice}")
    show_plots = st.toggle("📊 Show Performance Plots", value=False)

    if show_plots:
        plot_mode = st.radio(
            "Select plot view mode:",
            ["Static (PNG)", "Interactive (Live Chart)"],
            horizontal=True,
        )

        if plot_mode == "Static (PNG)":
            col1, col2 = st.columns(2)
            col1.image(f"{model_dir}/accuracy_plot.png", caption="Validation Accuracy")
            col2.image(f"{model_dir}/loss_plot.png", caption="Validation Loss")
        else:
            train_accs = metrics.get("train_accs", [])
            val_accs = metrics.get("val_accs", [])
            train_losses = metrics.get("train_losses", [])
            val_losses = metrics.get("val_losses", [])

            if train_accs and val_accs:
                df_acc = pd.DataFrame(
                    {"Train Accuracy": train_accs, "Validation Accuracy": val_accs}
                )
                st.line_chart(df_acc)

            if train_losses and val_losses:
                df_loss = pd.DataFrame(
                    {"Train Loss": train_losses, "Validation Loss": val_losses}
                )
                st.line_chart(df_loss)

    # --- Image Upload & Prediction ---
    st.subheader("🖼️ Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose a sports image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=False)

        if st.button("🎯 Predict"):
            st.info("Classifying...")
            input_tensor = preprocess_image(img, input_size=input_dim)
            pred_class = predict(model, input_tensor)
            # class_name = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else f"Class {pred_class}"
            st.success(f"🧠 Predicted Class: **{pred_class}**")

    # --- Model Ranking ---
    if st.sidebar.button("🏆 Rank Models by Val Accuracy"):
        st.subheader("Model Leaderboard (Highest Val Accuracy)")
        ranks = [(name, get_val_accuracy(path)) for name, path in model_paths.items()]
        ranks.sort(key=lambda x: x[1], reverse=True)
        for i, (name, acc) in enumerate(ranks, start=1):
            st.markdown(f"**{i}. {name}** — Accuracy: `{acc:.2f}`")

# --- Page 2: Input Transform Showcase ---
elif view_choice == "Input Transform Showcase":
    st.header("🔍 How the Model Sees Inputs")

    # 📝 Dataset Class Descriptions
    with st.expander("ℹ️ Dataset Class Differences"):
        st.markdown(
            """
        - **`ImageDataset1`**
            - Assumes the folder is already split into train/test sets.
            - No internal data splitting.
        - **`ImageDataset2`**
            - Splits the dataset into training and testing using a `split_ratio`.
            - Good for flat folder structures.
        """
        )

    # 📤 Upload Image
    uploaded_image = st.file_uploader(
        "Upload a sample image to visualize transformations",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=False)

        input_size = st.selectbox("Select Input Size", [128, 224])

        # Define training & testing transforms
        train_transform = T.Compose([T.RandomResizedCrop(input_size), T.ToTensor()])
        test_transform = T.Compose(
            [T.Resize(input_size), T.CenterCrop(input_size), T.ToTensor()]
        )

        # Apply transforms
        train_img = train_transform(image)
        test_img = test_transform(image)

        st.subheader(f"🧪 Transformed Views ({input_size}x{input_size})")
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                train_img.permute(1, 2, 0).numpy(),
                caption="Random Resized Crop Transform",
                use_column_width=True,
            )
        with col2:
            st.image(
                test_img.permute(1, 2, 0).numpy(),
                caption="Center Crop Transform",
                use_column_width=True,
            )

elif view_choice == "Image Analysis":
    st.header("🧬 Dataset Distribution & Analysis")

    # 1. Train/Test Side-by-Side Class Distribution
    with st.expander("📊 Class Distribution: Train vs Test (Side by Side)"):
        st.image(
            "logs_analysis/distribution_of_each_class.png",
            caption="Class Distribution (Train and Test)",
        )
        st.markdown(
            """
        - The training set contains the highest number of images for **Cricket**, followed by **Wrestling**, **Tennis**, and **Badminton**.
        - The classes **Swimming** and **Karate** have noticeably fewer samples in the training set.
        - The test set follows a similar distribution trend, with **Cricket**, **Wrestling**, and **Badminton** having the most images.
        - **Swimming** and **Karate** also have the lowest counts in the test set.
        """
        )

    # 2. Combined Train vs Test Bar Plot
    with st.expander("📊 Combined Train/Test Distribution per Class"):
        st.image(
            "logs_analysis/per_class_distribution_train_vs_test.png",
            caption="Combined Train vs Test Distribution",
        )
        st.markdown(
            """
        - The bar heights confirm that **Cricket** has the highest image count across both splits.
        - **Swimming** and **Karate** have the lowest total image counts.
        - Each class label is clearly annotated with the corresponding number of images for both train and test sets.
        """
        )

    # 3. Pixel Value Distributions
    with st.expander("🌈 Pixel Intensity Distribution per Class"):
        st.image(
            "logs_analysis/Pixel_distribution_per_class.png",
            caption="Mean Pixel Value Distributions",
        )
        st.markdown(
            """
        - Each class shows a distinct distribution of mean pixel intensities.
        - The distributions resemble **normal (bell-shaped)** curves centered between pixel values **90 and 130**.
        - **Swimming**, **Soccer**, and **Tennis** have slightly broader distributions, extending toward higher pixel values.
        - **Karate** and **Wrestling** exhibit narrower peaks, indicating more concentrated pixel intensity ranges.
        """
        )

elif view_choice == "Optuna Analysis":
    st.header("🔍 Optuna Hyperparameter Search Insights")

    st.subheader("📊 Frequency of Parameter Choices")

    with st.expander("🧪 Optimizer Frequency"):
        st.image("log_analysis_optuna/frequency/optimizer_frequency.png", caption="Count of Optimizer Usage")
        st.markdown("""
        - This plot shows how frequently each optimizer was selected during the Optuna trials.
        - Useful to understand if certain optimizers dominated the search space.
        """)

    with st.expander("🧪 Dataset Class Frequency"):
        st.image("log_analysis_optuna/frequency/dataset_class_frequency.png", caption="Count of Dataset Class Usage")
        st.markdown("""
        - This illustrates how often `ImageClass1` vs `ImageClass2` was chosen.
        - Helpful for debugging class-specific tuning performance.
        """)

    with st.expander("🧪 Learning Rate Frequency"):
        st.image("log_analysis_optuna/frequency/learning_rate_frequenct.png", caption="Learning Rate Selection Frequency")
        st.markdown("""
        - Distribution of common learning rates used across all trials.
        - In this case, `0.01` and `0.1` seem to be among the most used.
        """)

    with st.expander("🧪 Model Frequency"):
        st.image("log_analysis_optuna/frequency/models_frequency.png", caption="Model Selection Frequency")
        st.markdown("""
        - Highlights which architectures (ResNet50) were favored during tuning.
        - Indicates popularity or effectiveness trends in the search space.
        """)

    st.subheader("📈 Validation Score Trends")

    with st.expander("📉 Validation Accuracy Across Trials"):
        st.image("log_analysis_optuna/scores_across_trials/trials_scores_validation.png", caption="Trial Validation Accuracy")
        st.markdown("""
        - This plot shows the validation accuracy of each trial.
        - Helps detect convergence trends and best performing trial index.
        """)
