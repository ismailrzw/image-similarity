# image-similarity-search

## Streamlit Image Similarity Search

This is an interactive web application built with **Streamlit** that allows users to perform image similarity searches. It's a great way to find visually similar images within a given dataset by uploading a query image.

-----

## 🚀 Features

  * **Image Upload**: Easily upload a new image to use as your search query.
  * **Similarity Search**: The application finds and displays the most similar images from a pre-defined dataset.
  * **Interactive UI**: A simple and intuitive user interface built with the Streamlit framework.

-----

## 🛠️ Installation and Setup

To run this application on your local machine, follow these steps.

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app**:

    ```bash
    streamlit run <your-app-file-name>.py
    ```

-----

## 🧠 How It Works

The application works by first processing a dataset of images and creating numerical representations, or **embeddings**, for each one. When a new image is uploaded, it generates an embedding for that query image. It then uses a similarity metric (e.g., **cosine similarity**) to compare it against all the pre-computed embeddings. The most similar images are then displayed to the user.

