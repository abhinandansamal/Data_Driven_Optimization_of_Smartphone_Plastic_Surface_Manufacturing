# Dockerfile

# 1. Use an official Python runtime as a base image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependencies file into the container
# This is done first to leverage Docker's layer caching.
# The dependencies will only be re-installed if this file changes.
COPY requirements.txt .

# 4. Install the required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application files and folders into the container
# This single command copies everything (app.py, streamlit_app.py, src/, models/, etc.)
# from your project root into the container's /app/ directory.
COPY . .

# 6. Expose the port that the Flask app will run on
EXPOSE 5000

# 7. Define the command to run the Flask application when the container starts
CMD ["python", "app.py"]