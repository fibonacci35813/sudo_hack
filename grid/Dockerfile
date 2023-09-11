# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the code and other necessary files into the container
COPY . /app

# Install dependencies (you may need to update this based on your actual requirements)
RUN pip install transformers torch pandas scikit-learn tqdm

# Expose any necessary ports (if applicable)
EXPOSE 80

# Define the command to run your script or Jupyter notebook
CMD ["python", "main.py"]
