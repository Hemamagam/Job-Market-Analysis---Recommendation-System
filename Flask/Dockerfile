# Use the official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5002

# Set the environment variable to tell Flask it’s running in a Docker container
ENV FLASK_RUN_HOST=127.0.0.1

# Command to run the Flask app
CMD ["flask", "run", "--port=5002", "--reload"]