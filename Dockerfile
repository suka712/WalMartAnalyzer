# Use a specific Python 3.11 slim image for reproducibility
FROM python:3.11.9-slim

# Install system dependencies required by lightgbm
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Make the start script executable
RUN chmod +x /app/start.sh

# Expose the ports for the backend (5000) and frontend (8501)
EXPOSE 5000
EXPOSE 8501

# Set the command to run when the container starts
CMD ["/app/start.sh"]