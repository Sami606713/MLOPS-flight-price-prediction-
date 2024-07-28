# Use a specific version of Python
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy only the requirements
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]
