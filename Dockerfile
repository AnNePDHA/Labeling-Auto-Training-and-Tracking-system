# Use an official Python runtime as a parent image
FROM python:3.10.5

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0
    
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip to the latest version
RUN /usr/local/bin/python -m pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# # Define environment variable
# ENV FLASK_APP=app.py

# # Run app.py when the container launches
# CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]

CMD [ "python", "app.py"]
