# Using python version 3.9.7 as base image  
FROM python:3.9.7 

# Mind you, set the work directory in the container to /EASYEDIT 
WORKDIR /EASYEDIT 

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install pip then upgrade it
RUN /usr/local/bin/python3 -m pip install --upgrade pip 

# Once pip is upgraded, install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# This instruction informs Docker that the container listens on the specified network ports at runtime
EXPOSE 8080 

# Copy the content of the local src directory to the working directory in the container
COPY edit.py . 

# Set environment variable
ENV FLASK_APP=edit.py 

# This command will be executed when docker container starts up 
CMD ["flask", "run", "--host", "0.0.0.0"]