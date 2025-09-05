# Use an official NVIDIA PyTorch image as a base
# Choose a version compatible with your CUDA/cuDNN setup
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel

# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary application files
# .dockerignore will prevent copying the large dataset
COPY . /app

# Install Python dependencies
# It's good practice to use a requirements.txt
# For now, I'll list them directly
RUN pip install --no-cache-dir \
    Pillow \
    numpy \
    tqdm \
    # Add any other specific versions if needed, e.g., torch==2.0.1 torchvision==0.15.2

# Set environment variables (optional, but good practice)
ENV PYTHONUNBUFFERED=1

# Command to run the training script (this can be overridden at runtime)
# ENTRYPOINT ["python3", "AI/train.py"]
# Or just define the default command
CMD ["python3", "AI/train.py"]
