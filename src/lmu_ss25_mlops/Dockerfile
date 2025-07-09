FROM train:latest

# Copy the updated training script into the image
COPY train.py /src/lmu_ss25_mlops/train.py

# Set the default command to start training
ENTRYPOINT ["python", "-u", "src/lmu_ss25_mlops/train.py"]
