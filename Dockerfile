FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy entire Quad_Sim folder
COPY Quadcopter_SimCon /app/Quadcopter_SimCon

COPY Quadcopter_SimCon/Simulation/output.xlsx /app/Quadcopter_SimCon/Simulation/output.xlsx

# Install dependencies
RUN pip install -r /app/Quadcopter_SimCon/requirements.txt

# Command to run your Python script
CMD ["python", "/app/Quadcopter_SimCon/Simulation/run_3D_simulation.py"]
