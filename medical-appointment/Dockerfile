FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Your model and the server. The sample data under data/ is not copied: only
# local_evaluator.py needs it, and solution/ needs configs/ (base + profile
# YAML) but not the calibrated output that's generated locally.
COPY dtos.py utils.py example.py api.py ./
COPY solution/ ./solution/
COPY configs/ ./configs/

# Selects which configs/<profile>.yaml overlays base.yaml. dev_8gb is the
# safe default; override at `docker run -e MEDICAL_APPT_PROFILE=workstation_32gb`.
ENV MEDICAL_APPT_PROFILE=dev_8gb

EXPOSE 9054

CMD ["python", "api.py"]
