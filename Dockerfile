FROM python:3.11

# Optional build-time proxy configuration. Only applied if you pass these as
# --build-arg when building (e.g. from inside a network that requires one);
# harmless to leave unset otherwise.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}

WORKDIR /workdir

# --- Radiomic_Feature_Extraction dependencies (PyRadiomics) -------------------------
RUN python -m pip install --upgrade pip
RUN python -m pip install numpy SimpleITK
# --no-build-isolation: pyradiomics' own build process needs numpy already
# installed (see above) to build itself; pip's normal isolated build
# environment doesn't include it, which makes the build fail otherwise.
RUN python -m pip install pyradiomics==3.0.1 --no-build-isolation
COPY Radiomic_Feature_Extraction Radiomic_Feature_Extraction
COPY Radiomic_Feature_Extraction/code/imageoperations.py /usr/local/lib/python3.11/site-packages/radiomics/imageoperations.py

# --- matto_radiomics (the shared pipeline as one installable package) --------
# 0_Packaging maps the shared code as matto_radiomics.<name>. See
# 0_Packaging/setup.cfg.
COPY Radiomics_Model_Development Radiomics_Model_Development
COPY MATTO-GBM_Models MATTO-GBM_Models
COPY 0_Common 0_Common
COPY 0_Packaging 0_Packaging
RUN pip3 install --no-cache-dir -e 0_Packaging

# No fixed ENTRYPOINT/CMD: this single image is shared by the pipeline steps
# (Radiomic_Feature_Extraction, Radiomics_Model_Development and
# MATTO-GBM_Models). Each step's README documents the exact command.

