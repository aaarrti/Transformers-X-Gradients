# Build base image
apptainer build --fakeroot --nv container.sif container.def
# Create image for venv
mkfs.ext3 -E root_owner venv.img 10G
# Create venv inside of image
apptainer exec --nv -B venv.img:/venv:image-src=/ container.sif /usr/bin/python3 -m venv /venv --system-site-packages
# Install dependencies into image
apptainer exec --nv -B venv.img:/venv:image-src=/ container.sif /venv/bin/python -m pip install transformers \
  tokenizers safetensors tensorflow_probability pytest datasets
# Run shell in container with venv bounded at root
apptainer shell --nv -B venv.img:/venv:image-src=/,ro container.sif
