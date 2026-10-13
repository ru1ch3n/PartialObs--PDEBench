FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:21f6fab17ace4dc76e1c56de91ee95d3d03da021df94b41056ba036f6c6f55e9

COPY . $SRC/pdeobs
WORKDIR $SRC/pdeobs

RUN python3 -m pip install --disable-pip-version-check --no-cache-dir \
    --require-hashes -r .clusterfuzzlite/requirements.txt

COPY .clusterfuzzlite/build.sh $SRC/build.sh
