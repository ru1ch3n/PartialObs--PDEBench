FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:6a90ec89a18436e057a06f54266c3f120eb87d96c095b0c67a2b15cab33ec1f4

COPY . $SRC/pdeobs
WORKDIR $SRC/pdeobs

RUN python3 -m pip install --disable-pip-version-check --no-cache-dir \
    --require-hashes -r .clusterfuzzlite/requirements.txt

COPY .clusterfuzzlite/build.sh $SRC/build.sh
