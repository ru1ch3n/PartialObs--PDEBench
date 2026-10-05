FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:b02a06f9d3918428825b1ebfa2c3627b591a5924a83436b8aaec40c85fd15b16

COPY . $SRC/pdeobs
WORKDIR $SRC/pdeobs

RUN python3 -m pip install --disable-pip-version-check --no-cache-dir \
    --require-hashes -r .clusterfuzzlite/requirements.txt

COPY .clusterfuzzlite/build.sh $SRC/build.sh
