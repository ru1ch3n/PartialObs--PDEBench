FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:030add5351585b43fb46a1939aad7c82dc11c7e690eaaaf118e8b71e3b674519

COPY . $SRC/pdeobs
WORKDIR $SRC/pdeobs

RUN python3 -m pip install --disable-pip-version-check --no-cache-dir \
    --require-hashes -r .clusterfuzzlite/requirements.txt

COPY .clusterfuzzlite/build.sh $SRC/build.sh
