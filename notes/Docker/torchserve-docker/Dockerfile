# Stage 0: Fetch Model Files
FROM alpine/git AS fetcher
RUN apk add --no-cache git
RUN git clone https://huggingface.co/bert-base-uncased /model

# Stage 1: Build Stage
FROM pytorch/torchserve:latest AS build
RUN pip install transformers
RUN mkdir -p /home/model-server/model-store /home/model-server/code
COPY requirements.txt /home/model-server/
RUN pip install -r /home/model-server/requirements.txt
COPY --from=fetcher /model /home/model-server/model-store/bert
COPY handler.py /home/model-server/code/handler.py
# Archive the model
RUN torch-model-archiver \
    --model-name bert_seq_class \
    --version 1.0 \
    --serialized-file /home/model-server/model-store/bert/pytorch_model.bin \
    --handler /home/model-server/code/handler.py \
    --extra-files "/home/model-server/model-store/bert/config.json,/home/model-server/model-store/bert/tokenizer.json,/home/model-server/model-store/bert/tokenizer_config.json,/home/model-server/model-store/bert/vocab.txt" \
    --export-path /home/model-server/model-store

# Stage 2: Runtime Stage
FROM pytorch/torchserve:latest-gpu
COPY requirements.txt /home/model-server/
RUN pip install -r /home/model-server/requirements.txt
# Copy model archive and configuration from the build stage
COPY --from=build /home/model-server/model-store /home/model-server/model-store
EXPOSE 8080
EXPOSE 8081
# Start TorchServe
CMD ["torchserve", "--start", "--model-store", "/home/model-server/model-store", "--models", "bert_seq_class=bert_seq_class.mar"]
