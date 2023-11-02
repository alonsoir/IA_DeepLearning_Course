FROM python:3.11-alpine

COPY Dimensions.bin /app/

ENTRYPOINT ["/app/Dimensions.bin"]