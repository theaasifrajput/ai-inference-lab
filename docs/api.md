# API

The inference service exposes an HTTP API using FastAPI.

## Start the Server

From the repository root:

```bash
uvicorn services.gateway.app:app --host 0.0.0.0 --port 8000

The server will be available at:

http://localhost:8000

Health Check
GET /api/v1/health

Example response:

{
  "status": "healthy"
}

This endpoint confirms that the service is running.

Readiness Check
GET /api/v1/ready

Example response:

{
  "status": "ready",
  "worker_running": true
}

This endpoint confirms that the inference worker is running and ready to process requests.

Submit an Inference Request
POST /api/v1/inference/image

The endpoint accepts an image upload.

Supported image formats:

JPEG
PNG
WebP

Example using curl:

curl -X POST \
  -F "file=@image.jpg" \
  http://localhost:8000/api/v1/inference/image

  A successful request returns HTTP 202:

{
  "request_id": "<request-id>",
  "status": "QUEUED",
  "status_url": "/api/v1/inference/<request-id>"
}

The request is asynchronous. The client receives a request ID and can use it to retrieve the result.

Get Inference Result
GET /api/v1/inference/{request_id}

Example:

curl http://localhost:8000/api/v1/inference/<request-id>

The response contains:

Request status
Queue latency
Processing latency
Model execution latency
Detection results
Runtime information
Execution provider
Device information

A completed request has a status similar to:
{
  "request_id": "<request-id>",
  "status": "COMPLETED",
  "queue_latency_ms": 1.2,
  "processing_time_ms": 25.4,
  "model_execution_ms": 20.1,
  "detections": []
}

Request Lifecycle
Client
  ↓
POST /inference/image
  ↓
Validate image
  ↓
Create request ID
  ↓
Add request to queue
  ↓
Return HTTP 202
  ↓
Inference Worker
  ↓
Process image
  ↓
Store result
  ↓
Client polls request ID
  ↓
Return result

Queue

The service uses a bounded queue to prevent unlimited requests from accumulating in memory.

The queue size can be configured using:

MAX_QUEUE_SIZE

When the queue is full, the service returns:

HTTP 503

with a message indicating that the inference queue is full.

Upload Limits

The maximum upload size can be configured using:

MAX_UPLOAD_MB

The default limit is:

20 MB

Queue

The service uses a bounded queue to prevent unlimited requests from accumulating in memory.

The queue size can be configured using:

MAX_QUEUE_SIZE

When the queue is full, the service returns:

HTTP 503

with a message indicating that the inference queue is full.

Upload Limits

The maximum upload size can be configured using:

MAX_UPLOAD_MB

The default limit is:

20 MB

202  Request accepted
404  Unknown request ID
413  Upload too large
415  Unsupported image type
422  Invalid image
503  Inference queue unavailable/full