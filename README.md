# LLM Batched Inference Toolkit

vLLM Online Benchmark CLI is more than a benchmarking utility — it also functions as a high-throughput batched inference toolkit for large language models. Locally, it maximizes GPU utilization by batching and scheduling multiple prompts to increase throughput and reduce per-request overhead. It can additionally translate batched inference requests into formats supported by major cloud providers' batch APIs, making it straightforward to run the same workloads on managed batch services.
