from huggingface_hub import snapshot_download

# this will download into ./models/mistral-7b-instruct-v0.3
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir="models/mistral-7b-instruct-v0.3",
    use_auth_token=True,
)
