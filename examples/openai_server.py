#!/usr/bin/env python3
"""OpenAI-compatible API server example with SwiftLLM.

This example shows how to start a server that's compatible with
the OpenAI API, allowing you to use SwiftLLM as a drop-in replacement.
"""

import os
import sys


def start_server():
    """Start the OpenAI-compatible server."""
    # This would typically be done via CLI:
    # swiftllm serve --model meta-llama/Llama-2-7b-hf --port 8000

    print("Starting SwiftLLM OpenAI-compatible server...")
    print()
    print("Usage:")
    print("  swiftllm serve --model <model_path> --port 8000")
    print()
    print("Example:")
    print("  swiftllm serve --model meta-llama/Llama-2-7b-hf --port 8000")
    print()
    print("The server will expose:")
    print("  - POST /v1/chat/completions")
    print("  - POST /v1/completions")
    print("  - GET  /v1/models")
    print("  - GET  /health")
    print()
    print("You can then use the OpenAI Python client:")
    print()
    print("  from openai import OpenAI")
    print("  client = OpenAI(base_url='http://localhost:8000/v1', api_key='not-needed')")
    print("  response = client.chat.completions.create(")
    print("      model='llama-2-7b',")
    print("      messages=[{'role': 'user', 'content': 'Hello!'}]")
    print("  )")
    print()


def client_example():
    """Example of using the OpenAI client with SwiftLLM server."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai package: pip install openai")
        return

    # Configure client to use local SwiftLLM server
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # SwiftLLM doesn't require API key by default
    )

    # Example 1: Chat completion
    print("=" * 60)
    print("Chat Completion")
    print("=" * 60)

    response = client.chat.completions.create(
        model="llama-2-7b",  # Model name from server
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        temperature=0.7,
        max_tokens=100,
    )

    print(f"Response: {response.choices[0].message.content}")
    print()

    # Example 2: Streaming chat completion
    print("=" * 60)
    print("Streaming Chat Completion")
    print("=" * 60)

    stream = client.chat.completions.create(
        model="llama-2-7b",
        messages=[
            {"role": "user", "content": "Write a short poem about coding:"},
        ],
        temperature=0.8,
        max_tokens=150,
        stream=True,
    )

    print("Response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")

    # Example 3: Text completion (non-chat)
    print("=" * 60)
    print("Text Completion")
    print("=" * 60)

    response = client.completions.create(
        model="llama-2-7b",
        prompt="The quick brown fox",
        max_tokens=50,
        temperature=0.5,
    )

    print(f"Completion: {response.choices[0].text}")
    print()

    # Example 4: List models
    print("=" * 60)
    print("List Models")
    print("=" * 60)

    models = client.models.list()
    for model in models.data:
        print(f"  - {model.id}")


def curl_examples():
    """Print curl examples for using the API."""
    print("\n" + "=" * 60)
    print("cURL Examples")
    print("=" * 60)

    print("""
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "llama-2-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# Streaming chat completion
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "llama-2-7b",
    "messages": [{"role": "user", "content": "Write a story:"}],
    "stream": true
  }'

# Text completion
curl http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "llama-2-7b",
    "prompt": "The meaning of life is",
    "max_tokens": 50
  }'
""")


def main():
    """Run server examples."""
    if len(sys.argv) > 1 and sys.argv[1] == "--client":
        # Run client examples (assumes server is running)
        client_example()
    else:
        # Show server usage
        start_server()
        curl_examples()


if __name__ == "__main__":
    main()
