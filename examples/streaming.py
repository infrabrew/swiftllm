#!/usr/bin/env python3
"""Streaming inference example with SwiftLLM.

This example demonstrates how to stream tokens as they are generated,
useful for real-time applications like chatbots.
"""

import asyncio
from swiftllm import AsyncLLM, SamplingParams


async def stream_single_prompt():
    """Stream a single prompt response."""
    print("=" * 60)
    print("Streaming Single Prompt")
    print("=" * 60)

    llm = AsyncLLM(model="meta-llama/Llama-2-7b-hf")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
    )

    prompt = "Write a short story about a robot learning to paint:"

    print(f"Prompt: {prompt}\n")
    print("Response: ", end="", flush=True)

    async for output in llm.generate(prompt, sampling_params):
        # Print new tokens as they are generated
        if output.outputs:
            text = output.outputs[0].text
            # In a real implementation, you'd track the delta
            print(text, end="", flush=True)

    print("\n")


async def stream_multiple_prompts():
    """Stream multiple prompts concurrently."""
    print("=" * 60)
    print("Streaming Multiple Prompts Concurrently")
    print("=" * 60)

    llm = AsyncLLM(model="meta-llama/Llama-2-7b-hf")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
    )

    prompts = [
        "Count from 1 to 10:",
        "Name the planets in order:",
        "List primary colors:",
    ]

    async def stream_prompt(prompt: str, index: int):
        """Stream a single prompt and collect output."""
        result = []
        async for output in llm.generate(prompt, sampling_params):
            if output.outputs:
                result.append(output.outputs[0].text)
        return index, prompt, "".join(result) if result else ""

    # Run all prompts concurrently
    tasks = [stream_prompt(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    for index, prompt, response in sorted(results):
        print(f"\n--- Prompt {index + 1} ---")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")


async def stream_chat_conversation():
    """Simulate a streaming chat conversation."""
    print("=" * 60)
    print("Streaming Chat Conversation")
    print("=" * 60)

    llm = AsyncLLM(model="meta-llama/Llama-2-7b-hf")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=150,
    )

    # Build conversation history
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    user_messages = [
        "Hello! What's your name?",
        "Can you tell me about Python programming?",
        "What's the best way to learn it?",
    ]

    for user_msg in user_messages:
        conversation.append({"role": "user", "content": user_msg})

        # Format conversation as prompt
        prompt = ""
        for msg in conversation:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant:"

        print(f"\nUser: {user_msg}")
        print("Assistant: ", end="", flush=True)

        response_text = ""
        async for output in llm.generate(prompt, sampling_params):
            if output.outputs:
                response_text = output.outputs[0].text
                print(response_text, end="", flush=True)

        print()  # New line after response

        # Add assistant response to history
        conversation.append({"role": "assistant", "content": response_text.strip()})


async def stream_with_cancellation():
    """Demonstrate cancelling a stream mid-generation."""
    print("\n" + "=" * 60)
    print("Stream with Cancellation")
    print("=" * 60)

    llm = AsyncLLM(model="meta-llama/Llama-2-7b-hf")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=500,  # Long response
    )

    prompt = "Write a very long essay about the history of computing:"
    request_id = "cancel-demo-1"

    print(f"Prompt: {prompt}\n")
    print("Response (will cancel after 50 chars): ", end="", flush=True)

    char_count = 0
    async for output in llm.generate(prompt, sampling_params, request_id=request_id):
        if output.outputs:
            text = output.outputs[0].text
            print(text, end="", flush=True)
            char_count += len(text)

            if char_count > 50:
                print("\n\n[Cancelling...]")
                await llm.abort(request_id)
                break

    print("[Stream cancelled]")


async def main():
    """Run all streaming examples."""
    await stream_single_prompt()
    await stream_multiple_prompts()
    await stream_chat_conversation()
    await stream_with_cancellation()


if __name__ == "__main__":
    asyncio.run(main())
