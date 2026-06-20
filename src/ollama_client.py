#ollama_client.py

import openai
import backoff
import json
import urllib.request
from openai import OpenAI, OpenAIError, RateLimitError
import parameters


def _is_reasoning_model(model_name):
    model_name = str(model_name or "").lower()
    return "deepseek-r1" in model_name or "reasoning" in model_name

class OllamaClient:
    def __init__(self, model_name, base_url="http://localhost:11434/v1"):
        """
        Initialize the OllamaClient to use a local Ollama instance.
        
        Parameters:
        - model_name (str): The name of the model in Ollama (e.g., "llama3.1", "mistral").
        - base_url (str): The local endpoint for Ollama's OpenAI-compatible API.
        """
        # We use a dummy API key because Ollama doesn't require one for local runs
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=float(parameters.OLLAMA_REQUEST_TIMEOUT_SECONDS)
        )
        self.model_name = model_name
        self.deployment_name = model_name
        self.total_cost = 0.0  # Cost is always 0 for local Ollama

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def send_request(self, model_name, prompt, max_tokens=768, temperature=0.7, top_p=1.0, response_format=None, **kwargs):
        """
        Send a prompt to the local Ollama instance.
        """
        try:
            if _is_reasoning_model(self.model_name):
                return self._send_request_via_http(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    require_json=bool(response_format),
                )

            messages = [{"role": "user", "content": prompt}]
            
            import parameters
            create_args = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": 1,
                "seed": parameters.SEED,
            }
            if response_format and not _is_reasoning_model(self.model_name):
                create_args["response_format"] = response_format
                
            create_args.update(kwargs)

            # Call the ChatCompletion endpoint
            response = self.client.chat.completions.create(**create_args)

            message = response.choices[0].message
            generated_text = (getattr(message, "content", None) or "").strip()

            # Some reasoning-model backends can return empty content when JSON mode
            # is not supported or when the reasoning trace is exposed separately.
            if not generated_text:
                reasoning_text = getattr(message, "reasoning_content", None) or ""
                generated_text = reasoning_text.strip()

            return generated_text

        except OpenAIError as e:
            raise Exception(f"Ollama Error: {str(e)}")

    def _send_request_via_cli(self, prompt):
        """
        Use the ollama CLI for reasoning models so the full response,
        including <think> traces, is preserved reliably.
        """
        command = ["ollama", "run", self.model_name, prompt]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=float(parameters.OLLAMA_REQUEST_TIMEOUT_SECONDS),
            check=False,
        )

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            raise Exception(
                f"Ollama CLI error (code {completed.returncode}). STDERR: {stderr[:500]} STDOUT: {stdout[:500]}"
            )

        return (completed.stdout or "").strip()

    def _send_request_via_http(self, prompt, max_tokens=768, temperature=0.7, top_p=1.0, require_json=False):
        """
        Use Ollama's native HTTP API for reasoning models.
        This avoids the OpenAI compatibility layer while preserving any
        reasoning trace fields exposed by the backend.
        """
        url = parameters.LLM_BASE_URL.rstrip("/").replace("/v1", "") + "/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "seed": parameters.SEED,
                "num_predict": min(max_tokens, 768),
            },
        }
        # DeepSeek reasoning models can suppress message.content when JSON mode is
        # forced. We rely on the prompt contract plus downstream parsing instead.

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=float(parameters.OLLAMA_REQUEST_TIMEOUT_SECONDS)) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        message = response_data.get("message", {}) if isinstance(response_data, dict) else {}
        content = (message.get("content") or response_data.get("response") or "").strip() if isinstance(response_data, dict) else ""
        reasoning = (
            message.get("thinking")
            or message.get("reasoning")
            or message.get("reasoning_content")
            or response_data.get("thinking")
            or response_data.get("reasoning")
            or response_data.get("reasoning_content")
            or ""
        ) if isinstance(response_data, dict) else ""

        reasoning = str(reasoning).strip()
        content = str(content).strip()

        if reasoning and content:
            return f"<think>\n{reasoning}\n</think>\n{content}"
        if reasoning:
            return f"<think>\n{reasoning}\n</think>"
        return content

    def get_total_cost(self):
        """
        Return the total cost (always 0.0 for local runs).
        """
        return 0.0
