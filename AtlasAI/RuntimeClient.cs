using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace AtlasAI
{
    /// <summary>
    /// Client for communicating with the Python runtime API.
    /// </summary>
    public class RuntimeClient : IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly AppConfiguration _config;
        private bool _disposed;

        public RuntimeClient(AppConfiguration config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _httpClient = new HttpClient
            {
                BaseAddress = new Uri(_config.RuntimeBaseUrl),
                Timeout = TimeSpan.FromSeconds(_config.RequestTimeoutSeconds)
            };
        }

        /// <summary>
        /// Waits for the runtime to become healthy.
        /// </summary>
        public async Task<bool> WaitForHealthyAsync(CancellationToken cancellationToken = default)
        {
            Console.WriteLine("Waiting for Python runtime to be ready...");

            for (int i = 0; i < _config.MaxHealthCheckAttempts; i++)
            {
                try
                {
                    var response = await _httpClient.GetAsync("/health", cancellationToken);
                    
                    if (response.IsSuccessStatusCode)
                    {
                        var health = await response.Content.ReadFromJsonAsync<HealthResponse>(cancellationToken: cancellationToken);
                        
                        if (health?.Status == "healthy")
                        {
                            Console.WriteLine("Python runtime is ready!");
                            Console.WriteLine($"Configuration: {health.Message}");
                            return true;
                        }
                        else
                        {
                            Console.WriteLine($"Runtime status: {health?.Status ?? "unknown"} - {health?.Message ?? "no message"}");
                        }
                    }
                }
                catch (HttpRequestException)
                {
                    // Expected during startup
                }
                catch (TaskCanceledException)
                {
                    // Timeout or cancellation
                    if (cancellationToken.IsCancellationRequested)
                    {
                        throw;
                    }
                }

                if (i < _config.MaxHealthCheckAttempts - 1)
                {
                    await Task.Delay(_config.HealthCheckIntervalMs, cancellationToken);
                }
            }

            Console.WriteLine("Timeout waiting for Python runtime to be ready.");
            return false;
        }

        /// <summary>
        /// Sends a chat message to the runtime and returns the response.
        /// </summary>
        public async Task<ChatResponse> SendChatAsync(string message, CancellationToken cancellationToken = default)
        {
            var request = new ChatRequest { Message = message };

            try
            {
                var response = await _httpClient.PostAsJsonAsync("/chat", request, cancellationToken);
                response.EnsureSuccessStatusCode();

                var chatResponse = await response.Content.ReadFromJsonAsync<ChatResponse>(cancellationToken: cancellationToken);
                
                if (chatResponse == null)
                {
                    throw new InvalidOperationException("Received null response from runtime");
                }

                return chatResponse;
            }
            catch (HttpRequestException ex)
            {
                throw new InvalidOperationException($"Failed to communicate with runtime: {ex.Message}", ex);
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _httpClient?.Dispose();
            _disposed = true;
        }
    }

    // Request/Response DTOs
    public class ChatRequest
    {
        [JsonPropertyName("message")]
        public string Message { get; set; } = string.Empty;

        [JsonPropertyName("additional_documents")]
        public List<string>? AdditionalDocuments { get; set; }
    }

    public class ChatResponse
    {
        [JsonPropertyName("answer")]
        public string Answer { get; set; } = string.Empty;

        [JsonPropertyName("sources")]
        public List<Source> Sources { get; set; } = new();
    }

    public class Source
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }

        [JsonPropertyName("source")]
        public string SourceName { get; set; } = string.Empty;

        [JsonPropertyName("page")]
        public string Page { get; set; } = string.Empty;
    }

    public class HealthResponse
    {
        [JsonPropertyName("status")]
        public string Status { get; set; } = string.Empty;

        [JsonPropertyName("message")]
        public string Message { get; set; } = string.Empty;

        [JsonPropertyName("config")]
        public Dictionary<string, object> Config { get; set; } = new();
    }
}
