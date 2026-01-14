using System;
using System.Threading;
using System.Threading.Tasks;

namespace AtlasAI
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("   AtlasAI - LLM Chatbot with RAG");
            Console.WriteLine("===========================================");
            Console.WriteLine();

            // Load configuration
            var config = AppConfiguration.Load();

            // Set up cancellation for Ctrl+C
            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (sender, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
            };

            try
            {
                // Start Python runtime
                using var runtimeManager = new PythonRuntimeManager(config);
                runtimeManager.Start();

                // Wait for runtime to be ready
                using var client = new RuntimeClient(config);
                bool isReady = await client.WaitForHealthyAsync(cts.Token);

                if (!isReady)
                {
                    Console.WriteLine("ERROR: Python runtime failed to start or become healthy.");
                    Console.WriteLine();
                    Console.WriteLine("Make sure you have:");
                    Console.WriteLine("1. Python installed and in your PATH");
                    Console.WriteLine("2. Required Python packages installed (pip install -r requirements.txt)");
                    Console.WriteLine("3. ML models downloaded to the configured paths");
                    Console.WriteLine();
                    return 1;
                }

                Console.WriteLine();
                Console.WriteLine("===========================================");
                Console.WriteLine("AtlasAI is ready! You can now ask questions.");
                Console.WriteLine("Press Ctrl+C to exit.");
                Console.WriteLine("===========================================");
                Console.WriteLine();

                // Interactive chat loop
                while (!cts.Token.IsCancellationRequested)
                {
                    Console.Write("You: ");
                    string? input = Console.ReadLine();

                    if (string.IsNullOrWhiteSpace(input))
                    {
                        continue;
                    }

                    if (input.Equals("exit", StringComparison.OrdinalIgnoreCase) ||
                        input.Equals("quit", StringComparison.OrdinalIgnoreCase))
                    {
                        break;
                    }

                    try
                    {
                        Console.WriteLine();
                        Console.WriteLine("Processing...");

                        var response = await client.SendChatAsync(input, cts.Token);

                        Console.WriteLine();
                        Console.WriteLine("Assistant:");
                        Console.WriteLine(response.Answer);

                        if (response.Sources.Count > 0)
                        {
                            Console.WriteLine();
                            Console.WriteLine("Sources:");
                            foreach (var source in response.Sources)
                            {
                                Console.WriteLine($"  {source.Index}. {source.SourceName} (page {source.Page})");
                            }
                        }

                        Console.WriteLine();
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine();
                        Console.WriteLine($"ERROR: {ex.Message}");
                        Console.WriteLine();
                    }
                }

                Console.WriteLine();
                Console.WriteLine("Shutting down...");
                return 0;
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine();
                Console.WriteLine("Shutting down...");
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: An exception occurred: {ex.Message}");
                Console.WriteLine();
                Console.WriteLine("Make sure you have:");
                Console.WriteLine("1. Python installed and in your PATH");
                Console.WriteLine("2. Required Python packages installed (pip install -r requirements.txt)");
                Console.WriteLine("3. ML models downloaded to the configured paths");
                Console.WriteLine();
                return 1;
            }
        }
    }
}
