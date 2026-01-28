using System;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace AtlasAI
{
    /// <summary>
    /// Manages the Python runtime process lifecycle.
    /// </summary>
    public class PythonRuntimeManager : IDisposable
    {
        private readonly AppConfiguration _config;
        private Process? _process;
        private bool _disposed;

        public PythonRuntimeManager(AppConfiguration config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Starts the Python runtime process.
        /// </summary>
        public void Start()
        {
            if (_process != null)
            {
                Console.WriteLine("Python runtime is already running.");
                return;
            }

            // Run setup script to install requirements and check Ollama
            Console.WriteLine("Running environment setup...");
            RunSetupScript();
            Console.WriteLine();

            Console.WriteLine("Starting Python runtime...");
            Console.WriteLine($"Host: {_config.RuntimeHost}");
            Console.WriteLine($"Port: {_config.RuntimePort}");
            Console.WriteLine();

            // Construct path to the Python module
            string solutionRoot = _config.SolutionRoot;
            
            // Configure the process to run the Python runtime module
            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = _config.PythonExecutable,
                Arguments = $"-m atlasai_runtime --host {_config.RuntimeHost} --port {_config.RuntimePort}",
                WorkingDirectory = solutionRoot,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = false
            };

            try
            {
                _process = Process.Start(startInfo);
                
                if (_process == null)
                {
                    throw new InvalidOperationException("Failed to start Python runtime process");
                }

                // Set up output handlers
                _process.OutputDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        Console.WriteLine($"[Runtime] {e.Data}");
                    }
                };

                _process.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        // Python logging and uvicorn write INFO/DEBUG logs to stderr by default
                        // Only label as error if it actually indicates an error
                        string prefix = IsErrorMessage(e.Data) ? "[Runtime Error] " : "[Runtime] ";
                        Console.WriteLine($"{prefix}{e.Data}");
                    }
                };

                _process.BeginOutputReadLine();
                _process.BeginErrorReadLine();

                Console.WriteLine("Python runtime process started.");
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to start Python runtime: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Stops the Python runtime process.
        /// </summary>
        public void Stop()
        {
            if (_process == null || _process.HasExited)
            {
                Console.WriteLine("Python runtime is not running.");
                return;
            }

            Console.WriteLine("Stopping Python runtime...");

            try
            {
                // Try graceful shutdown first
                _process.CancelOutputRead();
                _process.CancelErrorRead();

                if (!_process.HasExited)
                {
                    _process.Kill(entireProcessTree: true);
                    _process.WaitForExit(5000); // Wait up to 5 seconds
                }

                Console.WriteLine("Python runtime stopped.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Error stopping Python runtime: {ex.Message}");
            }
            finally
            {
                _process?.Dispose();
                _process = null;
            }
        }

        /// <summary>
        /// Checks if the runtime process is still running.
        /// </summary>
        public bool IsRunning => _process != null && !_process.HasExited;

        /// <summary>
        /// Determines if a log message indicates an actual error.
        /// Python logging and uvicorn write INFO/DEBUG logs to stderr by default.
        /// </summary>
        private static bool IsErrorMessage(string message)
        {
            if (string.IsNullOrEmpty(message))
            {
                return false;
            }

            // Check for actual error indicators using case-insensitive comparison
            var comparison = StringComparison.OrdinalIgnoreCase;
            
            // Explicit error keywords
            if (message.Contains("error", comparison) || 
                message.Contains("exception", comparison) || 
                message.Contains("traceback", comparison) ||
                message.Contains("failed", comparison) ||
                message.Contains("fatal", comparison))
            {
                // But exclude INFO/DEBUG level messages that happen to contain "error" in them
                if (message.Contains("info:", comparison) || 
                    message.Contains("debug:", comparison) ||
                    message.Contains("- info -", comparison) ||
                    message.Contains("- debug -", comparison))
                {
                    return false;
                }
                return true;
            }

            // Check for WARNING level (these should be marked but not as errors)
            if (message.Contains("warning", comparison) || message.Contains("- warning -", comparison))
            {
                return false; // Warnings are not errors
            }

            return false;
        }

        /// <summary>
        /// Runs the setup script to install requirements and check environment.
        /// </summary>
        private void RunSetupScript()
        {
            try
            {
                string solutionRoot = _config.SolutionRoot;
                string setupScript = Path.Combine(solutionRoot, "setup_env.py");

                if (!File.Exists(setupScript))
                {
                    Console.WriteLine("Warning: setup_env.py not found, skipping environment setup");
                    return;
                }

                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = _config.PythonExecutable,
                    Arguments = $"\"{setupScript}\"",
                    WorkingDirectory = solutionRoot,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using var setupProcess = Process.Start(startInfo);
                if (setupProcess == null)
                {
                    Console.WriteLine("Warning: Failed to start setup process");
                    return;
                }

                // Read output
                string output = setupProcess.StandardOutput.ReadToEnd();
                string error = setupProcess.StandardError.ReadToEnd();
                
                setupProcess.WaitForExit();

                // Display output
                if (!string.IsNullOrEmpty(output))
                {
                    Console.WriteLine(output);
                }
                
                if (!string.IsNullOrEmpty(error))
                {
                    Console.WriteLine(error);
                }

                if (setupProcess.ExitCode != 0)
                {
                    Console.WriteLine($"Warning: Setup script exited with code {setupProcess.ExitCode}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Error running setup script: {ex.Message}");
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            Stop();
            _disposed = true;
        }
    }
}
