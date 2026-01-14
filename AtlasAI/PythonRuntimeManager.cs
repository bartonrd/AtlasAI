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
                        Console.WriteLine($"[Runtime Error] {e.Data}");
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
