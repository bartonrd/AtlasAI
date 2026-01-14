using System;
using System.IO;

namespace AtlasAI
{
    /// <summary>
    /// Configuration for the AtlasAI application.
    /// </summary>
    public class AppConfiguration
    {
        /// <summary>
        /// Gets or sets the Python executable path.
        /// </summary>
        public string PythonExecutable { get; set; } = "python";

        /// <summary>
        /// Gets or sets the runtime host.
        /// </summary>
        public string RuntimeHost { get; set; } = "127.0.0.1";

        /// <summary>
        /// Gets or sets the runtime port.
        /// </summary>
        public int RuntimePort { get; set; } = 8000;

        /// <summary>
        /// Gets the runtime base URL.
        /// </summary>
        public string RuntimeBaseUrl => $"http://{RuntimeHost}:{RuntimePort}";

        /// <summary>
        /// Gets or sets the maximum health check attempts.
        /// </summary>
        public int MaxHealthCheckAttempts { get; set; } = 30;

        /// <summary>
        /// Gets or sets the health check interval in milliseconds.
        /// </summary>
        public int HealthCheckIntervalMs { get; set; } = 2000;

        /// <summary>
        /// Gets or sets the request timeout in seconds.
        /// </summary>
        public int RequestTimeoutSeconds { get; set; } = 120;

        /// <summary>
        /// Gets the solution root directory.
        /// </summary>
        public string SolutionRoot
        {
            get
            {
                string? executableDir = AppContext.BaseDirectory;
                if (string.IsNullOrEmpty(executableDir))
                {
                    throw new InvalidOperationException("Could not determine application directory");
                }
                // From: AtlasAI/bin/Debug/net10.0/ to AtlasAI/ (up 3 levels)
                return Path.GetFullPath(Path.Combine(executableDir, "..", "..", "..", ".."));
            }
        }

        /// <summary>
        /// Loads configuration from environment variables or uses defaults.
        /// </summary>
        public static AppConfiguration Load()
        {
            var config = new AppConfiguration();

            // Load from environment variables if present
            string? pythonExe = Environment.GetEnvironmentVariable("ATLASAI_PYTHON_PATH");
            if (!string.IsNullOrEmpty(pythonExe))
            {
                config.PythonExecutable = pythonExe;
            }

            string? host = Environment.GetEnvironmentVariable("ATLASAI_RUNTIME_HOST");
            if (!string.IsNullOrEmpty(host))
            {
                config.RuntimeHost = host;
            }

            string? port = Environment.GetEnvironmentVariable("ATLASAI_RUNTIME_PORT");
            if (!string.IsNullOrEmpty(port) && int.TryParse(port, out int portNum))
            {
                config.RuntimePort = portNum;
            }

            return config;
        }
    }
}
