using System.Diagnostics;

namespace AtlasAI
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("   AtlasAI - LLM Chatbot with RAG");
            Console.WriteLine("===========================================");
            Console.WriteLine();
            Console.WriteLine("Starting the chatbot application...");
            Console.WriteLine();

            try
            {
                // Get the directory where the executable is running
                string? executableDir = AppContext.BaseDirectory;
                
                if (string.IsNullOrEmpty(executableDir))
                {
                    Console.WriteLine("ERROR: Could not determine application directory.");
                    Console.WriteLine("Press any key to exit...");
                    Console.ReadKey();
                    return;
                }
                
                // Navigate up to the solution root (where chatapp.py is located)
                // From: AtlasAI/bin/Debug/net10.0/ to AtlasAI/ (up 3 levels)
                string solutionRoot = Path.GetFullPath(Path.Combine(executableDir, "..", "..", "..", ".."));
                string pythonScriptPath = Path.Combine(solutionRoot, "chatapp.py");

                if (!File.Exists(pythonScriptPath))
                {
                    Console.WriteLine($"ERROR: Python script not found at: {pythonScriptPath}");
                    Console.WriteLine("Press any key to exit...");
                    Console.ReadKey();
                    return;
                }

                Console.WriteLine($"Script location: {pythonScriptPath}");
                Console.WriteLine($"Working directory: {solutionRoot}");
                Console.WriteLine();
                Console.WriteLine("Launching Streamlit chatbot...");
                Console.WriteLine("The chatbot UI will open in your default web browser.");
                Console.WriteLine();
                Console.WriteLine("Press Ctrl+C to stop the application.");
                Console.WriteLine("===========================================");
                Console.WriteLine();

                // Configure the process to run streamlit
                // Use "python -m streamlit" to avoid PATH issues with streamlit executable
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"-m streamlit run \"{pythonScriptPath}\"",
                    WorkingDirectory = solutionRoot,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = false
                };

                using (Process? process = Process.Start(startInfo))
                {
                    if (process == null)
                    {
                        Console.WriteLine("ERROR: Failed to start the chatbot process.");
                        return;
                    }

                    // Read output asynchronously
                    process.OutputDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            Console.WriteLine(e.Data);
                        }
                    };

                    process.ErrorDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            Console.WriteLine($"ERROR: {e.Data}");
                        }
                    };

                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();

                    // Wait for the process to exit
                    process.WaitForExit();

                    Console.WriteLine();
                    Console.WriteLine("Chatbot application has stopped.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: An exception occurred: {ex.Message}");
                Console.WriteLine();
                Console.WriteLine("Make sure you have:");
                Console.WriteLine("1. Python installed and in your PATH");
                Console.WriteLine("2. Streamlit installed (pip install streamlit)");
                Console.WriteLine("3. All required Python packages installed");
                Console.WriteLine();
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
            }
        }
    }
}
