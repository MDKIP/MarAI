using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MarAI.Mathematics;
using MarAI.NeuralNetworks;

namespace ConsoleTest
{
    class Program
    {
        static void Main(string[] args)
        {
            TestNeuralNetwork();
        }
        static void TestPerceptron()
        {
            StaticPerceptron brain = new StaticPerceptron(2, Functions.Activation.Identity, 0.01m);

            Dictionary<decimal[], decimal> data = new Dictionary<decimal[], decimal>();
            data.Add(new decimal[] { 0, 0 }, 0);
            data.Add(new decimal[] { 1, 0 }, 0);
            data.Add(new decimal[] { 0, 1 }, 0);
            data.Add(new decimal[] { 1, 1 }, 1);

            Console.WriteLine("BEFORE TRAINING");
            Console.WriteLine($"0, 0 -> {brain.CalculateOutput(new decimal[] { 0, 0 })}");
            Console.WriteLine($"1, 0 -> {brain.CalculateOutput(new decimal[] { 1, 0 })}");
            Console.WriteLine($"0, 1 -> {brain.CalculateOutput(new decimal[] { 0, 1 })}");
            Console.WriteLine($"1, 1 -> {brain.CalculateOutput(new decimal[] { 1, 1 })}");

            brain.StartTraining(data, 1000);

            Console.WriteLine("AFTER TRAINING");
            Console.WriteLine($"0, 0 -> {brain.CalculateOutput(new decimal[] { 0, 0 })}");
            Console.WriteLine($"1, 0 -> {brain.CalculateOutput(new decimal[] { 1, 0 })}");
            Console.WriteLine($"0, 1 -> {brain.CalculateOutput(new decimal[] { 0, 1 })}");
            Console.WriteLine($"1, 1 -> {brain.CalculateOutput(new decimal[] { 1, 1 })}");

            Console.ReadKey();
        }
        static void TestNeuralNetwork()
        {
            OneHiddenLayerNetwork brain = new OneHiddenLayerNetwork(2, 2, 1, Functions.Activation.BinaryStep, Functions.ActivationDerivatives.Identity, 0.001m);

            Dictionary<decimal[], decimal[]> data = new Dictionary<decimal[], decimal[]>();
            data.Add(new decimal[] { 0, 0 }, new decimal[] { 0 });
            data.Add(new decimal[] { 1, 0 }, new decimal[] { 1 });
            data.Add(new decimal[] { 0, 1 }, new decimal[] { 1 });
            data.Add(new decimal[] { 1, 1 }, new decimal[] { 0 });

            Console.WriteLine("BEFORE TRAINING");
            Console.WriteLine($"0, 0 -> {brain.CalculateOutput(new decimal[] { 0, 0 })[0]}");
            Console.WriteLine($"1, 0 -> {brain.CalculateOutput(new decimal[] { 1, 0 })[0]}");
            Console.WriteLine($"0, 1 -> {brain.CalculateOutput(new decimal[] { 0, 1 })[0]}");
            Console.WriteLine($"1, 1 -> {brain.CalculateOutput(new decimal[] { 1, 1 })[0]}");

            for (int i = 0; i < 10; i++)
            {
                brain.StartTraining(data, 1000);

                Console.WriteLine("AFTER TRAINING");
                Console.WriteLine($"0, 0 -> {brain.CalculateOutput(new decimal[] { 0, 0 })[0]}");
                Console.WriteLine($"1, 0 -> {brain.CalculateOutput(new decimal[] { 1, 0 })[0]}");
                Console.WriteLine($"0, 1 -> {brain.CalculateOutput(new decimal[] { 0, 1 })[0]}");
                Console.WriteLine($"1, 1 -> {brain.CalculateOutput(new decimal[] { 1, 1 })[0]}"); 
            }

            Console.ReadKey();
        }
    }
}
