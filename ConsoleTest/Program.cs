using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MarAI.Mathematics;
using MarAI.NeuralNetworks;
using MarAI.Visualization;

namespace ConsoleTest
{
    class Program
    {
        static void Main(string[] args)
        {
            TestOneHiddenLayerNeuralNetwork();
        }
        
        static void TestStandardPerceptron()
        {
            IPerceptron brain = new StandardPerceptron(2, 0.01m, Functions.Activation.BinaryStep, Functions.ActivationDerivatives.BinaryStep);

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
        static void TestOneHiddenLayerNeuralNetwork()
        {
            int h = 100;
            OneHiddenLayerNeuralNetwork brain = new OneHiddenLayerNeuralNetwork(2, h, 1, (x, y) => new StandardPerceptron(x == 0 ? 2 : h, 0.1m, Functions.Activation.Tanh, Functions.ActivationDerivatives.Tanh));

            Dictionary<decimal[], decimal[]> data = new Dictionary<decimal[], decimal[]>();
            data.Add(new decimal[] { 0, 0 }, new decimal[] { 0 });
            data.Add(new decimal[] { 1, 0 }, new decimal[] { 1 });
            data.Add(new decimal[] { 0, 1 }, new decimal[] { 1 });
            data.Add(new decimal[] { 1, 1 }, new decimal[] { 0 });

            Console.WriteLine("BEFORE TRAINING");
            Console.WriteLine($"0, 0 -> {brain.Feedforward(new decimal[] { 0, 0 })[0]}");
            Console.WriteLine($"1, 0 -> {brain.Feedforward(new decimal[] { 1, 0 })[0]}");
            Console.WriteLine($"0, 1 -> {brain.Feedforward(new decimal[] { 0, 1 })[0]}");
            Console.WriteLine($"1, 1 -> {brain.Feedforward(new decimal[] { 1, 1 })[0]}");

            //brain.Backpropagation(new decimal[] { 1, 1 }, new decimal[] { 0 });
            
            for (int i = 0; i < 10; i++)
            {
                brain.StartTraining(data, 10000);

                Console.WriteLine("AFTER TRAINING");
                Console.WriteLine($"0, 0 -> {brain.Feedforward(new decimal[] { 0, 0 })[0]}");
                Console.WriteLine($"1, 0 -> {brain.Feedforward(new decimal[] { 1, 0 })[0]}");
                Console.WriteLine($"0, 1 -> {brain.Feedforward(new decimal[] { 0, 1 })[0]}");
                Console.WriteLine($"1, 1 -> {brain.Feedforward(new decimal[] { 1, 1 })[0]}"); 
            }
            
            
            Console.ReadKey();
        }
        static void TestSingleLayerNeuralNetwork()
        {
            SingleLayerNeuralNetwork brain = new SingleLayerNeuralNetwork(2, 4, (y) => new StandardPerceptron(2, 0.01m, Functions.Activation.Tanh, Functions.ActivationDerivatives.Tanh));

            Dictionary<decimal[], decimal[]> data = new Dictionary<decimal[], decimal[]>();
            data.Add(new decimal[] { 0, 0 }, new decimal[] { 0, 1, 0, 1 });
            data.Add(new decimal[] { 1, 0 }, new decimal[] { 1, 0, 0, 1 });
            data.Add(new decimal[] { 0, 1 }, new decimal[] { 1, 0, 0, 1 });
            data.Add(new decimal[] { 1, 1 }, new decimal[] { 1, 0, 1, 0 });

            Console.WriteLine("BEFORE TRAINING");
            decimal[] o = brain.Feedforward(new decimal[] { 0, 0 });
            Console.WriteLine($"0, 0 -> {o[0]}, {o[1]}, {o[2]}, {o[3]}");
            o = brain.Feedforward(new decimal[] { 0, 1 });
            Console.WriteLine($"0, 1 -> {o[0]}, {o[1]}, {o[2]}, {o[3]}");
            o = brain.Feedforward(new decimal[] { 1, 0 });
            Console.WriteLine($"1, 0 -> {o[0]}, {o[1]}, {o[2]}, {o[3]}");
            o = brain.Feedforward(new decimal[] { 1, 1});
            Console.WriteLine($"1, 1 -> {o[0]}, {o[1]}, {o[2]}, {o[3]}");

            //brain.Backpropagation(new decimal[] { 1, 1 }, new decimal[] { 0 });

            for (int i = 0; i < 10; i++)
            {
                brain.StartTraining(data, 10000);

                Console.WriteLine("AFTER TRAINING");
                o = brain.Feedforward(new decimal[] { 0, 0 });
                Console.WriteLine($"0, 0 -> {o[0]}, {o[1]}, {o[2]}, {o[3]}");
                o = brain.Feedforward(new decimal[] { 0, 1 });
                Console.WriteLine($"0, 1 -> {o[0]}, {o[1]}, {o[2]}, {o[3]}");
                o = brain.Feedforward(new decimal[] { 1, 0 });
                Console.WriteLine($"1, 0 -> {o[0]}, {o[1]}, {o[2]}, {o[3]}");
                o = brain.Feedforward(new decimal[] { 1, 1 });
                Console.WriteLine($"1, 1 -> {o[0]}, {o[1]}, {o[2]}, {o[3]}");
            }


            Console.ReadKey();
        }
        static void TestPerceptronParametersDirectVisualization()
        {
            IPerceptron perceptron = new StandardPerceptron(12, 0, null, null);
            foreach (decimal w in perceptron.Weights)
            {
                Console.WriteLine(w);
            }

            IDirectParametersVisualizer visualizer = new PerceptronDirectParametersVisualizer(perceptron, DirectParametersVisualizationType.SquareOfRGB);
            Bitmap img = visualizer.Visualize();
            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    Color c = img.GetPixel(x, y);
                    Console.WriteLine($"{x}, {y} = [{c.R}, {c.G}, {c.B}]");
                }
            }

            Console.ReadKey();
        }
    }
}
