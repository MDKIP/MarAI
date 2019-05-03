using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MarAI.NeuralNetworks
{
    interface INeuralNetwork
    {
        long InputsNumber { get; }
        long OutputsNumber { get; }

        decimal[] Feedforward(decimal[] inputs);
        void Backpropagation(decimal[] inputs, decimal[] answers);
        void StartTraining(Dictionary<decimal[], decimal[]> dataset, long iterations);
        Task StartTrainingAsync(Dictionary<decimal[], decimal[]> dataset, long iterations);
    }
}
