using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using MarAI.Mathematics;

namespace MarAI.NeuralNetworks
{
    public interface IPerceptron
    {
        DecimalOperation ActivationFunction { get; }
        DecimalOperation ActivationDerivative { get; }
        decimal[] Weights { get; }
        decimal Bias { get; }
        decimal LearningRate { get; set; }
        long InputsNumber { get; }

        decimal CalculateOutput(decimal[] inputs);
        decimal Train(decimal[] inputs, decimal answer);
        void AdjustParametersByError(decimal[] inputs, decimal error);
        void StartTraining(Dictionary<decimal[], decimal> dataset, long iterations);
        Task StartTrainingAsync(Dictionary<decimal[], decimal> dataset, long iterations);
    }
}
