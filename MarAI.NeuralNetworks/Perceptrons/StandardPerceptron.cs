using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Threading.Tasks;
using MarAI.Mathematics;

namespace MarAI.NeuralNetworks
{
    public class StandardPerceptron : IPerceptron
    {
        public StandardPerceptron(long inputsNumber, decimal learningRate, DecimalOperation activationFunction, DecimalOperation activationDerivative)
        {
            InputsNumber = inputsNumber;
            LearningRate = learningRate;
            ActivationFunction = activationFunction;
            ActivationDerivative = activationDerivative;

            Weights = new decimal[InputsNumber];
            for (long i = 0; i < InputsNumber; i++)
            {
                Weights[i] = MathHelper.GetRandomDecimal() * 2 - 1;
            }
            Bias = MathHelper.GetRandomDecimal() * 2 -1;
        }

        public DecimalOperation ActivationFunction { get; private set; }
        public DecimalOperation ActivationDerivative { get; private set; }
        public decimal[] Weights { get; private set; }
        public decimal Bias { get; private set; }
        public decimal LearningRate { get; set; }
        public long InputsNumber { get; private set; }

        public void AdjustParametersByError(decimal[] inputs, decimal error)
        {
            for (long i = 0; i < InputsNumber; i++)
            {
                Weights[i] += inputs[i] * error * LearningRate;
            }
            Bias += error * LearningRate;
        }
        public decimal CalculateOutput(decimal[] inputs)
        {
            decimal output = 0;
            for (long i = 0; i < InputsNumber; i++)
            {
                output += inputs[i] * Weights[i];
            }
            output += Bias;
            output = ActivationFunction.Invoke(output);
            return output;
        }
        public void StartTraining(Dictionary<decimal[], decimal> dataset, long iterations)
        {
            int l = dataset.Count;
            for (long i = 0; i < iterations; i++)
            {
                var data = dataset.ElementAt(MathHelper.GetRandomInt(l));
                Train(data.Key, data.Value);
            }
        }
        public async Task StartTrainingAsync(Dictionary<decimal[], decimal> dataset, long iterations)
        {
            Task.Run(() => StartTraining(dataset, iterations));
        }
        public decimal Train(decimal[] inputs, decimal answer)
        {
            decimal output = CalculateOutput(inputs);
            decimal error = answer - output;
            AdjustParametersByError(inputs, error);
            return error;
        }
    }
}
