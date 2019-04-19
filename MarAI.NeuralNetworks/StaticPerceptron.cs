using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using MarAI.Mathematics;

namespace MarAI.NeuralNetworks
{
    public class StaticPerceptron
    {
        public StaticPerceptron(int inputsNumber, DecimalOperation activationFunction, decimal learningRate)
        {
            InputsNumber = inputsNumber;
            ActivationFunction = activationFunction;
            LearningRate = learningRate;
            Weights = new decimal[InputsNumber];
        }

        public DecimalOperation ActivationFunction { get; private set; }
        public decimal[] Weights { get; private set; }
        public decimal LearningRate { get; private set; }
        public decimal Bias { get; private set; }
        public int InputsNumber { get; private set; }

        public decimal CalculateOutput(decimal[] inputs)
        {
            decimal output = 0;
            for (int i = 0; i < InputsNumber; i++)
            {
                output += inputs[i] * Weights[i];
            }
            output += Bias;
            output = ActivationFunction.Invoke(output);
            return output;
        }
        public void Train(decimal[] inputs, decimal answer)
        {
            decimal output = CalculateOutput(inputs);
            decimal error = answer - output;
            AdjustParametersByError(inputs, error);
        }
        public void AdjustParametersByError(decimal[] inputs, decimal error)
        {
            Bias += error * LearningRate;
            for (int i = 0; i < InputsNumber; i++)
            {
                Weights[i] += inputs[i] * error * LearningRate;
            }
        }
        public void StartTraining(Dictionary<decimal[], decimal> inputsAnswers, int iterations)
        {
            for (int i = 0; i < iterations; i++)
            {
                var pair = inputsAnswers.ElementAt(MathHelper.GetRandomInt(inputsAnswers.Count));
                Train(pair.Key, pair.Value);
            }
        }
    }
}
