using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MarAI.Mathematics;

namespace MarAI.NeuralNetworks
{
    public class OneHiddenLayerNetwork
    {
        public OneHiddenLayerNetwork(int inputSize, int hiddenSize, int outputSize, DecimalOperation activationFunction, DecimalOperation derivativeFunction, decimal learningRate)
        {
            InputSize = inputSize;
            HiddenSize = hiddenSize;
            OutputSize = outputSize;
            ActivationFunction = activationFunction;
            DerivativeFunction = derivativeFunction;
            LearningRate = learningRate;
            LayersSizes = new int[3] { inputSize, hiddenSize, outputSize };

            Weights = new decimal[2][][];
            Weights[0] = new decimal[hiddenSize][];
            Weights[1] = new decimal[outputSize][];
            for (int i = 0; i < hiddenSize; i++)
            {
                Weights[0][i] = new decimal[inputSize];
            }
            for (int i = 0; i < outputSize; i++)
            {
                Weights[1][i] = new decimal[hiddenSize];
            }
            for (int layer = 0; layer < 2; layer++)
            {
                for (int perceptron = 0; perceptron < LayersSizes[layer+1]; perceptron++)
                {
                    for (int weight = 0; weight < LayersSizes[layer]; weight++)
                    {
                        Weights[layer][perceptron][weight] = MathHelper.GetRandomDecimal();
                    }
                }
            }

            Biases = new decimal[2][];
            Biases[0] = new decimal[hiddenSize];
            Biases[1] = new decimal[outputSize];
        }

        public DecimalOperation ActivationFunction { get; private set; }
        public DecimalOperation DerivativeFunction { get; private set; }
        public decimal[][][] Weights { get; private set; }
        public decimal[][] Biases { get; private set; }
        public decimal LearningRate { get; set; }
        public int[] LayersSizes { get; private set;}
        public int InputSize { get; private set; }
        public int HiddenSize { get; private set; }
        public int OutputSize { get; private set; }

        public decimal[] CalculateOutput(decimal[] inputs)
        {
            decimal[][] outputs = new decimal[3][];
            outputs[0] = inputs;

            for (int layer = 0; layer < 2; layer++)
            {
                int currentLayer = layer + 1;
                int earlierLayer = layer;
                int currentLayerSize = LayersSizes[currentLayer];
                int earlierLayerSize = LayersSizes[layer];
                outputs[currentLayer] = new decimal[currentLayerSize];

                for (int perceptron = 0; perceptron < currentLayerSize; perceptron++)
                {
                    decimal perceptronOutput = 0;
                    for (int weight = 0; weight < earlierLayerSize; weight++)
                    {
                        perceptronOutput += outputs[earlierLayer][weight] * Weights[layer][perceptron][weight];
                    }
                    perceptronOutput += Biases[layer][perceptron];
                    perceptronOutput = ActivationFunction.Invoke(perceptronOutput);
                    outputs[currentLayer][perceptron] = perceptronOutput;
                }
            }
            return outputs[2];
        }
        public void Backpropagation(decimal[] inputs, decimal[] answers)
        {
            // FEEDFORWARD
            decimal[][] outputs = new decimal[3][];
            outputs[0] = inputs;
            for (int layer = 0; layer < 2; layer++)
            {
                int currentLayer = layer + 1;
                int earlierLayer = layer;
                int currentLayerSize = LayersSizes[currentLayer];
                int earlierLayerSize = LayersSizes[layer];
                outputs[currentLayer] = new decimal[currentLayerSize];

                for (int perceptron = 0; perceptron < currentLayerSize; perceptron++)
                {
                    decimal perceptronOutput = 0;
                    for (int weight = 0; weight < earlierLayerSize; weight++)
                    {
                        perceptronOutput += outputs[earlierLayer][weight] * Weights[layer][perceptron][weight];
                    }
                    perceptronOutput += Biases[layer][perceptron];
                    perceptronOutput = ActivationFunction.Invoke(perceptronOutput);
                    outputs[currentLayer][perceptron] = perceptronOutput;
                }
            }

            // BACKPROPAGATION
            // Calculating Errors
            decimal[][] errors = new decimal[2][];
            errors[0] = new decimal[HiddenSize];
            errors[1] = new decimal[OutputSize];
            // output error
            for (int i = 0; i < OutputSize; i++)
            {
                errors[1][i] = answers[i] - outputs[2][i];
            }
            // hidden error
            for (int perceptron = 0; perceptron < HiddenSize; perceptron++)
            {
                decimal sum = 0;
                for (int outputPerceptron = 0; outputPerceptron < OutputSize; outputPerceptron++)
                {
                    sum += errors[1][outputPerceptron] * Weights[1][outputPerceptron][perceptron];
                }
                sum *= outputs[1][perceptron] * (1 - outputs[1][perceptron]);
                errors[0][perceptron] = sum;
            }

            // Bias Change
            for (int layer = 0; layer < 2; layer++)
            {
                for (int perceptron = 0; perceptron < LayersSizes[layer + 1]; perceptron++)
                {
                    Biases[layer][perceptron] += errors[layer][perceptron] * LearningRate;
                }
            }

            // Weights Change
            for (int layer = 0; layer < 2; layer++)
            {
                for (int perceptron = 0; perceptron < LayersSizes[layer + 1]; perceptron++)
                {
                    for (int weight = 0; weight < LayersSizes[layer]; weight++)
                    {
                        Weights[layer][perceptron][weight] += outputs[layer][weight] * errors[layer][perceptron] * LearningRate;
                    }
                }
            }
        }
        public void StartTraining(Dictionary<decimal[], decimal[]> inputsAnswers, int iterations)
        {
            for (int i = 0; i < iterations; i++)
            {
                var pair = inputsAnswers.ElementAt(MathHelper.GetRandomInt(inputsAnswers.Count));
                Backpropagation(pair.Key, pair.Value);
            }
        }
    }
}
