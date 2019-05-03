using MarAI.Mathematics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MarAI.NeuralNetworks
{
    public class OneHiddenLayerNeuralNetwork : ILayeredNeuralNetwork
    {
        public OneHiddenLayerNeuralNetwork(long inputsNumber, long hiddensNumber, long outputsnumber, SecondDimensionUnitBuilder<IPerceptron> builder)
        {
            InputsNumber = inputsNumber;
            HiddensNumber = hiddensNumber;
            OutputsNumber = outputsnumber;

            HiddenLayer = new PerceptronLayer(HiddensNumber, y => builder.Invoke(0, y));
            OutputLayer = new PerceptronLayer(OutputsNumber, y => builder.Invoke(1, y));
            Layers = new object[2];
            Layers[0] = HiddenLayer;
            Layers[1] = OutputLayer;
        }

        public object[] Layers { get; private set; }
        public PerceptronLayer HiddenLayer { get; private set; }
        public PerceptronLayer OutputLayer { get; private set; }
        public long InputsNumber { get; private set; }
        public long HiddensNumber { get; private set; }
        public long OutputsNumber { get; private set; }

        public decimal[] Feedforward(decimal[] inputs)
        {
            decimal[] hiddenOutputs = HiddenLayer.CalculateOutputs(inputs);
            decimal[] outputOutputs = OutputLayer.CalculateOutputs(hiddenOutputs);
            return outputOutputs;
        }
        public void Backpropagation(decimal[] inputs, decimal[] answers)
        {
            // FEEDFORWARD
            decimal[] hiddenOutputs = HiddenLayer.CalculateOutputs(inputs);
            decimal[] outputOutputs = OutputLayer.CalculateOutputs(hiddenOutputs);

            // BACKPROPAGATION
            decimal[] outputErros = OutputLayer.CalculateErrors(outputOutputs, answers);
            decimal[] outputDerivatives = OutputLayer.CalculateDerivatives(outputOutputs);
            decimal[] outputGradients = OutputLayer.CalculateGradients(outputErros, outputDerivatives);

            decimal[] hiddenDerivatives = HiddenLayer.CalculateDerivatives(hiddenOutputs);
            decimal[] hiddenGradients = HiddenLayer.CalculateGradients(OutputLayer, outputGradients, hiddenDerivatives);

            OutputLayer.Adjust(hiddenOutputs, outputGradients);
            HiddenLayer.Adjust(inputs, hiddenGradients);
        }
        public void StartTraining(Dictionary<decimal[], decimal[]> dataset, long iterations)
        {
            int l = dataset.Count;
            for (long i = 0; i < iterations; i++)
            {
                var data = dataset.ElementAt(MathHelper.GetRandomInt(l));
                Backpropagation(data.Key, data.Value);
            }
        }
        public async Task StartTrainingAsync(Dictionary<decimal[], decimal[]> dataset, long iterations)
        {
            Task.Run(() => StartTraining(dataset, iterations));
        }
    }
}
